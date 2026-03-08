"""
agents/layout_detection_agent.py
----------------------------------
Agent 2 — Layout Detection Agent

Consumes: DocumentExtractionResult  (output of Agent 1)
Produces: DocumentLayoutResult       (input for Agent 3)

Responsibilities
----------------
1. Select the best available detection backend:
      LayoutParser > DiT > Heuristic (auto-detected at runtime)
2. Render each page to an image (needed for ML backends)
3. Run detection: ML model → raw LayoutRegions R = {(cj, rj, pj, γj)}
4. Fuse ML predictions with heuristic refinements
5. Enrich regions with text content from Step 2 text blocks
6. Serialize output to JSON

Paper reference (Section IV-B / Step 2 in the pipeline):
  "layout-aware models like LayoutLM and associated multimodal
   transformers integrate spatial features … for entity extraction
   and document classification."
  "Each region gets a class label + confidence score γj."

Backend selection (automatic)
------------------------------
  priority 1 → LayoutParser (if `layoutparser` importable)
  priority 2 → DiT           (if `transformers` importable)
  priority 3 → Heuristic     (always available, zero deps)

Force a backend by passing backend="layoutparser"|"dit"|"heuristic".
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Literal, Optional

from extractors.models import DocumentExtractionResult, PageExtractionResult
from extractors.layout_models import (
    DocumentLayoutResult, PageLayoutResult, LayoutRegion,
    RegionClass, DetectionBackend,
)
from extractors.heuristic_layout_detector import HeuristicLayoutDetector
from extractors.ml_layout_detector import (
    LayoutParserDetector, DiTLayoutDetector, fuse_ml_and_heuristic,
)

logger = logging.getLogger(__name__)

BackendChoice = Literal["auto", "layoutparser", "dit", "heuristic"]

RENDER_DPI = 150   # DPI for rendering pages (ML backends need images)


def _try_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


class LayoutDetectionAgent:
    """
    Agent 2: Layout Detection

    Parameters
    ----------
    backend        : "auto" | "layoutparser" | "dit" | "heuristic"
    output_dir     : root directory for JSON outputs
    render_dpi     : DPI for page rasterisation (ML backends)
    score_threshold: minimum confidence to keep an ML detection
    iou_threshold  : IoU threshold for heuristic/ML fusion
    verbose        : enable debug logging
    """

    AGENT_NAME = "LayoutDetectionAgent"

    def __init__(
        self,
        backend:         BackendChoice = "auto",
        output_dir:      str           = "output",
        render_dpi:      int           = RENDER_DPI,
        score_threshold: float         = 0.70,
        iou_threshold:   float         = 0.30,
        verbose:         bool          = False,
    ):
        self.output_dir      = Path(output_dir)
        self.render_dpi      = render_dpi
        self.score_threshold = score_threshold
        self.iou_threshold   = iou_threshold

        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            level=level)

        # Resolve backend
        self._backend_name = self._resolve_backend(backend)
        logger.info(f"[{self.AGENT_NAME}] Using backend: {self._backend_name}")

        # Instantiate detectors
        self._heuristic = HeuristicLayoutDetector(
            confidence_text  = 0.80,
            confidence_image = 0.95,
        )
        self._ml_detector = None
        if self._backend_name == "layoutparser":
            self._ml_detector = LayoutParserDetector(score_threshold=score_threshold)
        elif self._backend_name == "dit":
            self._ml_detector = DiTLayoutDetector(score_threshold=score_threshold)

    # ── Public API ─────────────────────────────────────────

    def run(
        self,
        extraction: DocumentExtractionResult,
        pdf_path:   Optional[str] = None,
    ) -> DocumentLayoutResult:
        """
        Run layout detection on a fully extracted document.

        Parameters
        ----------
        extraction : DocumentExtractionResult from Agent 1
        pdf_path   : path to original PDF (required for ML backends
                     that need to render page images)

        Returns
        -------
        DocumentLayoutResult — R = {(cj, rj, pj, γj)} for all pages
        """
        doc_id   = extraction.doc_id
        self._prepare_output_dirs(doc_id)

        # ML backends need rendered page images
        pdf_doc = None
        if self._ml_detector and pdf_path:
            pdf_doc = self._open_pdf(pdf_path)

        pages: list[PageLayoutResult] = []

        for page in extraction.pages:
            logger.info(
                f"  Layout detection: page {page.page_index + 1}/{extraction.total_pages}..."
            )
            page_result = self._detect_page(
                page            = page,
                doc_id          = doc_id,
                page_idx_in_doc = page.page_index,
                pdf_doc         = pdf_doc,
            )
            pages.append(page_result)

            # Log per-page summary
            cls_counts = {}
            for r in page_result.regions:
                cls_counts[r.region_class.value] = cls_counts.get(r.region_class.value, 0) + 1
            logger.info(f"    → {len(page_result.regions)} regions: {cls_counts}")

        if pdf_doc:
            pdf_doc.close()

        result = DocumentLayoutResult(
            doc_id      = doc_id,
            source_path = extraction.source_path,
            total_pages = extraction.total_pages,
            pages       = pages,
        )

        stats = result.stats()
        logger.info(
            f"[{self.AGENT_NAME}] Complete — "
            f"{stats['total_regions']} regions across {stats['total_pages']} pages"
        )
        logger.info(f"  Region breakdown: {stats['by_class']}")

        self._save_json(result, doc_id)
        return result

    # ── Internal helpers ───────────────────────────────────

    def _resolve_backend(self, choice: BackendChoice) -> str:
        if choice != "auto":
            return choice
        if _try_import("layoutparser"):
            logger.info("  Auto-selected backend: layoutparser")
            return "layoutparser"
        if _try_import("transformers") and _try_import("torch"):
            logger.info("  Auto-selected backend: dit")
            return "dit"
        logger.info("  Auto-selected backend: heuristic (no ML libs found)")
        return "heuristic"

    def _open_pdf(self, pdf_path: str):
        try:
            import fitz
            doc = fitz.open(pdf_path)
            logger.debug(f"Opened PDF for rendering: {pdf_path}")
            return doc
        except Exception as e:
            logger.warning(f"Could not open PDF for rendering: {e}")
            return None

    def _render_page_image(self, pdf_doc, page_index: int):
        """Render a PDF page to a PIL Image at self.render_dpi."""
        try:
            import fitz
            from PIL import Image
            page = pdf_doc[page_index]
            mat  = fitz.Matrix(self.render_dpi / 72, self.render_dpi / 72)
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception as e:
            logger.warning(f"Could not render page {page_index}: {e}")
            return None

    def _detect_page(
        self,
        page:            PageExtractionResult,
        doc_id:          str,
        page_idx_in_doc: int,
        pdf_doc,
    ) -> PageLayoutResult:
        """
        Run detection on one page and return PageLayoutResult.

        Steps
        -----
        1. Always run heuristic detector (cheap, produces text-enriched regions)
        2. If ML backend available: render page → run ML → fuse with heuristic
        3. Return fused result (or heuristic-only if no ML)
        """
        # ── Step 1: Heuristic (always runs) ──────────────────
        heur_result = self._heuristic.detect_page(
            page            = page,
            doc_id          = doc_id,
            page_idx_in_doc = page_idx_in_doc,
        )

        # ── Step 2: ML detection (if available) ──────────────
        if self._ml_detector and pdf_doc:
            page_img = self._render_page_image(pdf_doc, page.page_index)
            if page_img:
                try:
                    ml_regions = self._ml_detector.detect(
                        page_image = page_img,
                        page       = page,
                        doc_id     = doc_id,
                    )
                    # ── Step 3: Fuse ML + heuristic ──────────
                    fused = fuse_ml_and_heuristic(
                        ml_regions    = ml_regions,
                        heur_regions  = heur_result.regions,
                        iou_threshold = self.iou_threshold,
                    )
                    backend = (DetectionBackend.LAYOUTPARSER
                               if self._backend_name == "layoutparser"
                               else DetectionBackend.DIT)
                    return PageLayoutResult(
                        page_index = page.page_index,
                        width      = page.width,
                        height     = page.height,
                        regions    = fused,
                        backend    = DetectionBackend.HYBRID,
                    )
                except Exception as e:
                    logger.warning(
                        f"ML detection failed on page {page.page_index}: {e}. "
                        f"Falling back to heuristic."
                    )

        # ── Heuristic only ────────────────────────────────────
        return heur_result

    def _prepare_output_dirs(self, doc_id: str):
        (self.output_dir / doc_id / "json").mkdir(parents=True, exist_ok=True)

    def _save_json(self, result: DocumentLayoutResult, doc_id: str):
        path = self.output_dir / doc_id / "json" / "layout.json"
        result.save_json(str(path))
        logger.info(f"  Saved → {path}")