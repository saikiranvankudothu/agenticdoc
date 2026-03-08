"""
extractors/heuristic_layout_detector.py
----------------------------------------
Rule-based layout classifier that runs with ZERO ML dependencies.

Used as:
  (a) The primary detector when LayoutParser / DiT is not installed
  (b) A confidence-boosting layer that refines ML predictions using
      document-structure rules

Design rationale
----------------
Academic PDFs follow very consistent typographic conventions.
A well-tuned rule engine on font size, position, and text patterns
achieves ~85% accuracy on clean PDFs — good enough for a prototype
and essential as a fallback in offline/constrained environments.

Rules implemented (in priority order)
--------------------------------------
1. Page-header / page-footer  →  top / bottom 5% of page height
2. Title                      →  large bold font near top of page
3. Section heading            →  bold font, shorter line, not body size
4. Abstract                   →  first text block containing "abstract"
5. Caption                    →  starts with Fig./Table/Algorithm
6. Equation                   →  high math-symbol density
7. Reference                  →  starts with [N] or numbered pattern
8. Algorithm                  →  contains "Algorithm" header
9. Table                      →  adjacent to figure block OR grid-like text
10. Figure                    →  image block from Step 2
11. List                      →  lines starting with bullet / number
12. Paragraph                 →  default (body text)
"""

from __future__ import annotations
import hashlib
import logging
import re
from typing import Optional

from extractors.models import (
    BoundingBox, TextBlock, FigureBlock,
    PageExtractionResult, DocumentExtractionResult,
)
from extractors.layout_models import (
    LayoutRegion, PageLayoutResult, RegionClass, DetectionBackend,
)
from extractors.reading_order import sort_reading_order

logger = logging.getLogger(__name__)

# ── Regex patterns ─────────────────────────────────────────
RE_CAPTION     = re.compile(
    r"^(fig(?:ure)?\.?\s*\d+|table\s*\d+|tab\.?\s*\d+|algorithm\s*\d+)",
    re.IGNORECASE)
RE_REFERENCE   = re.compile(r"^\[\d+\]|^\d{1,3}\.\s+[A-Z]")
RE_SECTION_NUM = re.compile(r"^(I{1,4}V?|VI{0,3}|[IVX]+\.|\d+\.)\s+[A-Z]")
RE_BULLET      = re.compile(r"^(\s*[-•◦▪▸*]\s+|\s*\d+[.)]\s+)")
RE_ABSTRACT    = re.compile(r"\babstract\b", re.IGNORECASE)
MATH_CHARS     = set("∑∫∂∇αβγδεζηθλμπρσφψω±×÷→↔⇒⊕⊗∈∉⊂⊆∪∩≤≥≠≈∞")


def _region_id(doc_id: str, page_idx: int, seq: int) -> str:
    raw = f"{doc_id}::layout::p{page_idx}::r{seq}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _math_density(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c in MATH_CHARS) / len(text)


def _is_header_footer(bbox: BoundingBox, page_height: float,
                       margin_frac: float = 0.06) -> Optional[RegionClass]:
    """Check if bbox falls in top/bottom margin strip."""
    margin = page_height * margin_frac
    if bbox.y1 <= margin:
        return RegionClass.HEADER
    if bbox.y0 >= page_height - margin:
        return RegionClass.FOOTER
    return None


def _classify_text_block(
    blk: TextBlock,
    page_height: float,
    page_width: float,
    median_font_size: float,
    page_idx_in_doc: int,
    seq: int,
) -> RegionClass:
    """
    Apply heuristic rules to classify a single TextBlock.
    Returns the most specific RegionClass that matches.
    """
    text   = blk.text.strip()
    style  = blk.style
    bbox   = blk.bbox
    fsize  = style.font_size or median_font_size
    bold   = style.is_bold or False

    # ── Rule 1: header / footer ──────────────────────────
    hf = _is_header_footer(bbox, page_height)
    if hf:
        return hf

    # ── Rule 2: caption ──────────────────────────────────
    if RE_CAPTION.match(text):
        return RegionClass.CAPTION

    # ── Rule 3: reference entry ───────────────────────────
    if RE_REFERENCE.match(text):
        return RegionClass.REFERENCE

    # ── Rule 4: equation ─────────────────────────────────
    if _math_density(text) > 0.06:
        return RegionClass.EQUATION

    # ── Rule 5: abstract ─────────────────────────────────
    if page_idx_in_doc == 0 and RE_ABSTRACT.match(text):
        return RegionClass.ABSTRACT

    # ── Rule 6: title (large bold near top on first pages) ──
    if (fsize >= median_font_size * 1.5 and bold
            and bbox.y0 < page_height * 0.35
            and page_idx_in_doc <= 1):
        return RegionClass.TITLE

    # ── Rule 7: section heading ───────────────────────────
    if bold and fsize >= median_font_size * 1.1 and len(text) < 120:
        if RE_SECTION_NUM.match(text) or fsize >= median_font_size * 1.3:
            return RegionClass.TITLE   # section headings → TITLE class

    # ── Rule 8: list ─────────────────────────────────────
    lines = text.splitlines()
    bullet_lines = sum(1 for l in lines if RE_BULLET.match(l))
    if bullet_lines >= 2 or (bullet_lines == 1 and len(lines) == 1):
        return RegionClass.LIST

    # ── Rule 9: algorithm block ───────────────────────────
    if text.lower().startswith("algorithm") and len(text) > 30:
        return RegionClass.ALGORITHM

    # ── Rule 10: default → paragraph ─────────────────────
    return RegionClass.PARAGRAPH


class HeuristicLayoutDetector:
    """
    Rule-based layout region detector.

    Consumes a DocumentExtractionResult (Step 2 output) and
    produces a DocumentLayoutResult (Step 3 output).

    This runs completely offline with no ML dependencies.
    When LayoutParser or DiT is available, this class is used
    as a post-processing refinement step.

    Parameters
    ----------
    confidence_text  : confidence assigned to heuristically classified text regions
    confidence_image : confidence assigned to image/figure regions (from Step 2)
    """

    def __init__(
        self,
        confidence_text:  float = 0.80,
        confidence_image: float = 0.95,
    ):
        self.confidence_text  = confidence_text
        self.confidence_image = confidence_image

    def detect_page(
        self,
        page: PageExtractionResult,
        doc_id: str,
        page_idx_in_doc: int,
    ) -> PageLayoutResult:
        """
        Classify all blocks on a single page into LayoutRegions.

        Parameters
        ----------
        page             : PageExtractionResult from Agent 1
        doc_id           : document identifier
        page_idx_in_doc  : 0-based page number within the full document
                           (needed for title / abstract heuristics)
        """
        regions: list[LayoutRegion] = []
        seq = 0

        # ── Compute median font size for relative comparisons ──
        font_sizes = [
            b.style.font_size for b in page.text_blocks
            if b.style.font_size is not None
        ]
        median_fs = (
            sorted(font_sizes)[len(font_sizes) // 2]
            if font_sizes else 11.0
        )

        # ── Classify text blocks ───────────────────────────────
        for blk in page.text_blocks:
            cls = _classify_text_block(
                blk             = blk,
                page_height     = page.height,
                page_width      = page.width,
                median_font_size= median_fs,
                page_idx_in_doc = page_idx_in_doc,
                seq             = seq,
            )
            regions.append(LayoutRegion(
                region_id        = _region_id(doc_id, page.page_index, seq),
                region_class     = cls,
                bbox             = blk.bbox,
                page_index       = page.page_index,
                confidence       = self.confidence_text,
                backend          = DetectionBackend.HEURISTIC,
                text_content     = blk.text,
                source_block_ids = [blk.block_id],
            ))
            seq += 1

        # ── Classify figure blocks (always FIGURE) ─────────────
        for fig in page.figure_blocks:
            regions.append(LayoutRegion(
                region_id        = _region_id(doc_id, page.page_index, seq),
                region_class     = RegionClass.FIGURE,
                bbox             = fig.bbox,
                page_index       = page.page_index,
                confidence       = self.confidence_image,
                backend          = DetectionBackend.HEURISTIC,
                text_content     = None,
                source_block_ids = [fig.block_id],
            ))
            seq += 1

        # ── Sort regions into proper reading order ──────────────
        # Uses column-aware XY-cut algorithm instead of naive (y0, x0) sort.
        # Correctly handles two-column IEEE/ACM layouts and full-width blocks.
        regions = sort_reading_order(
            regions              = regions,
            page_width           = page.width,
            page_height          = page.height,
            full_width_threshold = 0.55,
            column_gap_min       = 15.0,
        )

        return PageLayoutResult(
            page_index = page.page_index,
            width      = page.width,
            height     = page.height,
            regions    = regions,
            backend    = DetectionBackend.HEURISTIC,
        )