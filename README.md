# Agentic Document Understanding – Step 1 Implementation

## Project Overview

This project implements the **first stage of a modular document understanding pipeline**.
The goal of this stage is to extract structured information from PDF documents and store it in a **JSON format with layout metadata** such as bounding boxes, block types, and styles.

The pipeline currently focuses on:

- Extracting **text blocks**
- Detecting **equations**
- Recording **layout coordinates (bounding boxes)**
- Storing results in a **structured JSON file**

This structured output enables future stages such as **layout detection, semantic chunking, and LLM-based reasoning**.

---

# Current Pipeline (Step 1)

```
PDF Document
     ↓
Document Parsing
     ↓
Text Block Extraction
     ↓
Bounding Box Detection
     ↓
Structured JSON Output
     ↓
Visualization of Detected Text
```

This step ensures that the document is converted from a raw PDF into a **machine-readable layout-aware format**.

---

# Repository Structure

```
project/
│
├── pdf/
│   └── paper.pdf
│
├── outputs/
│   └── extraction.json
│
├── scripts/
│   ├── parse_document.py
│   └── visualize_blocks.py
│
└── README.md
```

---

# Input Document

The input is a **PDF document** containing:

- Text
- Figures
- Mathematical equations
- Multi-column layouts

Example:

```
paper.pdf
```

---

# Output JSON Structure

The extracted data is stored in a structured JSON format.

Example:

```json
{
  "doc_id": "paper",
  "total_pages": 3,
  "pages": [
    {
      "page_index": 0,
      "width": 612,
      "height": 792,
      "text_blocks": [
        {
          "block_id": "08824b22e9a4",
          "text": "Information-Flow-Aware KV Recomputation for Long Context",
          "bbox": {
            "x0": 178.04,
            "y0": 47.22,
            "x1": 418.84,
            "y1": 56.19
          },
          "block_type": "text",
          "confidence": 1.0
        }
      ],
      "figure_blocks": []
    }
  ]
}
```

---

# Extracted Elements

The system currently detects:

| Element             | Status |
| ------------------- | ------ |
| Text Blocks         | ✅     |
| Equations           | ✅     |
| Bounding Boxes      | ✅     |
| Font Style Metadata | ✅     |
| Figures             | ❌     |
| Tables              | ❌     |

---

# Key Observations

During testing with the sample document:

- **Text extraction works correctly**
- **Bounding boxes are accurate**
- **Equations are detected**
- **Figure images are not detected**

Example:

The PDF contains **Figure 1**, but the output JSON contains:

```
"figure_blocks": []
```

This means the parser extracted the **figure caption text**, but not the **figure image region**.

---

# Visualizing Detected Text Blocks

To verify the extracted layout, we can visualize the bounding boxes on top of the original PDF.

This helps confirm that the system correctly detected text regions.

## Install Dependencies

```
pip install pymupdf
```

---

## Visualization Script

```python
import fitz
import json

pdf = fitz.open("paper.pdf")

with open("extraction.json") as f:
    data = json.load(f)

for page_data in data["pages"]:
    page = pdf[page_data["page_index"]]

    for block in page_data["text_blocks"]:
        bbox = block["bbox"]

        rect = fitz.Rect(
            bbox["x0"],
            bbox["y0"],
            bbox["x1"],
            bbox["y1"]
        )

        page.draw_rect(rect, color=(1,0,0), width=1)

pdf.save("visualized_output.pdf")
```

---

## Visualization Output

The generated PDF will contain **bounding boxes around detected text blocks**.

Example visualization:

```
┌──────────────────────────────┐
│ Title Text                   │
└──────────────────────────────┘

┌──────────────────────────────┐
│ Paragraph text               │
│ Paragraph text               │
└──────────────────────────────┘
```

This allows quick verification of the parsing results.

---

# Limitations of Current Implementation

The current system does not yet detect:

- Figures
- Tables
- Images
- Reading order

These components require **layout detection models**.

---

# Next Steps (Step 2)

The next stage of the project will introduce **layout detection**.

Future pipeline:

```
PDF
 ↓
Layout Detection
 ↓
OCR + Text Extraction
 ↓
Semantic Chunking
 ↓
LLM Reasoning
 ↓
Structured Knowledge Output
```

Upcoming improvements:

- Figure detection
- Table detection
- Caption linking
- Layout classification
- Reading order reconstruction

---

# Long-Term Goal

The final system will function as a **Document Intelligence Pipeline** capable of understanding complex documents such as:

- Research papers
- Financial reports
- Technical documentation
- Scientific articles

The pipeline will transform documents into **structured knowledge usable by AI systems**.

---

# Summary

Step 1 successfully implements:

- PDF parsing
- Text extraction
- Layout-aware bounding boxes
- JSON structured output
- Visualization for debugging

This provides a strong foundation for building a **complete multimodal document understanding system**.
