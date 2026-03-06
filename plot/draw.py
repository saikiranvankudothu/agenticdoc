import fitz
import json

# load PDF
pdf = fitz.open("pdf/paper.pdf")

# load json output
with open("output\paper\json\extraction.json") as f:
    data = json.load(f)

for page_data in data["pages"]:
    page_index = page_data["page_index"]
    page = pdf[page_index]

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