"""Write data/handbook.pdf from data/handbook.md. The notebook and build_data.py read this PDF.

Paragraphs are kept whole on a page and separated by a clear gap, so a PDF reader in layout mode
gives one block of text per paragraph.
"""
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
pdf = FPDF()
pdf.set_auto_page_break(True, margin=18)
pdf.add_page()
for block in (ROOT / "data" / "handbook.md").read_text(encoding="utf-8").split("\n\n"):
    block = block.strip()
    if not block:
        continue
    with pdf.unbreakable() as doc:
        if block.startswith("# "):
            doc.set_font("Helvetica", "B", 20); doc.multi_cell(0, 10, block[2:]); doc.ln(4)
        elif block.startswith("## "):
            doc.ln(4); doc.set_font("Helvetica", "B", 14); doc.multi_cell(0, 8, block[3:]); doc.ln(4)
        else:
            doc.set_font("Helvetica", "", 11); doc.multi_cell(0, 6, block.replace("**", "")); doc.ln(4)
pdf.output(str(ROOT / "data" / "handbook.pdf"))
print("wrote data/handbook.pdf,", pdf.page_no(), "pages")
