"""Add the speaker's name to the title slide and a LinkedIn QR code to the closing slide of both decks.

Usage: python tools/add_speaker.py   (safe to run again; it replaces what it added before)
"""
import re
from pathlib import Path

import qrcode

NAME = "Abdelrhman ElMoghazy"
LINK = "https://www.linkedin.com/in/abdelrhman-elmoghazy/"
SHOWN = "linkedin.com/in/abdelrhman-elmoghazy"
ROOT = Path(__file__).resolve().parent.parent

qr = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M)
qr.add_data(LINK)
qr.make(fit=True)
m = qr.get_matrix()
n = len(m)
d = "".join(f"M{x} {y}h1v1h-1z" for y, row in enumerate(m) for x, on in enumerate(row) if on)
QR = f'<svg viewBox="0 0 {n} {n}" width="210" height="210" role="img" aria-label="QR code for {SHOWN}"><rect width="{n}" height="{n}" fill="#fff"/><path d="{d}" fill="#111"/></svg>'

for deck in ["site", "site-short"]:
    p = ROOT / deck / "index.html"
    s = p.read_text(encoding="utf-8")
    # the title slide is laid out by make_title.py; here we only keep the name on it in step
    s = re.sub(r'<p class="speaker">.*?</p>', f'<p class="speaker">with <b>{NAME}</b></p>', s)
    a = s.index('<section class="slide" id="end"')
    b = s.index("</section>", a) + len("</section>")
    part = re.search(r'data-part="(\d+)"', s[a:b]).group(1)
    s = s[:a] + f'''<section class="slide" id="end" data-part="{part}" style="justify-content:center">
  <div class="row" style="align-items:center;gap:60px">
    <div class="col" style="flex:1;gap:22px">
      <div class="kicker">Thank you</div>
      <h1>Questions?</h1>
      <p class="lead">You now have a working RAG in your Colab. Break it on purpose, then fix it: that is how the ideas stick.</p>
      <p class="small">Press <kbd>O</kbd> to jump back to any slide or demo during questions.</p>
    </div>
    <div class="card col center" style="gap:10px;padding:22px 26px">
      {QR}
      <b style="font-size:22px">{NAME}</b>
      <span class="mono" style="font-size:15px">{SHOWN}</span>
    </div>
  </div>
</section>''' + s[b:]
    s = re.sub(r'\?v=(\d+)"', lambda mm: f'?v={int(mm.group(1)) + 1}"', s)
    p.write_text(s, encoding="utf-8")
    c = ROOT / deck / "style.css"
    css = c.read_text(encoding="utf-8")
    if ".speaker" not in css:
        c.write_text(css + "\n.speaker { font-size: 24px; color: var(--muted); }\n.speaker b { color: var(--ink); }\n", encoding="utf-8")
print("speaker added to both decks; QR modules:", n)
