"""Rebuild the prompt-caching slide in both decks: three requests, two colours, one sentence each.

Usage: python tools/make_cache_slide.py   (safe to run again)
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SLIDE = '''<section class="slide" id="lc-cache" data-part="8">
  <div class="kicker"><b>8</b> · RAG vs long context · caching</div>
  <h2>Prompt caching: the model skips what it has just read</h2>
  <div class="cols2 grow" style="grid-template-columns:1.5fr 1fr;align-items:stretch">
    <div class="panel col" style="gap:14px;justify-content:center">
      <div class="pc-legend"><span><i class="pc-new"></i>read now · full price</span><span><i class="pc-hit"></i>reused from the cache · a small fraction of the price</span></div>
      <div class="pc-axis"><span>start of the prompt</span><span>end</span></div>
      <div class="pc-row">
        <b>Question 1</b>
        <div class="pc-bar"><span class="pc-new" style="flex:8">instructions + documents</span><span class="pc-new" style="flex:1.3">Q1</span></div>
        <em class="pc-full">full price</em>
        <p>The first time, the model reads everything. The provider keeps that work for a while.</p>
      </div>
      <div class="pc-row" data-step="1">
        <b>Question 2</b>
        <div class="pc-bar"><span class="pc-hit" style="flex:8">instructions + documents · not read again</span><span class="pc-new" style="flex:1.3">Q2</span></div>
        <em class="pc-cheap">a fraction</em>
        <p>The prompt starts exactly like the last one, so only the new question is read.</p>
      </div>
      <div class="pc-row" data-step="2">
        <b>Question 3</b>
        <div class="pc-bar"><span class="pc-new pc-chg" style="flex:1.2">date</span><span class="pc-new" style="flex:6.8">instructions + documents · read again</span><span class="pc-new" style="flex:1.3">Q3</span></div>
        <em class="pc-full">full price again</em>
        <p>Someone added today's date at the top. The prompt no longer starts the same, so nothing can be reused.</p>
      </div>
    </div>
    <div class="col" style="justify-content:center">
      <div class="card" data-step="3"><b>Keep the start identical.</b> <span class="small">Instructions and documents first, the question last. Anything that changes goes at the end.</span></div>
      <div class="card" data-step="3"><b>It does not last.</b> <span class="small">The saved work is kept for minutes to hours, so it pays off when questions come often.</span></div>
      <div class="card bad" data-step="4"><b>It saves money and time, not attention.</b> <span class="small">The model still looks across every document, so a detail buried in a huge prompt is as easy to miss as before. And the documents still have to fit in the window.</span></div>
    </div>
  </div>
</section>'''

CSS = """
/* prompt caching slide */
.pc-legend { display: flex; gap: 26px; font-size: 16px; color: var(--ink); }
.pc-legend i { display: inline-block; width: 22px; height: 14px; border-radius: 4px; margin-right: 8px; vertical-align: -1px; }
.pc-new { background: var(--work); color: #1d1c1a; } .pc-hit { background: var(--hit); color: #fff; }
.pc-axis { display: flex; justify-content: space-between; font-size: 12.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 2px 142px -8px 118px; }
.pc-row { display: grid; grid-template-columns: 104px 1fr 128px; gap: 4px 14px; align-items: center; }
.pc-row > b { font-size: 18px; }
.pc-row > em { font-style: normal; font-weight: 800; font-size: 16px; text-align: right; }
.pc-row > em.pc-full { color: var(--work); } .pc-row > em.pc-cheap { color: var(--hit); }
.pc-row > p { grid-column: 2 / 4; font-size: 16.5px; color: var(--muted); }
.pc-bar { display: flex; gap: 3px; height: 46px; border-radius: 9px; overflow: hidden; font-size: 15.5px; font-weight: 600; }
.pc-bar span { display: flex; align-items: center; justify-content: center; white-space: nowrap; overflow: hidden; }
.pc-bar .pc-chg { box-shadow: inset 0 0 0 3px var(--bad); color: var(--bad); background: var(--bad-bg); font-weight: 800; }
"""

for deck in ["site", "site-short"]:
    p = ROOT / deck / "index.html"
    s = p.read_text(encoding="utf-8")
    a = s.index('<section class="slide" id="lc-cache"')
    b = s.index("</section>", a) + len("</section>")
    s = s[:a] + SLIDE + s[b:]
    s = re.sub(r'\?v=(\d+)"', lambda m: f'?v={int(m.group(1)) + 1}"', s)
    p.write_text(s, encoding="utf-8")
    c = ROOT / deck / "style.css"
    css = c.read_text(encoding="utf-8")
    if "/* prompt caching slide */" in css:
        x = css.index("\n/* prompt caching slide */")
        y = css.find("\n/* ", x + 5)
        css = css[:x] + (css[y:] if y > 0 else "")
    c.write_text(css + CSS, encoding="utf-8")
print("caching slide rebuilt in both decks")
