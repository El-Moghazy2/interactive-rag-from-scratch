"""Rebuild the title slide of both decks: big title, one sentence, the speaker, one setup strip.

Usage: python tools/make_title.py   (safe to run again)
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "Abdelrhman ElMoghazy"

CSS = """
/* title slide */
.title-main { display: flex; flex-direction: column; justify-content: center; gap: 22px; flex: 1; min-height: 0; }
.title-main h1 { font-size: 104px; white-space: nowrap; }
.title-main .lead { font-size: 26px; max-width: none; white-space: nowrap; }
.title-main .rule { width: 72px; height: 5px; border-radius: 3px; background: var(--work); }
.speaker { font-size: 22px; color: var(--muted); }
.speaker b { color: var(--ink); font-size: 26px; }
.title-foot { display: flex; justify-content: space-between; gap: 20px; font-size: 14.5px; color: var(--muted); }
#title .banner { margin-top: 0; font-size: 17px; padding: 11px 18px; }
#title .banner .time { white-space: nowrap; }
"""

for deck, kicker, note in [
    ("site", "Workshop · bring a laptop", ""),
    ("site-short", "Workshop · 90 minutes · bring a laptop", "Slides use a 16-line toy example. Your Colab uses the full handbook."),
]:
    p = ROOT / deck / "index.html"
    s = p.read_text(encoding="utf-8")
    a = s.index('<section class="slide" id="title"')
    b = s.index("</section>", a) + len("</section>")
    s = s[:a] + f'''<section class="slide" id="title" data-part="0" style="gap:18px">
  <div class="title-main">
    <div class="kicker">{kicker}</div>
    <h1>Chat with your Docs</h1>
    <p class="lead">Build your first RAG system, one component at a time: watch it fail, play with it, run the code.</p>
    <div class="rule"></div>
    <p class="speaker">with <b>{NAME}</b></p>
  </div>
  <div class="banner"><span>▶ <b>Do this now:</b> <b>el-moghazy2.github.io/interactive-rag-from-scratch</b> → Open in Colab → <b>T4 GPU</b> → run <b>0 · Setup</b></span><span class="time">2 min</span></div>
  <div class="title-foot"><span><kbd>→</kbd> next · <kbd>←</kbd> back · <kbd>O</kbd> all slides · <kbd>F</kbd> fullscreen · <kbd>T</kbd> theme</span><span>{note}</span></div>
</section>''' + s[b:]
    s = re.sub(r'\?v=(\d+)"', lambda m: f'?v={int(m.group(1)) + 1}"', s)
    p.write_text(s, encoding="utf-8")
    c = ROOT / deck / "style.css"
    css = c.read_text(encoding="utf-8")
    if "/* title slide */" in css:              # replace only this block and keep whatever comes after it
        x = css.index("\n/* title slide */")
        y = css.find("\n/* ", x + 5)
        css = css[:x] + (css[y:] if y > 0 else "")
    c.write_text(css + CSS, encoding="utf-8")
print("title slide rebuilt in both decks")
