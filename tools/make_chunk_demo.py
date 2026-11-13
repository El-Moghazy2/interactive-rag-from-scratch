"""Short deck: rebuild the chunking demo. Left: the document. Right: the chunks that get stored,
one card each, with overlapping text marked in both neighbours.

Usage: python tools/make_chunk_demo.py   (safe to run again; then run tools/build_short.py)
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "site-short"

TEXT = ("Deadlines are strict and the same rules apply to all courses. "
        "Coursework handed in late loses 10 percent of the marks for each day. After five days the mark is zero. "
        "Extensions must be requested 48 hours before the deadline. The longest extension is ten days. "
        "A medical note is needed for any extension longer than three days. "
        "Group projects follow the same rules, and every member receives the same penalty.")

# ---- the toy text lives in build_short.py so it ends up in data.js
b = ROOT / "tools" / "build_short.py"
t = b.read_text(encoding="utf-8")
a = t.index('CHUNK_DEMO = {')
e = t.index("}\n", a) + 2
t = t[:a] + 'CHUNK_DEMO = {\n    "text": ' + repr(TEXT) + ',\n    "keepFrom": "loses 10 percent", "keepTo": "the mark is zero",\n    "label": "The late rule (10% per day, zero after five days)",\n}\n' + t[e:]
b.write_text(t, encoding="utf-8")

# ---- slide
h = S / "index.html"
s = h.read_text(encoding="utf-8")
a = s.index('<section class="slide" id="chunk-demo"')
e = s.index("</section>", a) + len("</section>")
s = s[:a] + '''<section class="slide" id="chunk-demo" data-part="2" style="gap:12px">
  <div class="kicker"><b>2</b> · Chunking · play</div>
  <h2>Cut the text into chunks. Then let them overlap.</h2>
  <div class="grow" id="chunk-root"></div>
  <p class="small" style="font-size:14.5px">Try: size 120 with overlap 0 and find where the late rule is cut. Raise the overlap to 40: one chunk now holds the whole rule, at the cost of storing some text twice. Then switch to recursive splitting and try the same sizes.</p>
</section>''' + s[e:]
s = re.sub(r'\?v=(\d+)"', lambda m: f'?v={int(m.group(1)) + 1}"', s)
h.write_text(s, encoding="utf-8")

# ---- demo code
j = S / "demos.js"
t = j.read_text(encoding="utf-8")
a = t.index("  DEMOS['chunk-demo'] = {")
e = t.index("  /* ================= Part 3: embeddings")
t = t[:a] + r'''  DEMOS['chunk-demo'] = {
    init(s) {
      const CD = D.chunkDemo, text = CD.text;
      const A = text.indexOf(CD.keepFrom), B = text.indexOf(CD.keepTo) + CD.keepTo.length;
      const root = $('#chunk-root', s);
      root.style.cssText = 'display:flex;flex-direction:column;gap:10px;min-height:0';
      root.innerHTML = `
        <div class="controls">
          <label>split by <select id="ck-mode"><option value="fixed">fixed size (characters)</option><option value="rec">recursive (LangChain)</option></select></label>
          <label>chunk size <input type="range" id="ck-size" min="100" max="300" step="10" value="120"><output id="ck-size-o"></output></label>
          <label>overlap <input type="range" id="ck-ov" min="0" max="150" step="10" value="0"><output id="ck-ov-o"></output></label>
        </div>
        <div class="ckgrid grow">
          <div class="panel col" style="gap:6px"><h3>1 · The document</h3><div class="chunktext" id="ck-text"></div>
            <p class="small" style="margin-top:auto;font-size:14px">Each colour is one chunk. <span class="ckov" style="padding:1px 5px">Striped</span> text falls in two chunks.</p></div>
          <div class="panel col" style="gap:6px"><h3>2 · The chunks that get stored and searched</h3><div class="ckcards" id="ck-cards"></div></div>
        </div>
        <div class="stat"><span><b id="ck-n"></b> chunks</span><span><b id="ck-stored"></b> characters stored for a <b>${text.length}</b>-character document</span>
          <span style="margin-left:auto">${CD.label}:</span><span class="verdict" id="ck-verdict"></span></div>`;
      const mode = $('#ck-mode', s), size = $('#ck-size', s), ov = $('#ck-ov', s);
      const recLookup = (sz, o) => {            // real output of LangChain's splitter, computed ahead of time
        const R = D.chunkRec, snap = (v, [a, b, st]) => Math.max(a, Math.min(b, a + Math.round((v - a) / st) * st));
        const s2 = snap(sz, R.sizes), o2 = Math.floor(Math.min(snap(o, R.ovs), s2 / 2) / R.ovs[2]) * R.ovs[2];
        return R.grid[s2 + '|' + o2] || [];
      };
      const ranges = () => {
        const out = [], sz = +size.value;
        if (mode.value === 'fixed') {
          const st = sz - Math.min(+ov.value, Math.floor(sz / 2));     // overlap is capped at half a chunk
          for (let a = 0; a < text.length; a += st) { out.push([a, Math.min(text.length, a + sz)]); if (a + sz >= text.length) break; }
        } else return recLookup(sz, +ov.value);
        return out;
      };
      const draw = () => {
        const fixed = mode.value === 'fixed';
        $('#ck-size-o', s).textContent = size.value; $('#ck-ov-o', s).textContent = Math.floor(Math.min(+ov.value, +size.value / 2) / 10) * 10;
        const r = ranges();
        const cuts = [...new Set(r.flat())].sort((a, b) => a - b);
        let html = '';
        for (let k = 0; k < cuts.length - 1; k++) {
          const a = cuts[k], b = cuts[k + 1];
          const cover = r.map((x, n) => x[0] <= a && x[1] >= b ? n : -1).filter(n => n >= 0);
          html += `<span class="${cover.length > 1 ? 'ckov' : cover.length ? 'ck' + cover[0] % 4 : ''}">${esc(text.slice(a, b))}</span>`;
        }
        $('#ck-text', s).innerHTML = html;
        $('#ck-cards', s).innerHTML = r.map((x, n) => {
          const t = text.slice(x[0], x[1]), len = t.length;
          const head = n > 0 ? Math.min(len, Math.max(0, r[n - 1][1] - x[0])) : 0;          // also in the previous chunk
          const tail = n < r.length - 1 ? Math.min(len - head, Math.max(0, x[1] - r[n + 1][0])) : 0;   // also in the next chunk
          const whole = x[0] <= A && x[1] >= B, part = !whole && x[0] < B && x[1] > A;
          return `<div class="ckcard ck${n % 4}${whole ? ' whole' : ''}"><b>chunk ${n + 1}<i>${len} chars${whole ? ' · whole late rule ✓' : part ? ' · part of the late rule' : ''}</i></b>` +
            `<span class="rep">${esc(t.slice(0, head))}</span>${esc(t.slice(head, len - tail))}<span class="rep">${esc(t.slice(len - tail))}</span></div>`;
        }).join('');
        $('#ck-cards', s).classList.toggle('many', r.length > 4 || +size.value > 180);
        $('#ck-n', s).textContent = r.length;
        $('#ck-stored', s).textContent = r.reduce((sum, x) => sum + x[1] - x[0], 0);
        const ok = r.some(x => x[0] <= A && x[1] >= B);
        const v = $('#ck-verdict', s);
        v.className = 'verdict ' + (ok ? 'ok' : 'no');
        v.textContent = ok ? '✓ whole in one chunk' : '✗ cut across chunks';
      };
      [mode, size, ov].forEach(e => e.oninput = draw);
      draw();
    },
  };

''' + t[e:]
j.write_text(t, encoding="utf-8")

c = S / "style.css"
css = c.read_text(encoding="utf-8")
if "/* chunk demo v2 */" in css:
    x = css.index("\n/* chunk demo v2 */")
    y = css.find("\n/* ", x + 5)
    css = css[:x] + (css[y:] if y > 0 else "")
c.write_text(css + """
/* chunk demo v2 */
.ckgrid { display: grid; grid-template-columns: 1fr 1.25fr; gap: 16px; min-height: 0; }
.ckgrid .chunktext { font-size: 19px; line-height: 1.85; white-space: normal; }
.ckcards { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; align-content: start; min-height: 0; overflow: auto; }
.ckcard { border-radius: 9px; padding: 7px 11px; font-size: 15.5px; line-height: 1.4; border: 2px solid transparent; color: var(--ink); }
.ckcards.many { grid-template-columns: 1fr 1fr 1fr; }
.ckcards.many .ckcard { font-size: 12.5px; padding: 5px 8px; line-height: 1.32; }
.ckcards.many .ckcard > b { font-size: 11.5px; }
.ckcard > b { display: block; font-size: 13px; letter-spacing: .06em; text-transform: uppercase; margin-bottom: 2px; }
.ckcard > b i { font-style: normal; font-weight: 500; text-transform: none; letter-spacing: 0; margin-left: 8px; opacity: .8; }
.ckcard .rep { background: repeating-linear-gradient(45deg, rgba(0,0,0,.2) 0 4px, transparent 4px 8px); border-radius: 3px; font-weight: 600; }
:root[data-theme="dark"] .ckcard .rep { background: repeating-linear-gradient(45deg, rgba(255,255,255,.25) 0 4px, transparent 4px 8px); }
.ckcard.whole { border-color: var(--hit); box-shadow: 0 0 0 2px var(--hit); }
""", encoding="utf-8")
print("chunk demo rebuilt (short deck)")
