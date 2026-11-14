"""Chunking, part 2 of the story, in both decks:
  - 'chunk-rec'  : a play-button demo that animates recursive splitting on a long passage
  - 'chunk-code' : the code slide, now LangChain's RecursiveCharacterTextSplitter (the Colab blank)

Usage: python tools/make_rec_demo.py   (safe to run again)
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEMO_SLIDE = '''<section class="slide" id="chunk-rec" data-part="2" style="gap:11px">
  <div class="kicker"><b>2</b> · Chunking · a smarter cut</div>
  <h2>Recursive splitting: cut at the biggest natural break that fits</h2>
  <div class="controls">
    <label>maximum chunk size <input type="range" id="rc-size" min="80" max="400" step="20" value="200"><output id="rc-size-o"></output></label>
    <button class="btn go" id="rc-play">▶ Play</button>
    <button class="btn" id="rc-step">Step</button>
    <button class="btn" id="rc-reset">Reset</button>
    <span class="recrow" id="rc-ladder" style="margin-left:auto"></span>
  </div>
  <div class="recdemo grow">
    <div class="panel"><div class="rctext" id="rc-text"></div></div>
    <div class="panel col" style="gap:6px"><h3 id="rc-title"></h3><div class="rclist" id="rc-list"></div></div>
  </div>
  <div class="card" id="rc-cap" style="font-size:18.5px;padding:10px 16px;min-height:66px;transition:all .3s"></div>
</section>'''

CODE_SLIDE = '''<section class="slide" id="chunk-code" data-part="2">
  <div class="kicker"><b>2</b> · Chunking · code</div>
  <h2>Recursive splitting is one library call</h2>
<pre class="code" data-hl="1:1;2:5-6;3:7;4:8;5:10" style="font-size:17px">
from langchain_text_splitters import RecursiveCharacterTextSplitter

def make_splitter(size=300, overlap=50):
    return RecursiveCharacterTextSplitter(
        chunk_size=size,                    # the longest a chunk may be, in characters
        chunk_overlap=overlap,              # how much two neighbours share
        separators=["\\n\\n", "\\n", ". ", " ", ""],   # biggest break first
        keep_separator="end")               # a full stop stays with its sentence

pieces = make_splitter(200, 0).split_text(text)
</pre>
  <div class="caps card" style="min-height:104px">
    <p data-only="0" class="on">Press <kbd>→</kbd> to step through the code.</p>
    <p data-only="1"><b>The library.</b> LangChain ships the algorithm you just watched. You do not write the splitting yourself.</p>
    <p data-only="2"><b>Size and overlap.</b> The same two dials as in the first demo: the most characters a chunk may hold, and how much two neighbours repeat.</p>
    <p data-only="3"><b>The separators.</b> The list from the demo, from the biggest break to the smallest: blank line, line break, full stop, space, and finally "anywhere".</p>
    <p data-only="4"><b>Where the separator goes.</b> <span class="mono">"end"</span> keeps each full stop attached to the sentence before it, so chunks do not start with a stray dot.</p>
    <p data-only="5"><b>Use it.</b> A list of strings comes back. In Part 7 your own PDF is cut with this function.</p>
  </div>__BANNER__
</section>'''

BANNER = '\n  <div class="banner"><span>▶ <b>Colab · Part 2</b> <code>INDEXING</code> run the cells: load the PDF, then cut it into chunks</span><span class="time">4 min</span></div>'

CSS = """
/* recursive splitting demo */
.recrow { display: flex; align-items: center; gap: 7px; }
.recrow em { font-style: normal; color: var(--muted); font-size: 16px; }
.recstep { display: inline-flex; align-items: center; gap: 7px; font-size: 15px; background: var(--soft); border: 1px solid var(--line); border-radius: 99px; padding: 3px 11px 3px 4px; opacity: .55; transition: all .3s; }
.recstep b { width: 22px; height: 22px; border-radius: 50%; background: var(--ink); color: var(--stage); display: inline-flex; align-items: center; justify-content: center; font-size: 12.5px; flex: none; }
.recstep i { font-style: normal; font-weight: 700; }
.recstep.used { opacity: 1; }
.recstep.now { opacity: 1; background: var(--work); border-color: var(--work); color: #1d1c1a; }
.recdemo { display: grid; grid-template-columns: 1.6fr 1fr; gap: 16px; min-height: 0; }
.recdemo > * { min-width: 0; }
.rctext { font-size: 17.5px; line-height: 1.8; white-space: pre-wrap; }
.rctext .rp { padding: 3px 0; -webkit-box-decoration-break: clone; box-decoration-break: clone; transition: background .4s; }
.rctext .rp.ok { background: var(--hit-bg); } .rctext .rp.long { background: var(--bad-bg); }
.rctext .cut { display: inline-block; width: 3px; height: 1.15em; vertical-align: -3px; margin: 0 3px; border-radius: 2px; background: var(--muted); }
.rctext .cut.new { background: var(--work); width: 5px; box-shadow: 0 0 0 3px var(--work-bg); }
.rclist { display: flex; flex-direction: column; gap: 4px; font-size: 14.5px; overflow: hidden; }
.rclist div { display: grid; grid-template-columns: 24px 1fr auto; gap: 8px; align-items: center; background: var(--stage); border: 1px solid var(--line); border-radius: 7px; padding: 3px 9px; }
.rclist div.ok { box-shadow: inset 4px 0 0 var(--hit); } .rclist div.long { box-shadow: inset 4px 0 0 var(--bad); }
.rclist div span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rclist div b { font-family: var(--mono); font-weight: 600; } .rclist div.long b { color: var(--bad); } .rclist div.ok b { color: var(--hit); }
.rclist p { font-size: 14px; color: var(--muted); }
"""

DEMO_JS = r'''  /* ================= Part 2: recursive splitting, animated ================= */
  DEMOS['chunk-rec'] = {
    init(s) {
      /* The same algorithm as LangChain's RecursiveCharacterTextSplitter (no overlap), written out so we can
         show its rounds. window.RECDEMO.check() compares it with the library's real output. */
      const T = D.recDemo.text;
      const SEPS = ['\n\n', '\n', '. ', ' ', ''], NAMES = ['blank line', 'line break', 'full stop', 'space', 'anywhere'];
      const size = $('#rc-size', s);
      $('#rc-ladder', s).innerHTML = NAMES.map((n, i) => `${i ? '<em>→</em>' : ''}<span class="recstep" data-i="${i}"><b>${i + 1}</b><i>${n}</i></span>`).join('');

      const pick = (a, b, seps) => {                 // first separator in the list that occurs in this piece
        for (let i = 0; i < seps.length; i++) if (seps[i] === '' || T.slice(a, b).includes(seps[i])) return [seps[i], seps.slice(i + 1)];
        return ['', []];
      };
      const cutAt = (a, b, sep) => {                 // the separator stays at the end of the piece before it
        const out = [];
        if (sep === '') { for (let i = a; i < b; i++) out.push([i, i + 1]); return out; }
        let from = a, at;
        while ((at = T.indexOf(sep, from)) !== -1 && at < b) { out.push([from, at + sep.length]); from = at + sep.length; }
        if (from < b) out.push([from, b]);
        return out.filter(x => x[1] > x[0]);
      };
      const trim = ([a, b]) => { while (a < b && /\s/.test(T[a])) a++; while (b > a && /\s/.test(T[b - 1])) b--; return [a, b]; };
      const merge = (good, max) => {                 // glue neighbours while they stay within the maximum
        const out = []; let cur = [], total = 0;
        const flush = () => { if (cur.length) { const r = trim([cur[0][0], cur[cur.length - 1][1]]); if (r[1] > r[0]) out.push(r); } cur = []; total = 0; };
        good.forEach(d => { const len = d[1] - d[0]; if (total + len > max && cur.length) flush(); cur.push(d); total += len; });
        flush();
        return out;
      };
      const split = (a, b, seps, max) => {
        const [sep, rest] = pick(a, b, seps); let good = [], out = [];
        cutAt(a, b, sep).forEach(p => {
          if (p[1] - p[0] < max) good.push(p);
          else { if (good.length) { out = out.concat(merge(good, max)); good = []; } out = out.concat(rest.length ? split(p[0], p[1], rest, max) : [p]); }
        });
        if (good.length) out = out.concat(merge(good, max));
        return out;
      };
      /* the rounds shown on screen: every piece that is still too long is cut with its next separator */
      const rounds = max => {
        let pieces = [{ r: [0, T.length], seps: SEPS, long: T.length >= max }];
        const out = [{ kind: 'start', pieces, used: [] }];
        while (pieces.some(p => p.long)) {
          const used = new Set(), next = [], fresh = new Set();
          pieces.forEach(p => {
            if (!p.long) { next.push(p); return; }
            const [sep, rest] = pick(p.r[0], p.r[1], p.seps);
            const parts = cutAt(p.r[0], p.r[1], sep).filter(r => T.slice(r[0], r[1]).trim());   // ignore pieces that are only white space
            if (parts.length > 1) used.add(SEPS.indexOf(sep));
            parts.forEach((r, k) => { if (k) fresh.add(r[0]); next.push({ r, seps: rest, long: r[1] - r[0] >= max && rest.length > 0 }); });
          });
          pieces = next;
          if (used.size) out.push({ kind: 'round', pieces, used: [...used].sort(), fresh });   // show a round only if it made a visible cut
        }
        out.push({ kind: 'final', chunks: split(0, T.length, SEPS, max), used: [] });
        return out;
      };
      window.RECDEMO = { check: () => Object.entries(D.recDemo.real).filter(([sz, real]) => JSON.stringify(split(0, T.length, SEPS, +sz)) !== JSON.stringify(real)).map(x => x[0]) };

      let R = [], at = 0, token = 0;
      const listRow = (r, cls, n) => `<div class="${cls}"><i style="font-style:normal;color:var(--muted)">${n}</i><span>${esc(T.slice(r[0], r[1]).trim())}</span><b>${r[1] - r[0]}</b></div>`;
      const draw = () => {
        const max = +size.value, st = R[at], seen = new Set(R.slice(0, at + 1).flatMap(x => x.used));
        $('#rc-size-o', s).textContent = max + ' characters';
        s.querySelectorAll('#rc-ladder .recstep').forEach(e => { const i = +e.dataset.i; e.classList.toggle('now', st.used.includes(i)); e.classList.toggle('used', seen.has(i)); });
        const cap = $('#rc-cap', s);
        if (st.kind === 'final') {
          $('#rc-text', s).innerHTML = st.chunks.map((r, n) => `<span class="rp ck${n % 4}">${esc(T.slice(r[0], r[1]))}</span>`).join('<i class="cut"></i>');
          $('#rc-title', s).textContent = `Result: ${st.chunks.length} chunks`;
          $('#rc-list', s).innerHTML = st.chunks.slice(0, 9).map((r, n) => listRow(r, 'ok', n + 1)).join('') + (st.chunks.length > 9 ? `<p>… and ${st.chunks.length - 9} more</p>` : '');
          cap.className = 'card hit';
          cap.innerHTML = `<b>Glue.</b> Nothing is too long any more. Small neighbours are joined back together as long as they stay within ${max} characters. Each colour is one finished chunk.`;
          return;
        }
        const P = st.pieces, bad = P.filter(p => p.long).length;
        $('#rc-text', s).innerHTML = P.map((p, n) => (n ? `<i class="cut${st.fresh && st.fresh.has(p.r[0]) ? ' new' : ''}"></i>` : '') + `<span class="rp ${p.long ? 'long' : 'ok'}">${esc(T.slice(p.r[0], p.r[1]))}</span>`).join('');
        $('#rc-title', s).textContent = `${P.length} piece${P.length > 1 ? 's' : ''} · ${bad} too long`;
        const show = P.length > 9 ? P.filter(p => p.long).slice(0, 4).concat(P.filter(p => !p.long).slice(0, 5)) : P;
        $('#rc-list', s).innerHTML = show.map(p => listRow(p.r, p.long ? 'long' : 'ok', P.indexOf(p) + 1)).join('') + (P.length > 9 ? `<p>… ${P.length} pieces in total</p>` : '');
        if (st.kind === 'start') {
          cap.className = 'card ' + (bad ? 'bad' : 'hit');
          cap.innerHTML = bad ? `<b>Start.</b> The whole text is one piece of ${T.length} characters. The maximum is ${max}, so it is too long and must be split.`
            : `<b>Start.</b> The whole text is ${T.length} characters and the maximum is ${max}, so it already fits.`;
        } else {
          cap.className = 'card ' + (bad ? 'work' : 'hit');
          const names = st.used.map(i => NAMES[i]).join(' or ');
          cap.innerHTML = `<b>Split at every ${names}</b>, but only inside the pieces that were too long (red). ` +
            (bad ? `${bad} piece${bad > 1 ? 's are' : ' is'} still longer than ${max}, so the splitter moves down its list.` : `Every piece now fits within ${max}.`);
        }
      };
      const reset = () => { token++; R = rounds(+size.value); at = 0; draw(); };
      const step = () => { if (at < R.length - 1) { at++; draw(); return true; } return false; };
      $('#rc-step', s).onclick = () => { token++; if (!step()) reset(); };
      $('#rc-reset', s).onclick = reset;
      $('#rc-play', s).onclick = async () => { reset(); const my = token; while (my === token && at < R.length - 1) { await sleep(1700); if (my !== token) return; step(); } };
      size.oninput = reset;
      reset();
    },
  };

'''


def section(s, sid):
    a = s.index(f'<section class="slide" id="{sid}"')
    b = s.index("</section>", a) + len("</section>")
    return a, b


for deck in ["site", "site-short"]:
    short = deck == "site-short"
    p = ROOT / deck / "index.html"
    s = p.read_text(encoding="utf-8")
    a, b = section(s, "chunk-rec")
    s = s[:a] + DEMO_SLIDE + s[b:]
    a, b = section(s, "chunk-code")
    s = s[:a] + CODE_SLIDE.replace("__BANNER__", BANNER if short else "") + s[b:]
    if not short:   # the full deck carries its Colab banner on the quiz slide
        s = re.sub(r'(<b>Colab · Part 2</b>) [^<]*<code>step</code>.*?(</span><span class="time">)',
                   r'\1 <code>INDEXING</code> run the cells: load the PDF, then cut it into chunks\2', s, flags=re.S)
    s = re.sub(r'\?v=(\d+)"', lambda m: f'?v={int(m.group(1)) + 1}"', s)
    p.write_text(s, encoding="utf-8")

    j = ROOT / deck / "demos.js"
    t = j.read_text(encoding="utf-8")
    if "DEMOS['chunk-rec']" in t:
        x = t.index("  /* ================= Part 2: recursive splitting, animated")
        y = t.index("  /* ================= Part 3: embeddings")
        t = t[:x] + t[y:]
    y = t.index("  /* ================= Part 3: embeddings")
    j.write_text(t[:y] + DEMO_JS + t[y:], encoding="utf-8")

    c = ROOT / deck / "style.css"
    css = c.read_text(encoding="utf-8")
    for marker in ["/* recursive splitting slide */", "/* recursive splitting demo */"]:
        if marker in css:
            x = css.index("\n" + marker); y = css.find("\n/* ", x + 5)
            css = css[:x] + (css[y:] if y > 0 else "")
    c.write_text(css + CSS, encoding="utf-8")
    print(deck, "updated")
