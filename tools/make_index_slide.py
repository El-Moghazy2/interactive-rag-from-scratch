"""Add the 'kinds of vector index' slide to Part 4 of both decks and bring the index code slide
in line with the notebook, which now searches through a FAISS index.

Usage: python tools/make_index_slide.py   (safe to run again)
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOTS = [(30, 25), (55, 40), (40, 72), (70, 84), (95, 30), (115, 55), (100, 88), (140, 25), (160, 50), (145, 82), (176, 76), (180, 30)]
Q = (122, 70)
STAR = "M122 59l3.2 7.2l7.8 .8l-5.8 5.3l1.6 7.7l-6.8 -3.9l-6.8 3.9l1.6 -7.7l-5.8 -5.3l7.8 -.8z"


def svg(body):
    return f'<svg viewBox="0 0 210 112" class="ixpic">{body}<path class="ix-q" d="{STAR}"/></svg>'


def dots(hot=(), dim=()):
    return "".join(f'<circle cx="{x}" cy="{y}" r="6" class="ix-d{" hot" if i in hot else ""}{" dim" if i in dim else ""}"/>' for i, (x, y) in enumerate(DOTS))


flat = svg("".join(f'<path class="ix-l" d="M{Q[0]} {Q[1]}L{x} {y}"/>' for x, y in DOTS) + dots(hot=range(12)))
B = (4, 5, 6)
ivf = svg('<circle class="ix-c" cx="48" cy="55" r="40"/><circle class="ix-c on" cx="106" cy="60" r="38"/><circle class="ix-c" cx="161" cy="52" r="40"/>'
          + "".join(f'<path class="ix-l" d="M{Q[0]} {Q[1]}L{DOTS[i][0]} {DOTS[i][1]}"/>' for i in B) + dots(hot=B, dim=[i for i in range(12) if i not in B]))
EDGES = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (3, 6), (4, 7), (7, 8), (8, 9), (9, 10), (8, 11), (5, 8), (6, 9)]
PATH = [(0, 1), (1, 4), (4, 5)]
hnsw = svg("".join(f'<path class="ix-e{" on" if e in PATH else ""}" d="M{DOTS[a][0]} {DOTS[a][1]}L{DOTS[b][0]} {DOTS[b][1]}"/>' for e in EDGES for a, b in [e])
           + dots(hot=(0, 1, 4, 5), dim=[i for i in range(12) if i not in (0, 1, 4, 5)]))

WHAT = f'''<section class="slide" id="idx-what" data-part="4" style="gap:14px">
  <div class="kicker"><b>4</b> · Vector index · the kinds</div>
  <h2>A vector index answers one question: which stored vectors are nearest?</h2>
  <div class="cols3">
    <div class="card" data-step="1">{flat}<h3>Flat</h3><p class="ixp">Compare with <b>every</b> vector. Always exact. Fine for thousands of chunks, and what your Colab uses.</p></div>
    <div class="card" data-step="2">{ivf}<h3>Clusters (IVF)</h3><p class="ixp">Search only inside the <b>nearest clusters</b>. Much faster, can miss. The demo you just played.</p></div>
    <div class="card" data-step="3">{hnsw}<h3>Graph (HNSW)</h3><p class="ixp">Hop along <b>links between neighbours</b> toward the question. Fast and accurate, uses more memory. The default in most vector databases.</p></div>
  </div>
  <div data-step="4" class="col" style="gap:10px">
<pre class="code" style="font-size:17.5px;padding:12px 0">
index = faiss.IndexFlatIP(384)           # an empty index for 384-number vectors
index.add(emb)                           # store every chunk's vector
scores, ids = index.search(q, 3)         # ask for the 3 chunks nearest to the question
</pre>
    <p class="small">A vector database (Chroma, Qdrant, pgvector) is an index like these, plus storage on disk, filters and updates.</p>
  </div>__BANNER__
</section>'''

CODE = '''<section class="slide" id="idx-code" data-part="4">
  <div class="kicker"><b>4</b> · Vector index · code</div>
  <h2>A real vector index in your pipeline</h2>
<pre class="code" data-hl="1:1-2;2:5;3:6-7;4:9-11" style="font-size:17px">
index = faiss.IndexFlatIP(emb.shape[1])        # an empty index for 384-number vectors
index.add(emb)                                 # store every chunk's vector

def semantic_search(question, k=3):
    q = embedder.encode([question], normalize_embeddings=True)
    scores, ids = index.search(q, k)           # ask the index
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(384), 384, 6, faiss.METRIC_INNER_PRODUCT)
ivf.train(emb); ivf.add(emb)                   # find 6 clusters, file every chunk
ivf.nprobe = 2                                 # clusters to open per question
</pre>
  <div class="caps card" style="min-height:104px">
    <p data-only="0" class="on">Press <kbd>→</kbd> to step through the code.</p>
    <p data-only="1"><b>Build the index once.</b> A flat index compares with everything, so it is exact. "IP" is inner product, which equals cosine similarity because our vectors have length 1.</p>
    <p data-only="2"><b>Embed the question</b> with the same model as the chunks. A different model would give numbers from a different space.</p>
    <p data-only="3"><b>Ask the index.</b> It returns two arrays: the scores, and the positions of the nearest chunks. Every later part of your RAG searches through this function.</p>
    <p data-only="4"><b>The shortcut at scale.</b> The same call on a clustered index; <span class="mono">nprobe</span> is the slider you just played with. Your pipeline keeps the exact index.</p>
  </div>
</section>'''

CSS = """
/* vector index kinds */
#idx-what .card { padding: 12px 16px; }
#idx-what .card h3 { font-size: 21px; margin: 4px 0 2px; }
.ixp { font-size: 17px; line-height: 1.35; }
.ixpic { width: 100%; height: 118px; display: block; }
.ixpic .ix-d { fill: var(--muted); opacity: .9; stroke: var(--soft); stroke-width: 1.5; }
.ixpic .ix-d.hot { fill: var(--hit); } .ixpic .ix-d.dim { opacity: .28; }
.ixpic .ix-l { stroke: var(--hit); stroke-width: 1.6; opacity: .55; }
.ixpic .ix-c { fill: none; stroke: var(--line); stroke-width: 2; stroke-dasharray: 4 4; }
.ixpic .ix-c.on { stroke: var(--hit); fill: var(--hit-bg); stroke-dasharray: none; }
.ixpic .ix-e { stroke: var(--line); stroke-width: 2; }
.ixpic .ix-e.on { stroke: var(--work); stroke-width: 4; }
.ixpic .ix-q { fill: var(--bad); stroke: var(--soft); stroke-width: 1.5; }
"""

BANNER = '\n  <div class="banner"><span>▶ <b>Colab · Part 4</b> run the cells: build the vector index (<code>INDEXING</code>), then search it (<code>RETRIEVAL</code>)</span><span class="time">2 min</span></div>'


def section(s, sid):
    a = s.index(f'<section class="slide" id="{sid}"')
    b = s.index("</section>", a) + len("</section>")
    return a, b


for deck in ["site", "site-short"]:
    p = ROOT / deck / "index.html"
    s = p.read_text(encoding="utf-8")
    short = deck == "site-short"
    if 'id="idx-what"' in s:
        a, b = section(s, "idx-what")
        while s[b] == "\n":
            b += 1
        s = s[:a] + s[b:]
    a, b = section(s, "idx-demo")
    if short:                       # the Colab banner moves to the last slide of the part
        blk = re.sub(r'\n  <div class="banner">.*?</div>', "", s[a:b], flags=re.S)
        s = s[:a] + blk + s[b:]
        a, b = section(s, "idx-demo")
    s = s[:b] + "\n\n" + WHAT.replace("__BANNER__", BANNER if short else "") + s[b:]
    if not short:
        a, b = section(s, "idx-code")
        s = s[:a] + CODE + s[b:]
        old = "fill in <code>top</code> in <code>semantic_search</code>, then run the FAISS cell"
        if old in s:
            s = s.replace(old, "run the cells: build the vector index (<code>INDEXING</code>), then search it (<code>RETRIEVAL</code>)")
    s = re.sub(r'\?v=(\d+)"', lambda m: f'?v={int(m.group(1)) + 1}"', s)
    p.write_text(s, encoding="utf-8")
    c = ROOT / deck / "style.css"
    css = c.read_text(encoding="utf-8")
    if "/* vector index kinds */" in css:
        x = css.index("\n/* vector index kinds */")
        y = css.find("\n/* ", x + 5)
        css = css[:x] + (css[y:] if y > 0 else "")
    c.write_text(css + CSS, encoding="utf-8")
    print(deck, "slides:", len(re.findall(r'<section class="slide"', s)))
