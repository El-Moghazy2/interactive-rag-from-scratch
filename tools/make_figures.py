"""Draw the notebook's figures as SVG files in notebooks/figures/.

Every number in a figure is computed here with the notebook's own code path
(same chunks, same embedding model, same BM25 tokenizer, same rank fusion).

Usage: python tools/make_figures.py
"""
import html
import json
import math
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "figures"

# ---------------------------------------------------------------- palette
PAPER, INK, MUTED, LINE, WHITE = "#FCFBF7", "#1F2933", "#667085", "#98A2B3", "#FFFFFF"
STAGE = {                      # stroke, fill
    "INDEXING":     ("#2B6CB0", "#E4EEF9"),
    "RETRIEVAL":    ("#1B7F64", "#DFF3EC"),
    "AUGMENTATION": ("#B26A00", "#FCEFD6"),
    "GENERATION":   ("#9B2C6F", "#F8E4EF"),
    "EVALUATION":   ("#5546B8", "#EAE7FA"),
}
SANS = "Helvetica, Arial, sans-serif"
MONO = "Consolas, Menlo, 'DejaVu Sans Mono', monospace"


# ---------------------------------------------------------------- drawing helpers
def esc(s):
    return html.escape(str(s), quote=False)


def T(x, y, s, size=13, anchor="start", weight="normal", fill=INK, italic=False, family=None, spacing=None):
    extra = ' font-style="italic"' if italic else ""
    if family:
        extra += f' font-family="{family}"'
    if spacing:
        extra += f' letter-spacing="{spacing}"'
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}"{extra}>{esc(s)}</text>')


def TL(x, y, rows, size=13, lh=None, **kw):
    lh = lh or size * 1.35
    return "".join(T(x, y + n * lh, r, size, **kw) for n, r in enumerate(rows))


def R(x, y, w, h, fill=WHITE, stroke=LINE, r=6, sw=1.2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def L(pts, color=INK, sw=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"{d}/>'


def A(pts, color=INK, sw=1.5, dash=None, head=8):
    """A polyline that ends in an arrowhead."""
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    a = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(a), y2 - head * math.sin(a)
    left = (bx + head * 0.45 * math.sin(a), by - head * 0.45 * math.cos(a))
    right = (bx - head * 0.45 * math.sin(a), by + head * 0.45 * math.cos(a))
    tri = f'<polygon points="{x2:.1f},{y2:.1f} {left[0]:.1f},{left[1]:.1f} {right[0]:.1f},{right[1]:.1f}" fill="{color}"/>'
    return L(pts[:-1] + [(bx, by)], color, sw, dash) + tri


def chip(x, y, stage, size=10.5):
    stroke, _ = STAGE[stage]
    w = len(stage) * size * 0.68 + 16
    return (R(x, y, w, size + 9, fill=stroke, stroke=stroke, r=4, sw=0)
            + T(x + w / 2, y + size + 2.5, stage, size, "middle", "bold", WHITE, spacing="0.6"))


def node(x, y, w, h, title, sub=(), stage=None, size=14):
    stroke, fill = STAGE[stage] if stage else (LINE, WHITE)
    out = R(x, y, w, h, fill=fill, stroke=stroke, sw=1.6 if stage else 1.2)
    rows = len(sub)
    top = y + h / 2 - (rows * 15) / 2 + 5
    out += T(x + w / 2, top, title, size, "middle", "bold", stroke if stage else INK)
    for n, s in enumerate(sub):
        out += T(x + w / 2, top + 17 + n * 15, s, 11.5, "middle", fill=MUTED)
    return out


def wrap(s, width):
    rows, cur = [], ""
    for word in s.split():
        if len(cur) + len(word) + 1 > width and cur:
            rows.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    return rows + [cur] if cur else rows


def cut(s, n):
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def save(name, w, h, body):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
           f'font-family="{SANS}">'
           f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" fill="{PAPER}" stroke="#E6E1D3"/>'
           f'{body}</svg>')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(svg, encoding="utf-8", newline="\n")
    print("wrote", name)


# ---------------------------------------------------------------- the notebook's pipeline, for real numbers
STOPWORDS = set("a an the i my me you your is are am do does did can could what when where which who how "
                "if in on at to for of and or it be with from that this until much many there have has".split())


def tokenize(text):
    words = re.findall(r"[a-z]+-\d+|[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def rrf(rankings, k=60):
    points = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            points[i] = points.get(i, 0) + 1 / (k + rank)
    return sorted(points.items(), key=lambda item: -item[1])


raw = (ROOT / "site" / "data.js").read_text(encoding="utf-8")
chunks = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["chunks"]
texts = [c["text"] for c in chunks]
ids = [c["id"] for c in chunks]
titles = [t.split(". ")[0] for t in texts]
embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(texts, normalize_embeddings=True)
bm25 = BM25Okapi([tokenize(t) for t in texts])

Q_RETAKE = "Can I retake an exam I failed?"
Q_CODE = "Which semester is CS-240 taught in?"


def semantic(question, k):
    q = embedder.encode([question], normalize_embeddings=True)[0]
    s = emb @ q
    return [(int(i), float(s[i])) for i in np.argsort(-s, kind="stable")[:k]], s, q


def keyword(question, k):
    s = bm25.get_scores(tokenize(question))
    return [(int(i), float(s[i])) for i in np.argsort(-s, kind="stable")[:k] if s[i] > 0], s


def hybrid(question, k=3, pool=5):
    return rrf([[i for i, _ in semantic(question, pool)[0]], [i for i, _ in keyword(question, pool)[0]]])[:k]


# ================================================================ Figure 1: three ways to answer
def fig_three_ways():
    b = ""
    panels = [
        ("(a) The model alone", "a closed-book exam", None,
         ["+  nothing to build", "–  has never seen your documents", "–  may invent a confident answer"]),
        ("(b) Fine-tuning", "study the book before the exam", ("Labelled data", "Training"),
         ["+  the knowledge is inside the model", "–  needs labelled data and GPU time", "–  train again when a rule changes"]),
        ("(c) RAG", "an open-book exam", ("Documents", "Search"),
         ["+  the model stays as it is", "+  new rule? edit the document", "+  the answer can cite its source"]),
    ]
    for n, (title, sub, top, bullets) in enumerate(panels):
        x, y = 20 + n * 313, 20
        accent = [MUTED, STAGE["GENERATION"][0], STAGE["RETRIEVAL"][0]][n]
        b += R(x, y, 294, 300, fill=WHITE, stroke="#D9D4C5", r=8)
        b += T(x + 16, y + 30, title, 15.5, weight="bold")
        b += T(x + 16, y + 49, sub, 12.5, fill=MUTED, italic=True)
        ry = y + 132
        b += node(x + 16, ry, 74, 40, "Question", size=12.5)
        b += A([(x + 90, ry + 20), (x + 122, ry + 20)])
        b += R(x + 122, ry, 64, 40, fill="#F1F0EC", stroke=INK, sw=1.6) + T(x + 154, ry + 25, "LLM", 13.5, "middle", "bold")
        b += A([(x + 186, ry + 20), (x + 204, ry + 20)])
        b += node(x + 204, ry, 74, 40, "Answer", size=12.5)
        if top:
            b += R(x + 14, y + 68, 88, 32, fill=WHITE, stroke=accent) + T(x + 58, y + 88.5, top[0], 11.5, "middle", fill=accent, weight="bold")
            b += A([(x + 102, y + 84), (x + 114, y + 84)], accent, head=6)
            b += R(x + 114, y + 68, 80, 32, fill=WHITE, stroke=accent) + T(x + 154, y + 88.5, top[1], 11.5, "middle", fill=accent, weight="bold")
            b += A([(x + 154, y + 100), (x + 154, ry)], accent)
            note = "new weights" if n == 1 else "pages into the prompt"
            b += T(x + 162, y + 120, note, 10.5, fill=accent, italic=True)
        else:
            b += T(x + 154, y + 92, "only what it saw in training", 11, "middle", fill=MUTED, italic=True)
            b += L([(x + 154, y + 100), (x + 154, ry)], MUTED, 1.2, "3 3")
        b += L([(x + 16, y + 196), (x + 278, y + 196)], "#E6E1D3", 1)
        for m, row in enumerate(bullets):
            good = row.startswith("+")
            b += T(x + 18, y + 224 + m * 24, row, 12.5, fill="#1B7F64" if good else "#B42318" if n < 2 or not good else INK)
    save("fig1_three_ways.svg", 960, 340, b)


# ================================================================ Figure 2: the whole system
def fig_pipeline():
    ix, ixf = STAGE["INDEXING"]
    b = R(20, 20, 920, 172, fill="#F3F7FC", stroke="#C9DBF0", r=8)
    b += chip(36, 34, "INDEXING") + T(124, 47.5, "once, before any question is asked", 12.5, fill=MUTED, italic=True)
    y = 84
    b += node(40, y, 140, 64, "Document", ["handbook.pdf"], "INDEXING")
    b += node(232, y, 140, 64, "Chunks", [f"{len(chunks)} short passages"], "INDEXING")
    b += node(424, y, 140, 64, "Vectors", ["384 numbers per chunk"], "INDEXING")
    b += node(630, 58, 200, 52, "Vector index", ["FAISS: the chunks' vectors"], "INDEXING", size=13)
    b += node(630, 118, 200, 52, "Keyword index", ["BM25: the chunks' word counts"], "INDEXING", size=13)
    for x1, x2, lab in [(180, 232, "split"), (372, 424, "embed")]:
        b += A([(x1, y + 32), (x2, y + 32)], ix) + T((x1 + x2) / 2, y + 24, lab, 10.5, "middle", fill=ix)
    b += A([(564, y + 22), (596, y + 22), (596, 84), (630, 84)], ix) + T(597, 74, "store", 10.5, "middle", fill=ix)
    b += A([(302, y + 64), (302, 178), (606, 178), (606, 144), (630, 144)], ix)
    b += T(454, 172, "count the words", 10.5, "middle", fill=ix)
    for x, lab in [(110, "Part 2"), (302, "Part 2"), (494, "Part 3"), (730, "Parts 4 and 5")]:
        b += T(x, 62 if x != 730 else 48, lab, 10.5, "middle", fill=MUTED)

    b += R(20, 236, 920, 164, fill="#FAF9F5", stroke="#E3DECF", r=8)
    b += T(36, 262, "ANSWERING", 11, weight="bold", fill=INK, spacing="0.6")
    b += T(124, 262, "repeated for every question", 12.5, fill=MUTED, italic=True)
    y = 292
    b += node(40, y, 116, 76, "Question", ["from the student"])
    b += node(204, y, 176, 76, "Retrieval", ["search both indexes,", "merge the two rankings"], "RETRIEVAL")
    b += node(428, y, 176, 76, "Augmentation", ["prompt = rules +", "chunks + question"], "AUGMENTATION")
    b += node(652, y, 150, 76, "Generation", ["the LLM writes", "the answer"], "GENERATION")
    b += node(850, y, 80, 76, "Answer", ["with its", "sources"])
    for x1, x2 in [(156, 204), (380, 428), (604, 652), (802, 850)]:
        b += A([(x1, y + 38), (x2, y + 38)])
    for x, lab in [(292, "Parts 4 and 5"), (516, "Part 6"), (727, "Parts 1 and 6")]:
        b += T(x, y + 92, lab, 10.5, "middle", fill=MUTED)
    rt = STAGE["RETRIEVAL"][0]
    b += L([(830, 84), (858, 84), (858, 144), (830, 144)], rt, 1.8)
    b += A([(858, 114), (896, 114), (896, 214), (292, 214), (292, y)], rt, 1.8)
    b += T(594, 208, "both indexes are searched; the best-matching chunks come back", 11.5, "middle", fill=rt, weight="bold")
    save("fig2_pipeline.svg", 960, 420, b)


# ================================================================ Figure 3: chunking
def fig_chunking():
    ix, ixf = STAGE["INDEXING"]
    b = T(30, 34, "The document", 13, weight="bold") + T(300, 34, "Where may the splitter cut?", 13, weight="bold")
    b += T(640, 34, "The chunks", 13, weight="bold")
    b += R(30, 48, 210, 318, fill=WHITE, stroke=LINE, r=4)
    tints = ["#E4EEF9", "#DFF3EC", "#FCEFD6"]
    edge = ["#2B6CB0", "#1B7F64", "#B26A00"]
    yy = 64
    for p in range(3):
        b += R(42, yy - 6, 186, 92, fill=tints[p], stroke="none", r=4, sw=0)
        b += R(50, yy + 2, 70, 7, fill=edge[p], stroke="none", r=2, sw=0)
        for r_, wd in enumerate([170, 164, 170, 150, 96]):
            b += R(50, yy + 18 + r_ * 13, wd, 6, fill="#B9C0CC", stroke="none", r=2, sw=0)
        yy += 102
    b += T(135, 384, "one rule per paragraph", 11.5, "middle", fill=MUTED, italic=True)

    rungs = [("1", "blank line", "between paragraphs"), ("2", "line break", "inside a paragraph"),
             ("3", "full stop", "between sentences"), ("4", "space", "between words")]
    shades = ["#2B6CB0", "#5A8FC7", "#8FB3DA", "#BCD2EA"]
    for n, (num, name, where) in enumerate(rungs):
        y = 56 + n * 62
        b += R(300, y, 260, 44, fill=WHITE, stroke=shades[n], sw=1.6)
        b += R(300, y, 40, 44, fill=shades[n], stroke=shades[n], sw=1.6) + T(320, y + 28, num, 15, "middle", "bold", WHITE)
        b += T(354, y + 20, name, 13.5, weight="bold") + T(354, y + 36, where, 11.5, fill=MUTED)
        if n < 3:
            b += A([(430, y + 44), (430, y + 62)], MUTED, 1.3, head=6)
            b += T(440, y + 57, "piece still longer than chunk_size?", 10.5, fill=MUTED, italic=True)
    b += TL(300, 316, ["Start with the biggest natural break.", "Go one step down only for a piece",
                       "that is still too long."], 12, fill=INK)
    b += A([(240, 200), (292, 200)], ix) + A([(568, 200), (632, 200)], ix)

    for n, i in enumerate((18, 19, 20)):
        y = 48 + n * 84
        b += R(640, y, 296, 72, fill=tints[n], stroke=edge[n], sw=1.4)
        b += T(652, y + 20, ids[i], 12.5, weight="bold", fill=edge[n], family=MONO)
        b += T(924, y + 20, f"{len(texts[i])} characters", 11, "end", fill=MUTED)
        b += TL(652, y + 40, wrap(cut(texts[i], 84), 48)[:2], 11.5, 16)
    y = 314
    b += L([(640, y - 8), (936, y - 8)], "#E6E1D3", 1)
    b += R(640, y + 18, 176, 14, fill=ixf, stroke=ix, r=3) + R(776, y + 40, 160, 14, fill=ixf, stroke=ix, r=3)
    b += R(776, y + 18, 40, 14, fill=ix, stroke=ix, r=0) + R(776, y + 40, 40, 14, fill=ix, stroke=ix, r=0)
    b += T(728, y + 10, "chunk_size", 11, "middle", fill=ix, weight="bold", family=MONO)
    b += T(796, y + 70, "overlap", 11, "middle", fill=ix, weight="bold", family=MONO)
    b += T(640, y + 70, "two neighbours share", 10.5, fill=MUTED, italic=True)
    save("fig3_chunking.svg", 960, 400, b)


# ================================================================ Figure 4: embeddings
def fig_embeddings():
    ix, ixf = STAGE["INDEXING"]
    top, s, q = semantic(Q_RETAKE, 3)
    best = top[0][0]
    far = int(np.argmin(s))
    b = T(30, 34, "Text in, numbers out", 13, weight="bold") + T(430, 34, "Similar meaning, similar direction", 13, weight="bold")
    b += R(30, 52, 300, 46, fill=WHITE) + T(180, 80, "“" + Q_RETAKE + "”", 13, "middle", italic=True)
    b += A([(180, 98), (180, 132)], ix)
    b += R(80, 132, 200, 62, fill=ixf, stroke=ix, sw=1.6) + T(180, 158, "Embedding model", 14, "middle", "bold", ix)
    b += T(180, 177, "all-MiniLM-L6-v2", 11.5, "middle", fill=MUTED, family=MONO)
    b += A([(180, 194), (180, 228)], ix)
    lim = float(np.abs(q).max())
    for n in range(20):
        v = float(q[n]) / lim
        col = "#2B6CB0" if v >= 0 else "#C2410C"
        b += (f'<rect x="{40 + n * 14}" y="232" width="12" height="22" rx="2" fill="{col}" '
              f'fill-opacity="{0.15 + 0.85 * abs(v):.2f}"/>')
    b += T(326, 248, "…", 14, fill=MUTED)
    nums = ", ".join(f"{float(v):.2f}".replace("-", "–") for v in q[:4])
    b += T(180, 280, f"[{nums}, …]", 12.5, "middle", family=MONO)
    b += T(180, 300, "384 numbers: the question's meaning", 11.5, "middle", fill=MUTED, italic=True)
    b += TL(30, 338, ["Every chunk goes through the same model,", "so chunks and questions live in one space."], 12)

    ox, oy, rad = 560, 356, 250
    b += L([(ox, oy), (ox + 330, oy)], "#D0D5DD", 1) + L([(ox, oy), (ox, oy - 290)], "#D0D5DD", 1)
    base = 8
    items = [("question", 0.0, INK, None), (ids[best], math.degrees(math.acos(top[0][1])), STAGE["RETRIEVAL"][0], top[0][1]),
             (ids[far], math.degrees(math.acos(float(s[far]))), "#B42318", float(s[far]))]
    for name, ang, col, score in items:
        a = math.radians(base + ang)
        tx, ty = ox + rad * math.cos(a), oy - rad * math.sin(a)
        b += A([(ox, oy), (tx, ty)], col, 2.2, head=10)
        if score is None:
            b += T(tx + 8, ty + 4, "the question", 12.5, fill=col, weight="bold")
        else:
            r2 = 70 if name == ids[best] else 110
            n_pts = 24
            arc = [(ox + r2 * math.cos(math.radians(base + ang * k / n_pts)), oy - r2 * math.sin(math.radians(base + ang * k / n_pts)))
                   for k in range(n_pts + 1)]
            b += L(arc, col, 1.2, "3 3")
            lab_a = math.radians(base + ang)
            anchor = "start" if math.cos(lab_a) > 0.15 else "end"
            dx = 10 if anchor == "start" else -10
            b += T(tx + dx, ty - 14, f"{name}  {titles[ids.index(name)]}", 12.5, anchor, "bold", col)
            b += T(tx + dx, ty + 2, f"cosine similarity {score:.2f}".replace("-", "–"), 11.5, anchor, fill=col)
    b += T(ox + 334, oy + 18, "small angle = similar meaning", 11.5, "end", fill=MUTED, italic=True)
    save("fig4_embeddings.svg", 960, 400, b)


# ================================================================ Figure 5: the vector index
def fig_vector_index():
    rt, rtf = STAGE["RETRIEVAL"]
    ix, ixf = STAGE["INDEXING"]
    top, s, q = semantic(Q_RETAKE, 3)
    win = [i for i, _ in top]
    show_rows = sorted(set([0, 1, len(chunks) - 1] + win))
    lim = float(np.abs(emb).max())

    def strip(x, y, vec, n=14):
        out = ""
        for k in range(n):
            v = float(vec[k]) / lim
            col = "#2B6CB0" if v >= 0 else "#C2410C"
            out += (f'<rect x="{x + k * 13}" y="{y}" width="11" height="16" rx="2" fill="{col}" '
                    f'fill-opacity="{min(1, 0.15 + 0.85 * abs(v)):.2f}"/>')
        return out + T(x + n * 13 + 4, y + 12, "…", 12, fill=MUTED)

    b = T(30, 34, "Stored in the index: one vector per chunk", 13, weight="bold")
    b += T(350, 34, "Score = question · chunk", 13, weight="bold") + T(690, 34, "Keep the k = 3 largest", 13, weight="bold")
    b += R(30, 50, 280, 40, fill=WHITE, stroke=INK) + T(42, 75, "question", 12.5, weight="bold") + strip(108, 62, q)
    y = 112
    prev = None
    rank_y = {}
    for i in show_rows:
        if prev is not None and i != prev + 1:
            b += T(60, y + 12, "⋮", 16, "middle", fill=MUTED) + T(372, y + 12, "⋮", 16, "middle", fill=MUTED)
            y += 26
        hit = i in win
        b += R(30, y, 280, 32, fill=rtf if hit else WHITE, stroke=rt if hit else LINE, sw=1.5 if hit else 1.1)
        b += T(42, y + 21, ids[i], 12.5, weight="bold", family=MONO, fill=rt if hit else INK) + strip(108, y + 8, emb[i])
        sc = float(s[i])
        b += T(350, y + 21, f"{sc:.3f}".replace("-", "–"), 12.5, family=MONO, weight="bold" if hit else "normal", fill=rt if hit else INK)
        b += R(408, y + 9, max(2, sc / 0.5 * 190), 14, fill=rt if hit else "#C9CED6", stroke="none", r=3, sw=0)
        if hit:
            rank_y[i] = y + 16
        prev = i
        y += 40
    b += T(30, y + 14, f"IndexFlatIP compares the question with all {len(chunks)} vectors, so the result is exact.", 12,
           fill=MUTED, italic=True)
    b += A([(612, 130), (680, 130)], rt, 1.8)
    for n, (i, sc) in enumerate(top):
        yy = 60 + n * 70
        b += R(690, yy, 246, 56, fill=rtf, stroke=rt, sw=1.5)
        b += T(704, yy + 24, f"{n + 1}.  {ids[i]}", 13.5, weight="bold", fill=rt, family=MONO)
        b += T(924, yy + 24, f"{sc:.3f}", 12.5, "end", family=MONO, fill=rt)
        b += T(704, yy + 43, cut(titles[i], 34), 12, fill=INK)
    save("fig5_vector_index.svg", 960, y + 34, b)


# ================================================================ Figure 6: the keyword index
def fig_keyword():
    rt, rtf = STAGE["RETRIEVAL"]
    ix, ixf = STAGE["INDEXING"]
    toks = tokenize(Q_CODE)
    top, s = keyword(Q_CODE, 3)
    docs = [set(tokenize(t)) for t in texts]
    b = T(30, 34, "1  The question becomes search words", 13, weight="bold")
    b += R(30, 48, 330, 40, fill=WHITE) + T(195, 73, "“" + Q_CODE + "”", 13, "middle", italic=True)
    b += A([(360, 68), (410, 68)], rt) + T(385, 60, "tokenize", 10.5, "middle", fill=rt)
    x = 418
    for w in re.findall(r"[a-z]+-\d+|[a-z0-9]+", Q_CODE.lower()):
        keep = w in toks
        wd = len(w) * 8.2 + 20
        b += R(x, 52, wd, 32, fill=rtf if keep else "#F2F1EC", stroke=rt if keep else "#D5D2C8", r=16)
        b += T(x + wd / 2, 73, w, 12.5, "middle", "bold" if keep else "normal", rt if keep else "#A4A29A", family=MONO)
        if not keep:
            b += L([(x + 9, 68.5), (x + wd - 9, 68.5)], "#A4A29A", 1.1)
        x += wd + 8
    b += T(418, 104, "filler words are dropped; a code like cs-240 stays whole", 11.5, fill=MUTED, italic=True)

    b += T(30, 146, "2  The keyword index: which chunks contain each word?", 13, weight="bold")
    b += T(42, 174, "word", 11, fill=MUTED) + T(170, 174, "found in", 11, fill=MUTED) + T(470, 174, "weight (rarer = higher)", 11, fill=MUTED)
    y = 184
    wmax = max(bm25.idf.get(t, 0) for t in toks) or 1
    for t in toks:
        found = [ids[i] for i, d in enumerate(docs) if t in d]
        b += R(30, y, 610, 40, fill=WHITE, stroke=ix if found else LINE)
        b += T(42, y + 25, t, 13, weight="bold", family=MONO, fill=ix)
        lab = ", ".join(found[:5]) + (f"  … {len(found)} chunks" if len(found) > 5 else "") if found else "no chunk"
        b += T(170, y + 25, lab, 12.5, family=MONO)
        idf = float(bm25.idf.get(t, 0))
        b += R(470, y + 13, max(2, idf / wmax * 110), 14, fill=ix, stroke="none", r=3, sw=0)
        b += T(470 + max(2, idf / wmax * 110) + 8, y + 25, f"{idf:.2f}", 11.5, family=MONO, fill=MUTED)
        y += 48
    b += T(30, y + 12, "BM25 adds up the weights of the question's words that a chunk contains.", 12, fill=MUTED, italic=True)

    b += T(690, 146, "3  Best keyword matches", 13, weight="bold")
    for n, (i, sc) in enumerate(top):
        yy = 162 + n * 62
        b += R(690, yy, 246, 52, fill=rtf, stroke=rt, sw=1.5)
        b += T(704, yy + 22, f"{n + 1}.  {ids[i]}", 13.5, weight="bold", fill=rt, family=MONO)
        b += T(924, yy + 22, f"{sc:.2f}", 12.5, "end", family=MONO, fill=rt)
        b += T(704, yy + 40, cut(titles[i], 34), 12)
    b += A([(640, 250), (682, 250)], rt)
    save("fig6_keyword.svg", 960, max(y + 34, 162 + len(top) * 62 + 16), b)


# ================================================================ Figure 7: rank fusion
def fig_fusion():
    rt, rtf = STAGE["RETRIEVAL"]
    sem = [i for i, _ in semantic(Q_CODE, 5)[0]]
    key = [i for i, _ in keyword(Q_CODE, 5)[0]]
    fused = rrf([sem, key])[:3]
    winners = [i for i, _ in fused]
    tone = {winners[0]: ("#1B7F64", "#DFF3EC"), winners[1]: ("#2B6CB0", "#E4EEF9"), winners[2]: ("#B26A00", "#FCEFD6")}
    b = T(480, 32, "“" + Q_CODE + "”", 13.5, "middle", italic=True)

    def column(x, head, sub, lst):
        out = T(x, 66, head, 13, weight="bold") + T(x, 83, sub, 11.5, fill=MUTED)
        for n, i in enumerate(lst):
            y = 96 + n * 44
            st, fl = tone.get(i, (LINE, WHITE))
            out += R(x, y, 250, 36, fill=fl, stroke=st, sw=1.5 if i in tone else 1.1)
            out += f'<circle cx="{x + 20}" cy="{y + 18}" r="11" fill="{WHITE}" stroke="{st}"/>'
            out += T(x + 20, y + 22.5, n + 1, 12, "middle", "bold", st if i in tone else MUTED)
            out += T(x + 40, y + 23, ids[i], 12.5, weight="bold", family=MONO, fill=st if i in tone else INK)
            out += T(x + 76, y + 23, cut(titles[i], 23), 12, fill=INK if i in tone else MUTED)
        return out

    b += column(30, "Ranked by meaning", "the vector index, top 5", sem)
    b += column(310, "Ranked by keyword", f"BM25, top {len(key)} that matched", key)
    b += T(630, 66, "Fused ranking", 13, weight="bold") + T(630, 83, "points = sum of 1 / (60 + rank)", 11.5, fill=MUTED)
    for n, (i, pts) in enumerate(fused):
        y = 96 + n * 72
        st, fl = tone[i]
        parts = [f"1/(60+{lst.index(i) + 1})" for lst in (sem, key) if i in lst]
        b += R(630, y, 306, 60, fill=fl, stroke=st, sw=1.6)
        b += T(644, y + 25, f"{n + 1}.  {ids[i]}", 13.5, weight="bold", fill=st, family=MONO)
        b += T(712, y + 25, cut(titles[i], 26), 12.5)
        b += T(644, y + 46, " + ".join(parts) + f" = {pts:.3f}", 12, family=MONO, fill=st)
    b += A([(568, 190), (622, 190)], rt, 1.8)
    rows = max(len(sem), len(key))
    b += T(30, 96 + rows * 44 + 22, "A chunk that both searches like collects points twice, so it rises to the top. Raw scores are never compared.",
           12, fill=MUTED, italic=True)
    save("fig7_fusion.svg", 960, 96 + rows * 44 + 40, b)


# ================================================================ Figure 8: the prompt
def fig_prompt():
    res = hybrid(Q_RETAKE)
    au, auf = STAGE["AUGMENTATION"]
    blocks = [
        ("Role", "who the model is", MUTED, "#F1F0EC",
         ["You are a helpful assistant for Moghazy University students."]),
        ("Rules", "stay inside the context, admit a gap, show sources", "#B42318", "#FDECEA",
         ["Answer the question using ONLY the context below.",
          "If the answer is not in the context, say",
          "\"I don't know based on the handbook.\"",
          "Cite the chunk ids you used in square brackets, like [c12]."]),
        ("Context", "the chunks RETRIEVAL found, each with its id", STAGE["RETRIEVAL"][0], STAGE["RETRIEVAL"][1],
         ["Context:"] + [cut(f"[{ids[i]}] {texts[i]}", 68) for i, _ in res]),
        ("Question", "the student's own words, unchanged", STAGE["INDEXING"][0], STAGE["INDEXING"][1],
         [f"Question: {Q_RETAKE}"]),
        ("Cue", "GENERATION continues the text from here", STAGE["GENERATION"][0], STAGE["GENERATION"][1],
         ["Answer:"]),
    ]
    b = T(30, 34, "The prompt that build_prompt returns", 13, weight="bold") + T(640, 34, "What each part is for", 13, weight="bold")
    total = sum(len(rows) * 19 + 14 for *_, rows in blocks) + 4 * 6
    b += R(30, 48, 580, total + 24, fill=WHITE, stroke=LINE, r=4)
    y = 60
    for name, why, st, fl, rows in blocks:
        h = len(rows) * 19 + 14
        b += R(40, y, 560, h, fill=fl, stroke="none", r=4, sw=0)
        b += R(40, y, 4, h, fill=st, stroke="none", r=0, sw=0)
        for n, row in enumerate(rows):
            b += T(54, y + 21 + n * 19, row, 12, family=MONO)
        cy = y + h / 2
        b += L([(600, cy), (640, cy)], st, 1.3)
        b += f'<circle cx="640" cy="{cy:.1f}" r="3.5" fill="{st}"/>'
        b += T(652, cy - 2, name, 13.5, weight="bold", fill=st) + T(652, cy + 15, why, 11.5, fill=MUTED)
        y += h + 6
    save("fig8_prompt.svg", 960, y + 28, b)


# ================================================================ Figure 9: evaluation
def fig_ragas():
    rt, rtf = STAGE["RETRIEVAL"]
    ge, gef = STAGE["GENERATION"]

    def pill(x, y, w, name, ask, st, fl):
        return (R(x, y, w, 46, fill=fl, stroke=st, r=23, sw=1.5) + T(x + w / 2, y + 20, name, 13, "middle", "bold", st)
                + T(x + w / 2, y + 36, ask, 11.5, "middle", fill=INK))

    b = node(40, 104, 180, 60, "Question")
    b += node(390, 104, 180, 60, "Retrieved chunks", ["what the model was given"])
    b += node(740, 104, 180, 60, "Answer", ["what the model wrote"])
    b += A([(220, 134), (390, 134)]) + A([(570, 134), (740, 134)])
    b += chip(264, 108, "RETRIEVAL") + chip(608, 108, "GENERATION")
    b += R(390, 352, 180, 60, fill="#F1F0EC", stroke=INK, sw=1.6)
    b += T(480, 378, "Reference answer", 14, "middle", "bold") + T(480, 396, "written by a person", 11.5, "middle", fill=MUTED)

    b += L([(560, 68), (500, 104)], ge, 1.3, "4 3") + L([(760, 68), (820, 104)], ge, 1.3, "4 3")
    b += pill(510, 24, 300, "Faithfulness", "is every claim backed by the chunks?", ge, gef)
    b += L([(480, 164), (480, 352)], rt, 1.3, "4 3")
    b += pill(320, 190, 320, "Context recall", "was everything the reference needs retrieved?", rt, rtf)
    b += pill(320, 262, 320, "Context precision", "are the useful chunks ranked above the rest?", rt, rtf)
    b += L([(830, 164), (570, 372)], ge, 1.3, "4 3")
    b += pill(668, 236, 268, "Answer accuracy", "does it agree with the reference?", ge, gef)

    b += R(40, 330, 250, 82, fill=WHITE, stroke="#E6E1D3")
    b += R(54, 346, 16, 16, fill=rtf, stroke=rt, r=8) + T(80, 359, "low score: fix retrieval", 12.5)
    b += R(54, 378, 16, 16, fill=gef, stroke=ge, r=8) + T(80, 391, "low score: fix the prompt or model", 12.5)
    b += T(40, 318, "Each score points at one part of the system", 12, fill=MUTED, italic=True)
    save("fig9_ragas.svg", 960, 436, b)


if __name__ == "__main__":
    fig_three_ways()
    fig_pipeline()
    fig_chunking()
    fig_embeddings()
    fig_vector_index()
    fig_keyword()
    fig_fusion()
    fig_prompt()
    fig_ragas()
