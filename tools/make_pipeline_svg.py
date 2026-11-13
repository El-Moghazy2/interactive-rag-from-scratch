"""Rebuild the pipeline SVG on the 'pipeline' slide as pictograms and patch site/index.html."""
import re
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"
CK = ["#f3c766", "#6fcabb", "#9d90ee", "#ee8fa0"]          # chunk colours
OP = [[.9, .3, .6, 1, .45, .75, .25], [.4, .95, .3, .6, .8, .2, .7],
      [.7, .5, 1, .25, .55, .9, .35], [.3, .65, .45, .85, 1, .5, .6]]
TOP, BOT = 108, 322                                          # lane centre lines
TX = [84, 229, 374, 519]                                    # top lane centres
BX = [92, 262, 519, 708, 904, 1064]                         # bottom lane centres


def label(cx, y, title, sub):
    return (f'<text class="t" x="{cx}" y="{y}">{title}</text>'
            f'<text class="sub" x="{cx}" y="{y + 19}">{sub}</text>')


def arrow(x1, x2, y, at):
    return f'<path class="edge" data-at="{at}" d="M{x1} {y}H{x2}m-8 -6l8 6l-8 6"/>'


def page(cx, cy, w=58, h=76):
    x, y = cx - w / 2, cy - h / 2
    lines = "".join(f'<path class="tl" d="M{x + 9} {y + 14 + i * 10}h{w - 18 - (14 if i % 3 == 2 else 0)}"/>' for i in range(6))
    return f'<rect class="pg" x="{x}" y="{y}" width="{w}" height="{h}" rx="5"/>{lines}'


def cells(cx, cy, row, colour, n=7, s=9, gap=2):
    x0 = cx - (n * (s + gap) - gap) / 2
    return "".join(f'<rect x="{x0 + i * (s + gap):.1f}" y="{cy - s / 2}" width="{s}" height="{s}" rx="2" '
                   f'fill="{colour}" opacity="{OP[row][i]}"/>' for i in range(n))


DOTS = [(-30, -20, 0), (-22, -8, 0), (-34, -4, 0), (-18, -24, 0),
        (14, -22, 1), (26, -14, 1), (20, -28, 1), (32, -26, 1),
        (-8, 16, 2), (6, 22, 2), (-2, 28, 2), (14, 12, 2), (26, 24, 3), (34, 14, 3)]


def index_box(cx, cy, search=False):
    out = f'<rect class="box" x="{cx - 50}" y="{cy - 42}" width="100" height="84" rx="10"/>'
    if search:
        qx, qy = cx + 2, cy + 6
        for dx, dy, _ in (DOTS[8], DOTS[9], DOTS[11]):
            out += f'<path class="hitline" d="M{qx} {qy}L{cx + dx} {cy + dy}"/>'
    for n, (dx, dy, c) in enumerate(DOTS):
        hit = search and n in (8, 9, 11)
        out += f'<circle cx="{cx + dx}" cy="{cy + dy}" r="{6 if hit else 4.5}" fill="{CK[c]}"' + (' class="hitdot"' if hit else '') + '/>'
    if search:
        pts = []
        import math
        for i in range(10):
            a = math.pi / 5 * i - math.pi / 2
            r = 9 if i % 2 == 0 else 4
            pts.append(f"{qx + r * math.cos(a):.1f} {qy + r * math.sin(a):.1f}")
        out += f'<path class="qmark" d="M{"L".join(pts)}Z"/>'
    return out


def bubble(cx, cy, cls, text=None, w=112, h=54):
    x, y = cx - w / 2, cy - h / 2
    d = f"M{x + 10} {y}h{w - 20}a10 10 0 0 1 10 10v{h - 20}a10 10 0 0 1 -10 10h-{w - 44}l-12 12v-12h-12a10 10 0 0 1 -10 -10v-{h - 20}a10 10 0 0 1 10 -10z"
    inner = f'<text class="bt" x="{cx}" y="{cy + 5}">{text}</text>' if text else ""
    return f'<path class="{cls}" d="{d}"/>{inner}'


g = []


def band(x1, x2, y1, y2, cls, label):
    """A tinted box behind a group of steps, naming the RAG component they belong to."""
    return (f'<g class="band {cls}"><rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" rx="16"/>'
            f'<text x="{x1 + 16}" y="{y1 + 24}">{label}</text></g>')


Y1, Y2 = BOT - 84, BOT + 100                                   # the bottom lane's box
g.append(band(8, 596, TOP - 98, Y2, "ret", "RETRIEVAL"))      # retrieval holds both lanes
g.append(band(616, 800, Y1, Y2, "aug", "AUGMENTATION"))
g.append(band(820, 1152, Y1, Y2, "gen", "GENERATION"))
g.append(f'<text class="lane" x="130" y="{TOP - 74}">INDEXING · once, ahead of time</text>')
g.append(f'<path class="lanesplit" d="M24 {Y1 - 4}H580"/>')
g.append(f'<text class="lane" x="24" y="{Y1 + 20}">SEARCH · for every question</text>')

# ---- edges
g.append(arrow(TX[0] + 42, TX[1] - 42, TOP, 1))
g.append(arrow(TX[1] + 42, TX[2] - 52, TOP, 2))
g.append(arrow(TX[2] + 52, TX[3] - 58, TOP, 2))
g.append(f'<path class="edge" data-at="4" d="M{TX[3]} {TOP + 88}V{BOT - 50}m-6 -8l6 8l6 -8"/>')
g.append(arrow(BX[0] + 64, BX[1] - 52, BOT, 3))
g.append(arrow(BX[1] + 52, BX[2] - 58, BOT, 4))
g.append(arrow(BX[2] + 58, BX[3] - 42, BOT, 5))
g.append(arrow(BX[3] + 42, BX[4] - 48, BOT, 6))
g.append(arrow(BX[4] + 48, BX[5] - 66, BOT, 6))

# ---- top lane
cx = TX[0]
g.append(f'<g class="stage" data-at="1"><g transform="translate(9 -8)">{page(cx, TOP)}</g>{page(cx, TOP)}{label(cx, TOP + 62, "Documents", "the handbook")}</g>')
cx = TX[1]
blocks = "".join(f'<rect x="{cx - 29}" y="{TOP - 38 + i * 20}" width="58" height="16" rx="4" fill="{CK[i]}"/>'
                 f'<path class="tl dark" d="M{cx - 22} {TOP - 30 + i * 20}h{44 - 8 * (i % 2)}"/>' for i in range(4))
g.append(f'<g class="stage" data-at="1">{blocks}{label(cx, TOP + 62, "Chunks", "cut into pieces")}</g>')
cx = TX[2]
rows = "".join(cells(cx, TOP - 30 + i * 20, i, CK[i]) for i in range(4))
g.append(f'<g class="stage" data-at="2">{rows}{label(cx, TOP + 62, "Embeddings", "chunk → numbers")}</g>')
cx = TX[3]
g.append(f'<g class="stage" data-at="2">{index_box(cx, TOP)}{label(cx, TOP + 62, "Index", "grouped by meaning")}</g>')

# ---- bottom lane
cx = BX[0]
g.append(f'<g class="stage" data-at="3">{bubble(cx, BOT - 4, "qb", "Can I retake…?", w=124)}{label(cx, BOT + 62, "Question", "from a student")}</g>')
cx = BX[1]
g.append(f'<g class="stage" data-at="3">{cells(cx, BOT, 2, "var(--bad)")}{label(cx, BOT + 62, "Embed it", "same model")}</g>')
cx = BX[2]
g.append(f'<g class="stage" data-at="4">{index_box(cx, BOT, search=True)}{label(cx, BOT + 62, "Search", "nearest chunks win")}</g>')
cx = BX[3]
pr = (f'<rect class="pg" x="{cx - 33}" y="{BOT - 44}" width="66" height="88" rx="6"/>'
      f'<rect x="{cx - 26}" y="{BOT - 37}" width="52" height="12" rx="3" fill="var(--muted)" opacity=".55"/>'
      + "".join(f'<rect x="{cx - 26}" y="{BOT - 20 + i * 15}" width="52" height="12" rx="3" fill="{CK[2]}"/>' for i in range(3))
      + f'<rect x="{cx - 26}" y="{BOT + 25}" width="52" height="12" rx="3" fill="var(--bad)"/>')
g.append(f'<g class="stage" data-at="5">{pr}{label(cx, BOT + 62, "Prompt", "rules + chunks + question")}</g>')
cx = BX[4]
pins = "".join(f'<path class="pin" d="M{cx - 42} {BOT - 20 + i * 20}h8M{cx + 34} {BOT - 20 + i * 20}h8M{cx - 20 + i * 20} {BOT - 42}v8M{cx - 20 + i * 20} {BOT + 34}v8"/>' for i in range(3))
g.append(f'<g class="stage" data-at="6">{pins}<rect class="chip" x="{cx - 34}" y="{BOT - 34}" width="68" height="68" rx="12"/>'
         f'<text class="ct" x="{cx}" y="{BOT + 7}">LLM</text>{label(cx, BOT + 62, "Model", "reads, then writes")}</g>')
cx = BX[5]
ans = (bubble(cx, BOT - 4, "ab", w=124, h=62)
       + f'<path class="tl dark" d="M{cx - 48} {BOT - 20}h96M{cx - 48} {BOT - 8}h80M{cx - 48} {BOT + 4}h52"/>'
       + f'<rect x="{cx + 10}" y="{BOT - 4}" width="38" height="15" rx="4" fill="{CK[2]}"/><text class="cite" x="{cx + 29}" y="{BOT + 7}">[c20]</text>')
g.append(f'<g class="stage final" data-at="6">{ans}{label(cx, BOT + 62, "Answer", "with its source")}</g>')

svg = '<svg class="pipe" id="pipe-svg" viewBox="0 0 1160 432">\n      ' + "\n      ".join(g) + "\n    </svg>"

for deck, docs_label in [("site", "the handbook"), ("site-short", "our FAQ")]:   # both decks share this picture
    f = SITE.parent / deck / "index.html"
    html = f.read_text(encoding="utf-8")
    html, n = re.subn(r'<svg class="pipe" id="pipe-svg".*?</svg>', lambda m: svg.replace(">the handbook<", f">{docs_label}<"), html, flags=re.S)
    assert n == 1
    html = re.sub(r'\?v=(\d+)"', lambda m: f'?v={int(m.group(1)) + 1}"', html)
    f.write_text(html, encoding="utf-8")

css = (SITE / "style.css").read_text(encoding="utf-8")
if "/* pipeline pictograms */" not in css:
    css += """
/* pipeline pictograms */
.pipe .stage { opacity: .22; transition: opacity .4s; }
.pipe .stage.active, .pipe .stage.done, .pipe.lit .stage { opacity: 1; }
.pipe .stage text { text-anchor: middle; }
.pipe .stage .t { font-size: 19px; font-weight: 700; fill: var(--ink); transition: fill .3s; }
.pipe .stage.active .t { fill: var(--work); }
.pipe .stage .sub { font-size: 13.5px; fill: var(--muted); }
.pipe .pg { fill: var(--stage); stroke: var(--muted); stroke-width: 2; }
.pipe .tl { stroke: var(--line); stroke-width: 3; stroke-linecap: round; fill: none; }
.pipe .tl.dark { stroke: rgba(0,0,0,.28); }
.pipe .box { fill: var(--soft); stroke: var(--muted); stroke-width: 2; }
.pipe .hitline { stroke: var(--bad); stroke-width: 2; }
.pipe .hitdot { stroke: var(--bad); stroke-width: 2.5; }
.pipe .qmark { fill: var(--bad); }
.pipe .qb { fill: var(--ink); }
.pipe .bt { font-size: 14px; fill: var(--stage); font-weight: 600; }
.pipe .ab { fill: var(--res-bg); stroke: var(--res); stroke-width: 2; }
.pipe .chip { fill: var(--ink); }
.pipe .ct { font-size: 19px; font-weight: 800; fill: var(--stage); }
.pipe .pin { stroke: var(--muted); stroke-width: 3; stroke-linecap: round; }
.pipe .cite { font-size: 10px; font-weight: 700; fill: #1d1c1a; }
.pipe .edge { stroke-linecap: round; stroke-linejoin: round; opacity: .5; }
.pipe .edge.active, .pipe .edge.done, .pipe.lit .edge { opacity: 1; }
"""
    (SITE / "style.css").write_text(css, encoding="utf-8")
print("pipeline redrawn")
