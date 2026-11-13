"""Build site/data.js from data/handbook.md.

Usage:
    python tools/build_data.py            # embeddings, map, clusters, BM25 check
    python tools/build_data.py --gen      # also record real model answers via local Ollama

The functions here are the same ones students write in the notebook, so the
numbers on the slides match the numbers in Colab.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "gemma3:4b"
LOCAL_TAG = __import__("os").environ.get("RAG_LOCAL_MODEL", GEN_MODEL)   # e.g. gemma3:latest if that is how you pulled it
GEN_CACHE = ROOT / "data" / "gen_cache.json"

QUERIES = [  # (id, question, title of the chunk that answers it)
    ("retake", "Can I retake an exam I failed?", "Resits"),
    ("cs240", "Which semester is CS-240 taught in?", "CS-240"),
    ("late", "What happens if I hand in my assignment three days late?", "Late work"),
    ("books", "How many books can I borrow from the library?", "Borrowing"),
    ("quiet", "When is quiet time in the dorms?", "Quiet hours"),
    ("stress", "Where can I get help if I feel stressed?", "Counselling"),
    ("scholarship", "What GPA do I need to keep my scholarship?", "Merit scholarship"),
    ("guests", "Can a friend sleep over in my room?", "Visitors"),
    ("password", "I forgot my password, what do I do?", "Accounts and passwords"),
    ("drop", "Until when can I drop a course without it showing on my transcript?", "Add and drop period"),
    ("ai", "Am I allowed to use ChatGPT for my homework?", "Use of AI tools"),
    ("parking", "How much does a campus parking permit cost?", None),
]
PAIRS = [
    ("Can I retake an exam I failed?", "A student who does not pass a final assessment may attempt it again once."),
    ("Can I retake an exam I failed?", "Shared kitchens are cleaned by staff once a week."),
    ("The library opens at 8:00.", "The library closes at 23:00."),
]
SIMSETS = [  # one simple query, six sentences picked so their real similarities are spread from about 0.95 down to 0
    ('I forgot my password.', [
        'I forgot to do my homework.',
        'The river flows to the sea.',
        'I love pasta with tomato sauce.',
        'I have forgotten my password.',
        'I need help signing in.',
        'I forgot my passcode.',
    ]),
    ('Can I retake an exam I failed?', [
        'I failed to catch the bus this morning.',
        'Bananas are yellow.',
        'Students may borrow up to eight books.',
        'Am I allowed to retake an exam that I failed?',
        'A student who does not pass a final assessment may attempt it again once.',
        'What happens if I fail an exam?',
    ]),
    ('Where can I eat on campus?', [
        'Tuition is due in week two.',
        'My laptop battery died.',
        'The exam starts at nine.',
        'Where on campus can I eat?',
        'What is on the lunch menu today?',
        "I'm hungry, are there food places at the university?",
    ]),
]
GEN_QUERIES = ["retake", "cs240", "late", "parking"]
POOL = 5   # each search method nominates its top 5 before fusion
STOPWORDS = set("a an the i my me you your is are am do does did can could what when where which who how "
                "if in on at to for of and or it be with from that this until much many there have has".split())


# ---------- the pipeline (mirrors the notebook) ----------

def read_handbook(pdf_path):
    """The handbook's text, one block per paragraph (the same lines as in the notebook)."""
    from pypdf import PdfReader
    pages = [page.extract_text(extraction_mode="layout") for page in PdfReader(str(pdf_path)).pages]
    return "\n\n".join(pages)


def load_chunks(text):
    chunks = []
    for block in text.split("\n\n"):
        block = " ".join(block.split())          # undo the PDF's line breaks
        if len(block) > 60:                      # skip the title and the section headings
            chunks.append({"id": f"c{len(chunks) + 1:02d}", "section": "", "text": block})
    return chunks


def tokenize(text):
    words = re.findall(r"[a-z]+-\d+|[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


REC_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]   # paragraphs, lines, sentences, words, characters


def recursive_ranges(text, size, overlap):
    """Where LangChain's RecursiveCharacterTextSplitter cuts this text: a list of [start, end]."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separators=REC_SEPARATORS,
                                        keep_separator="end", add_start_index=True)
    out = [[d.metadata["start_index"], d.metadata["start_index"] + len(d.page_content)] for d in sp.create_documents([text])]
    assert all(a >= 0 for a, _ in out)
    return out


REC_DEMO_TEXT = (   # invented residence rules: a short, a medium and a long paragraph
    "Welcome to the residence. Reception is open every day from 8:00 to 20:00.\n\n"
    "Quiet hours start at 22:30 on weekdays. One overnight visitor is allowed for up to three nights a month. "
    "The visitor must be signed in at the desk before 21:00.\n\n"
    "Shared kitchens are cleaned once a week. Residents must wash their own dishes on the day they use them, "
    "label everything they keep in the shared fridge with their name and room number, and take out the rubbish "
    "whenever the bin is full. Anything left in the fridge after the Friday clear-out is thrown away."
)


def recursive_demo():
    """The passage and where LangChain really cuts it at each maximum size (no overlap)."""
    sizes = range(80, 401, 20)
    return {"text": REC_DEMO_TEXT, "sizes": [sizes.start, sizes.stop - 1, sizes.step],
            "real": {str(sz): recursive_ranges(REC_DEMO_TEXT, sz, 0) for sz in sizes}}


def recursive_grid(text, sizes, overlaps):
    """Real splitter output for every slider position the chunking demo can reach."""
    return {"sizes": [sizes.start, sizes.stop - 1, sizes.step], "ovs": [overlaps.start, overlaps.stop - 1, overlaps.step],
            "grid": {f"{sz}|{ov}": recursive_ranges(text, sz, ov) for sz in sizes for ov in overlaps if ov <= sz // 2}}


def rrf(rankings, weights=None, k=60):
    weights = weights or [1.0] * len(rankings)
    scores = {}
    for ranking, w in zip(rankings, weights):
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def build_prompt(question, context_chunks, strict=True, cite=True):
    lines = ["You are a helpful assistant for Moghazy University students."]
    if not context_chunks:
        lines += ["", f"Question: {question}", "Answer:"]
        return "\n".join(lines)
    if strict:
        lines.append("Answer the question using ONLY the context below.")
        lines.append('If the answer is not in the context, say "I don\'t know based on the handbook."')
    else:
        lines.append("Use the context below to help answer the question.")
    if cite:
        lines.append("Cite the chunk ids you used in square brackets, like [c12].")
    lines += ["", "Context:"]
    lines += [f"[{c['id']}] {c['text']}" for c in context_chunks]
    lines += ["", f"Question: {question}", "Answer:"]
    return "\n".join(lines)


# ---------- build ----------

def main():
    text = read_handbook(ROOT / "data" / "handbook.pdf")
    chunks = load_chunks(text)
    texts = [c["text"] for c in chunks]
    n = len(chunks)
    print(f"{n} chunks, {len(text.split())} words")

    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(texts, normalize_embeddings=True)
    q_emb = model.encode([q for _, q, _ in QUERIES], normalize_embeddings=True)
    pa = model.encode([a for a, _ in PAIRS], normalize_embeddings=True)
    pb = model.encode([b for _, b in PAIRS], normalize_embeddings=True)
    pairs = [{"a": a, "b": b, "score": round(float(x @ y), 3)} for (a, b), x, y in zip(PAIRS, pa, pb)]
    print(pairs)
    simsets = []
    for q, sents in SIMSETS:
        v = model.encode([q] + sents, normalize_embeddings=True)
        simsets.append({"query": q, "vec": [round(float(x), 3) for x in v[0][:64]],
                        "sentences": [{"text": t, "score": round(float(v[0] @ e), 3)} for t, e in zip(sents, v[1:])]})
        print(q, [(t[:28], x["score"]) for t, x in zip(sents, simsets[-1]["sentences"])])

    bm25 = BM25Okapi([tokenize(t) for t in texts])

    xy = TSNE(n_components=2, metric="cosine", perplexity=8, init="pca",
              random_state=0).fit_transform(emb)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))

    km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(emb)
    centroids = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)

    queries, gen_ctx = [], {}
    idx_of = {c["id"]: i for i, c in enumerate(chunks)}
    for (qid, q, gold_title), qe in zip(QUERIES, q_emb):
        gold = next((c["id"] for c in chunks if gold_title and c["text"].startswith(gold_title)), None)
        sem = emb @ qe
        sem_rank = [int(i) for i in np.argsort(-sem, kind="stable")]
        kw = bm25.get_scores(tokenize(q))
        kw_rank = [int(i) for i in np.argsort(-kw, kind="stable")]
        hyb_rank = rrf([sem_rank[:POOL], [i for i in kw_rank[:POOL] if kw[i] > 0]])
        cl_rank = [int(i) for i in np.argsort(-(centroids @ qe))]
        gi = idx_of.get(gold)
        print(f"\n[{qid}] {q}  gold {gold}: sem #{sem_rank.index(gi) + 1 if gold else 0},"
              f" bm25 #{kw_rank.index(gi) + 1 if gold else 0},"
              f" hybrid #{hyb_rank.index(gi) + 1 if gold and gi in hyb_rank else 0}")
        print("  semantic:", [(chunks[i]["id"], round(float(sem[i]), 3)) for i in sem_rank[:4]])
        print("  bm25    :", [(chunks[i]["id"], round(float(kw[i]), 2)) for i in kw_rank[:4]])
        print("  hybrid  :", [chunks[i]["id"] for i in hyb_rank[:4]])
        print("  ivf     : best chunk in cluster", int(km.labels_[sem_rank[0]]),
              "| probe order", cl_rank[:3])
        gen_ctx[qid] = [chunks[i]["id"] for i in hyb_rank[:5]]
        queries.append({
            "id": qid, "text": q, "gold": gold,
            "emb": [round(float(v), 4) for v in qe],
            "bm25Top": [[chunks[i]["id"], round(float(kw[i]), 3)] for i in kw_rank[:5]],
            "semTop": [[chunks[i]["id"], round(float(sem[i]), 4)] for i in sem_rank[:5]],
        })

    gen = json.loads(GEN_CACHE.read_text(encoding="utf-8")) if GEN_CACHE.exists() else {}
    if "--gen" in sys.argv:
        import ollama
        by_id = {q["id"]: q for q in queries}
        for qid in GEN_QUERIES:
            q = by_id[qid]["text"]
            ctx = [chunks[idx_of[c]] for c in gen_ctx[qid]]
            variants = [("none", 0, False, False)]
            for k in (1, 3, 5):
                for strict in (False, True):
                    for cite in (False, True):
                        variants.append((f"k{k}_s{int(strict)}_c{int(cite)}", k, strict, cite))
            for key, k, strict, cite in variants:
                full = f"{qid}|{key}"
                if full in gen:
                    continue
                prompt = build_prompt(q, ctx[:k], strict, cite)
                r = ollama.generate(model=LOCAL_TAG, prompt=prompt,
                                    options={"temperature": 0, "seed": 0, "num_predict": 200})
                gen[full] = r["response"].strip()
                print(f"--- {full}\n{gen[full]}\n")
                GEN_CACHE.write_text(json.dumps(gen, indent=1, ensure_ascii=False), encoding="utf-8")



    demo_text = "\n\n".join(next(c["text"] for c in chunks if c["text"].startswith(t)) for t in ["Grade scale", "Late work", "Extensions"])
    chunk_rec = recursive_grid(demo_text, range(60, 701, 20), range(0, 201, 20))

    data = {
        "embedModel": EMBED_MODEL,
        "genModel": GEN_MODEL,
        "handbookWords": len(text.split()),
        "chunks": chunks,
        "emb": [[round(float(v), 4) for v in row] for row in emb],
        "xy": [[round(float(v), 4) for v in row] for row in xy],
        "cluster": [int(c) for c in km.labels_],
        "centroids": [[round(float(v), 4) for v in row] for row in centroids],
        "queries": queries,
        "pairs": pairs,
        "simsets": simsets,
        "genCtx": gen_ctx,
        "gen": gen,
        "chunkRec": chunk_rec,
        "recDemo": recursive_demo(),
    }
    out = ROOT / "site" / "data.js"
    out.write_text("window.RAG_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"\nwrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
