"""Build site-short/data.js: the short deck's toy corpus (16 one-line facts in 4 topics).

Usage:
    python tools/build_short.py            # embeddings, map, clusters
    python tools/build_short.py --gen      # also record model answers via local Ollama

Same pipeline functions as build_data.py; only the corpus and questions differ.
All facts are invented for the workshop.
"""
import json
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_data import (EMBED_MODEL, GEN_MODEL, LOCAL_TAG, POOL, SIMSETS, build_prompt,  # noqa: E402
                        recursive_demo, recursive_grid, rrf, tokenize)

ROOT = Path(__file__).resolve().parent.parent
GEN_CACHE = ROOT / "data" / "gen_cache_short.json"

CORPUS = [
    ("Exams", "Students who do not pass a final assessment may attempt it again once in August, capped at grade C."),
    ("Exams", "Bring your student ID card to every exam."),
    ("Exams", "Doors close 30 minutes after a test starts, and latecomers are not admitted."),
    ("Exams", "If you are ill on exam day, send a medical note within five days."),
    ("Library", "The library is open from 8:00 to 23:00 on weekdays."),
    ("Library", "You can borrow up to eight books for three weeks."),
    ("Library", "An overdue book costs 2 lumens per day."),
    ("Library", "Group study rooms can be booked for two hours a day."),
    ("Housing", "Quiet hours in the residences start at 22:30."),
    ("Housing", "One overnight visitor is allowed for up to three nights a month."),
    ("Housing", "Shared kitchens are cleaned once a week."),
    ("Housing", "Room keys must be returned by noon on the last day of the contract."),
    ("Rooms", "Room B-204 is the algorithms lab."),
    ("Rooms", "Room B-240 is the database lab."),
    ("Rooms", "Room B-402 is the networks lab."),
    ("Rooms", "Lecture halls open at 7:30 every morning."),
]
QUERIES = [  # (id, question, start of the sentence that answers it)
    ("retake", "Can I retake an exam I failed?", "Students who do not pass"),
    ("room", "What is room B-204 used for?", "Room B-204"),
    ("guests", "Can a friend sleep over in my room?", "One overnight visitor"),
    ("books", "How many books can I take home?", "You can borrow"),
    ("quiet", "When do I have to be quiet in the dorm?", "Quiet hours"),
    ("fine", "What is the fine for returning a book late?", "An overdue book"),
    ("sick", "I was sick on the day of my exam, what now?", "If you are ill"),
    ("parking", "How much does a parking permit cost?", None),
]
GEN_QUERIES = ["retake", "room", "fine", "parking"]
CHUNK_DEMO = {
    "text": 'Deadlines are strict and the same rules apply to all courses. Coursework handed in late loses 10 percent of the marks for each day. After five days the mark is zero. Extensions must be requested 48 hours before the deadline. The longest extension is ten days. A medical note is needed for any extension longer than three days. Group projects follow the same rules, and every member receives the same penalty.',
    "keepFrom": "loses 10 percent", "keepTo": "the mark is zero",
    "label": "The late rule (10% per day, zero after five days)",
}


def main():
    chunks = [{"id": f"c{i + 1:02d}", "section": sec, "text": t} for i, (sec, t) in enumerate(CORPUS)]
    texts = [c["text"] for c in chunks]
    n = len(chunks)
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(texts, normalize_embeddings=True)
    q_emb = model.encode([q for _, q, _ in QUERIES], normalize_embeddings=True)
    bm25 = BM25Okapi([tokenize(t) for t in texts])

    dist = np.clip(1 - emb @ emb.T, 0, None)
    xy = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=8).fit_transform(dist)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(emb)
    centroids = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    print("clusters:", [[chunks[i]["id"] for i in range(n) if km.labels_[i] == c] for c in range(4)])

    simsets = []
    for q, sents in SIMSETS:
        v = model.encode([q] + sents, normalize_embeddings=True)
        simsets.append({"query": q, "vec": [round(float(x), 3) for x in v[0][:64]],
                        "sentences": [{"text": t, "score": round(float(v[0] @ e), 3)} for t, e in zip(sents, v[1:])]})

    queries, gen_ctx = [], {}
    idx_of = {c["id"]: i for i, c in enumerate(chunks)}
    for (qid, q, gold_start), qe in zip(QUERIES, q_emb):
        gold = next((c["id"] for c in chunks if gold_start and c["text"].startswith(gold_start)), None)
        sem = emb @ qe
        sem_rank = [int(i) for i in np.argsort(-sem, kind="stable")]
        kw = bm25.get_scores(tokenize(q))
        kw_rank = [int(i) for i in np.argsort(-kw, kind="stable")]
        hyb = rrf([sem_rank[:POOL], [i for i in kw_rank[:POOL] if kw[i] > 0]])
        probe = [int(i) for i in np.argsort(-(centroids @ qe))]
        gi = idx_of.get(gold)
        print(f"[{qid}] {q}  gold {gold}: sem #{sem_rank.index(gi) + 1 if gold else 0}, bm25 #{kw_rank.index(gi) + 1 if gold else 0}"
              f" (score {kw[gi]:.2f})" if gold else f"[{qid}] {q} (no answer)",
              f"hybrid #{hyb.index(gi) + 1 if gold and gi in hyb else 0} | ivf: best in cluster {int(km.labels_[sem_rank[0]])}, probe order {probe}")
        print("     sem:", [(chunks[i]["id"], round(float(sem[i]), 2)) for i in sem_rank[:3]],
              "kw:", [(chunks[i]["id"], round(float(kw[i]), 2)) for i in kw_rank[:3]], "hyb:", [chunks[i]["id"] for i in hyb[:3]])
        gen_ctx[qid] = [chunks[i]["id"] for i in hyb[:5]]
        queries.append({"id": qid, "text": q, "gold": gold, "emb": [round(float(v), 4) for v in qe],
                        "bm25Top": [[chunks[i]["id"], round(float(kw[i]), 3)] for i in kw_rank[:5]],
                        "semTop": [[chunks[i]["id"], round(float(sem[i]), 4)] for i in sem_rank[:5]]})

    gen = json.loads(GEN_CACHE.read_text(encoding="utf-8")) if GEN_CACHE.exists() else {}
    if "--gen" in sys.argv:
        import ollama
        for qid in GEN_QUERIES:
            q = next(x["text"] for x in queries if x["id"] == qid)
            ctx = [chunks[idx_of[c]] for c in gen_ctx[qid]]
            variants = [("none", 0, False, False)] + [(f"k{k}_s{int(s)}_c{int(c)}", k, s, c)
                                                      for k in (1, 3) for s in (False, True) for c in (False, True)]
            for key, k, strict, cite in variants:
                full = f"{qid}|{key}"
                if full in gen:
                    continue
                r = ollama.generate(model=LOCAL_TAG, prompt=build_prompt(q, ctx[:k], strict, cite),
                                    options={"temperature": 0, "seed": 0, "num_predict": 200})
                gen[full] = r["response"].strip()
                print(f"--- {full}\n{gen[full]}\n", flush=True)
                GEN_CACHE.write_text(json.dumps(gen, indent=1, ensure_ascii=False), encoding="utf-8")

    data = {
        "embedModel": EMBED_MODEL, "genModel": GEN_MODEL, "handbookWords": sum(len(t.split()) for t in texts),
        "chunks": chunks, "topics": sorted({c["section"] for c in chunks}, key=[c["section"] for c in chunks].index),
        "emb": [[round(float(v), 4) for v in row] for row in emb],
        "xy": [[round(float(v), 4) for v in row] for row in xy],
        "cluster": [int(c) for c in km.labels_],
        "centroids": [[round(float(v), 4) for v in row] for row in centroids],
        "queries": queries, "pairs": [], "simsets": simsets, "genCtx": gen_ctx, "gen": gen, "chunkDemo": CHUNK_DEMO,
        "recDemo": recursive_demo(),
        "chunkRec": recursive_grid(CHUNK_DEMO["text"], range(100, 301, 10), range(0, 151, 10)),
    }
    out = ROOT / "site-short" / "data.js"
    out.write_text("window.RAG_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
