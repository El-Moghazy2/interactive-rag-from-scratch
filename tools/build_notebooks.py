"""Generate notebooks/rag_workshop.ipynb: the complete, runnable workshop notebook.

Usage: python tools/build_notebooks.py
"""
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
REPO = "El-Moghazy2/interactive-rag-from-scratch"
PDF_URL = f"https://raw.githubusercontent.com/{REPO}/main/data/handbook.pdf"

CELLS = []


def md(text):
    CELLS.append(("md", text.strip(), {}))


def code(text, **opts):
    CELLS.append(("code", text.strip("\n"), opts))


# ------------------------------------------------------------------ intro
md("""
# Chat with your Docs: build your first RAG

Follow the slides, then run the cells for each part. All the code is complete: read it, run it, then change a number and run it again.

**Before anything else:** `Runtime → Change runtime type → T4 GPU`, then run the Setup cell below. It takes about two minutes, so start it now.

""")

code('''
#@title 0 · Setup: run me first (about 2 minutes)
MODEL = "gemma3:4b"   # no GPU? use "gemma3:1b" instead

!pip -q install sentence-transformers faiss-cpu rank_bm25 pypdf ollama langchain-text-splitters
!sudo apt-get -qq install -y zstd pciutils > /dev/null 2>&1
!curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1

import subprocess, time
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)
!ollama pull {MODEL}
print("Setup done. Model ready:", MODEL)
''', colab_only=True, form=True)

code('''
#@title A small helper for printing search results (run, no need to read)
def show(results):
    """Print search results as: id, score, start of the text."""
    for i, score in results:
        print(f"{chunks[i]['id']}  {score:6.3f}  {chunks[i]['text'][:90]}...")

print("Helper ready.")
''', form=True)

# ------------------------------------------------------------------ part 1
md("""
## Part 1 · The problem: the model has never read your document

The rules in our Moghazy University handbook are invented for this workshop, so the model cannot know them. Ask anyway.
""")

code('''
import ollama

def ask_llm(prompt):
    reply = ollama.generate(model=MODEL, prompt=prompt, options={"temperature": 0})
    return reply["response"].strip()

question = "Can I retake an exam I failed?"
print(ask_llm(f"You are a helpful assistant for Moghazy University students.\\n\\nQuestion: {question}\\nAnswer:"))
''')

md("Now the open-book version: paste the right paragraph into the prompt by hand. This is all RAG does. The rest of the workshop is about finding that paragraph automatically.")

code('''
paragraph = """Resits. A student who does not pass a final assessment may attempt it again once, during the August resit period. The grade for a repeated attempt is capped at C, and a third attempt is not permitted."""

print(ask_llm(f"Answer using only this context.\\n\\nContext: {paragraph}\\n\\nQuestion: {question}\\nAnswer:"))
''')

# ------------------------------------------------------------------ part 2
md("""
## Part 2 · Chunking

First we need the text. The handbook is a PDF, like most real documents. We download it from the workshop's GitHub repository, then read it page by page. `extraction_mode="layout"` keeps the gaps between paragraphs as blank lines, which we will use in a moment.
""")

code('''
import urllib.request
from pypdf import PdfReader

PDF_URL = "__PDF_URL__"
urllib.request.urlretrieve(PDF_URL, "handbook.pdf")        # download the handbook next to this notebook

reader = PdfReader("handbook.pdf")
pages = [page.extract_text(extraction_mode="layout") for page in reader.pages]
HANDBOOK = "\\n\\n".join(pages)

print(len(pages), "pages,", len(HANDBOOK.split()), "words")
print(HANDBOOK[:420])
'''.replace("__PDF_URL__", PDF_URL))

md("""
Now cut it up. The simplest idea first: cut every `size` characters, and let neighbours share `overlap` characters. Run it and look at where the cuts land.
""")

code('''
def chunk_text(text, size=300, overlap=50):
    step = size - overlap                      # how far the window moves each time
    pieces = []
    for start in range(0, len(text), step):
        pieces.append(text[start:start + size])
    return pieces

late_rule = HANDBOOK[HANDBOOK.index("Late work."):HANDBOOK.index("Extensions.")]
late_rule = " ".join(late_rule.split())        # one paragraph, without the PDF's line breaks
for piece in chunk_text(late_rule, size=120, overlap=0):
    print(repr(piece))
''')

md("""
The cuts land in the middle of words and sentences. Change `overlap=0` to `overlap=40` and run it again: neighbours now share some text, but words are still cut.

### Recursive splitting

LangChain's `RecursiveCharacterTextSplitter` has a list of places it prefers to cut and works down it. A piece that is still too long is split with the next separator in the list, and small neighbours are glued back together until they are just under `chunk_size`.

The setting that matters most is the list of separators, from the biggest break to the smallest. Five strings: a blank line `"\\n\\n"`, a line break `"\\n"`, a full stop followed by a space `". "`, a space `" "`, and the empty string `""`, which means "anywhere".
""")

code('''
from langchain_text_splitters import RecursiveCharacterTextSplitter

def make_splitter(size=300, overlap=50):
    return RecursiveCharacterTextSplitter(
        chunk_size=size,                       # the longest a chunk may be, in characters
        chunk_overlap=overlap,                 # how much two neighbours share
        separators=["\\n\\n", "\\n", ". ", " ", ""],
        keep_separator="end")                  # a full stop stays with its sentence
''')

code('''
for size in (120, 200):
    print("chunk_size =", size)
    for piece in make_splitter(size, 0).split_text(late_rule):
        print("  ", repr(piece))
''')

md("""
At 120 each sentence is longer than a chunk, so the splitter has to go down its list to spaces: no word is broken, but the sentences still are. At 200 each sentence fits, so every chunk is a whole sentence. The lesson: pick a chunk size that is bigger than your sentences. Then try `make_splitter(200, 60)`.

The handbook is already written as one rule per paragraph, and the PDF reader gave us a blank line between paragraphs. So for it we split on blank lines, which is the first separator in that list. In Part 7 your own PDF has no such tidy structure, so there we use `make_splitter`.
""")

code('''
chunks = []
for block in HANDBOOK.split("\\n\\n"):
    text = " ".join(block.split())               # undo the PDF's line breaks
    if len(text) > 60:                           # skip the title and the section headings
        chunks.append({"id": f"c{len(chunks) + 1:02d}", "text": text})

texts = [c["text"] for c in chunks]
print(len(chunks), "chunks")
print(chunks[19])
''')

# ------------------------------------------------------------------ part 3
md("""
## Part 3 · Embeddings

An embedding model turns a text into a list of numbers (a vector). Texts with similar meaning get vectors that point in similar directions.
""")

code('''
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")
v = embedder.encode("resit policy")
print(v.shape)
print(v[:8])
''')

md("""
Cosine similarity is the dot product of two vectors divided by the product of their lengths. `a @ b` is the dot product, `np.linalg.norm(a)` is the length.
""")

code('''
def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
''')

code('''
a, b, c = embedder.encode(["Can I retake an exam I failed?",
                           "A student who does not pass a final assessment may attempt it again once.",
                           "Shared kitchens are cleaned once a week."])
print("question vs resit rule  :", round(float(cosine(a, b)), 3))
print("question vs kitchen rule:", round(float(cosine(a, c)), 3))
''')

md("Embed every chunk once. `normalize_embeddings=True` makes every vector length 1, so from now on cosine similarity is just a dot product.")

code('''
emb = embedder.encode(texts, normalize_embeddings=True)
print(emb.shape)   # (number of chunks, 384)
''')

# ------------------------------------------------------------------ part 4
md("""
## Part 4 · The vector index

A vector index stores the chunk vectors and answers one question fast: *which stored vectors are nearest to this one?* We use FAISS. `IndexFlatIP` compares with every vector, so it is exact. "IP" means inner product, which equals cosine similarity here because our vectors have length 1.

To search, we ask the index for the `k` nearest vectors. `index.search(q, k)` returns two arrays: the scores and the positions of the chunks.
""")

code('''
import faiss

index = faiss.IndexFlatIP(emb.shape[1])       # an empty index for 384-number vectors
index.add(emb)                                # store every chunk's vector
print(index.ntotal, "vectors in the index")

def semantic_search(question, k=3):
    q = embedder.encode([question], normalize_embeddings=True)
    scores, ids = index.search(q, k)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
''')

code('''
show(semantic_search("Can I retake an exam I failed?"))
''')

md("What did the index just do? For a flat index, exactly this: score the question against every chunk and keep the best `k`. The three lines below do it by hand and should print the same chunks.")

code('''
q = embedder.encode("Can I retake an exam I failed?", normalize_embeddings=True)
scores = emb @ q                              # one score per chunk
top = np.argsort(-scores)[:3]                 # positions of the 3 largest scores
print("by hand :", [chunks[i]["id"] for i in top])
print("the index:", [chunks[i]["id"] for i, _ in semantic_search("Can I retake an exam I failed?")])
''')

md("Comparing with everything is fine for 47 chunks. For millions, an index takes a shortcut. The IVF index below is the one from the slides: it groups the vectors into clusters and only looks inside the nearest ones. Watch what happens with one cluster open.")

code('''
ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(emb.shape[1]), emb.shape[1], 6, faiss.METRIC_INNER_PRODUCT)
ivf.train(emb)                                 # finds 6 clusters
ivf.add(emb)

q = embedder.encode(["Can I retake an exam I failed?"], normalize_embeddings=True)
print("exact         :", [chunks[i]["id"] for i in index.search(q, 3)[1][0]])
for nprobe in (1, 3):
    ivf.nprobe = nprobe                        # how many clusters to open
    print(f"ivf, nprobe={nprobe}:", [chunks[i]["id"] for i in ivf.search(q, 3)[1][0] if i >= 0])
''')

md("Our pipeline keeps the exact index from here on, so later parts never miss a chunk because of the index.")

# ------------------------------------------------------------------ part 5
md("""
## Part 5 · Keyword search and hybrid search

Semantic search is weak on exact codes and names. Try it (the right chunk starts with `CS-240`):
""")

code('''
show(semantic_search("Which semester is CS-240 taught in?"))
''')

md("BM25 is the classic keyword scorer: a chunk scores high when it contains the question's words, and rare words count more.")

code('''
import re
from rank_bm25 import BM25Okapi

STOPWORDS = set("a an the i my me you your is are am do does did can could what when where which who how "
                "if in on at to for of and or it be with from that this until much many there have has".split())

def tokenize(text):                      # lower-case words, keep codes like cs-240, drop filler words
    words = re.findall(r"[a-z]+-\\d+|[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]

bm25 = BM25Okapi([tokenize(t) for t in texts])

def keyword_search(question, k=3):
    scores = bm25.get_scores(tokenize(question))
    top = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]   # only chunks that matched

print(tokenize("Which semester is CS-240 taught in?"))
show(keyword_search("Which semester is CS-240 taught in?"))
print()
show(keyword_search("Can I retake an exam I failed?"))   # and here keywords lose
''')

md("""
Each method wins on a different question, so combine them. **Reciprocal rank fusion** ignores the raw scores and uses only positions: a chunk earns `1 / (60 + rank)` from every list it appears in.

In the code below, look for the one line that gives a chunk its points for being at position `rank`.
""")

code('''
def rrf(rankings, k=60):
    points = {}
    for ranking in rankings:                         # one ranked list per search method
        for rank, i in enumerate(ranking, start=1):
            points[i] = points.get(i, 0) + 1 / (k + rank)
    return sorted(points.items(), key=lambda item: -item[1])

def hybrid_search(question, k=3, pool=5):
    by_meaning = [i for i, _ in semantic_search(question, k=pool)]   # from the FAISS vector index
    by_keyword = [i for i, _ in keyword_search(question, k=pool)]    # from the BM25 keyword index
    return rrf([by_meaning, by_keyword])[:k]
''')

code('''
for q in ["Which semester is CS-240 taught in?", "Can I retake an exam I failed?"]:
    print(q)
    show(hybrid_search(q))
    print()
''')

# ------------------------------------------------------------------ part 6
md("""
## Part 6 · Generation

Retrieval found the chunks. Now put them in the prompt, with rules: use only the context, admit when the answer is missing, cite the chunk ids.

The first line of `build_prompt` turns the retrieved chunks into the context text, one line per chunk, like `[c20] Resits. A student who does not pass ...`
""")

code('''
def build_prompt(question, results):
    context = "\\n".join(f"[{chunks[i]['id']}] {chunks[i]['text']}" for i, _ in results)
    return f"""You are a helpful assistant for Moghazy University students.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the handbook."
Cite the chunk ids you used in square brackets, like [c12].

Context:
{context}

Question: {question}
Answer:"""

def rag(question, k=3):
    results = hybrid_search(question, k)
    return ask_llm(build_prompt(question, results))
''')

code('''
print(build_prompt("Can I retake an exam I failed?", hybrid_search("Can I retake an exam I failed?")))
''')

code('''
for q in ["Can I retake an exam I failed?",
          "Which semester is CS-240 taught in?",
          "What happens if I hand in my assignment three days late?",
          "How much does a campus parking permit cost?"]:      # not in the handbook!
    print("Q:", q)
    print("A:", rag(q))
    print()
''')

# ------------------------------------------------------------------ part 7
md("""
## Part 7 · Chat with *your* document

Upload any text PDF (a syllabus, lecture notes, a paper). The same pipeline runs on it, this time cut with your `make_splitter`: new chunks, new vector index, new keyword index.
""")

code('''
from google.colab import files
uploaded = files.upload()
pdf_path = next(iter(uploaded))
''', colab_only=True)

code('''
from pypdf import PdfReader

pages = [page.extract_text() or "" for page in PdfReader(pdf_path).pages]
doc = "\\n\\n".join(pages)                             # keep the line breaks: the splitter uses them

chunks = [{"id": f"c{n + 1:02d}", "text": " ".join(t.split())} for n, t in enumerate(make_splitter(500, 100).split_text(doc))]
texts = [c["text"] for c in chunks]
emb = embedder.encode(texts, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1])          # rebuild the vector index
index.add(emb)
bm25 = BM25Okapi([tokenize(t) for t in texts])   # and the keyword index
print(len(pages), "pages ->", len(chunks), "chunks. Index rebuilt.")
''')

code('''
while True:
    q = input("Ask your document (empty line to stop): ")
    if not q.strip():
        break
    print(rag(q), "\\n")
''', colab_only=True)

# ------------------------------------------------------------------ part 8
md("""
## Part 8 · Measure it with Ragas (optional)

So far you judged answers by reading them. Ragas turns that into scores from 0 to 1, one per part of the system:

- **faithfulness**: is every claim in the answer backed by the retrieved chunks?
- **context recall**: did we retrieve everything the reference answer needs?
- **context precision**: are the useful chunks ranked above the useless ones?
- **answer accuracy**: does the answer agree with the reference answer?

Ragas needs an LLM as the judge. We use the same Ollama model, through Ollama's OpenAI-compatible address, so no API key is needed. A small judge is lenient and sometimes wrong: read its scores as a signal, not a verdict.
""")

code('''
!pip -q install ragas "langchain-community<0.4"
''', colab_only=True)

code('''
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, ContextRecall, ContextPrecision, AnswerAccuracy

judge = llm_factory(MODEL, client=AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))

# Part 7 replaced the handbook with your PDF, so load the handbook again
chunks = []
for block in HANDBOOK.split("\\n\\n"):
    text = " ".join(block.split())
    if len(text) > 60:
        chunks.append({"id": f"c{len(chunks) + 1:02d}", "text": text})
texts = [c["text"] for c in chunks]
emb = embedder.encode(texts, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1])          # rebuild the vector index
index.add(emb)
bm25 = BM25Okapi([tokenize(t) for t in texts])   # and the keyword index

# a tiny test set: questions with the correct answer written by a person
tests = [
    ("Can I retake an exam I failed?",
     "Yes. A failed final assessment can be attempted again once, in the August resit period, and the grade for that attempt is capped at C."),
    ("Which semester is CS-240 taught in?",
     "CS-240 Databases is offered in the spring semester only."),
    ("What happens if I hand in my assignment three days late?",
     "Late coursework loses 10 percent of the available marks for each calendar day, so three days late loses 30 percent."),
]
''')

code('''
async def evaluate(question, reference, k=3):
    results = hybrid_search(question, k)                        # your retrieval
    contexts = [chunks[i]["text"] for i, _ in results]
    answer = rag(question, k)                                   # your generation
    metrics = {
        "faithfulness":      Faithfulness(llm=judge).ascore(user_input=question, response=answer, retrieved_contexts=contexts),
        "context recall":    ContextRecall(llm=judge).ascore(user_input=question, retrieved_contexts=contexts, reference=reference),
        "context precision": ContextPrecision(llm=judge).ascore(user_input=question, reference=reference, retrieved_contexts=contexts),
        "answer accuracy":   AnswerAccuracy(llm=judge).ascore(user_input=question, response=answer, reference=reference),
    }
    print("Q:", question)
    print("A:", answer)
    scores = {}
    for name, job in metrics.items():
        try:
            scores[name] = (await job).value
            print(f"   {name:18} {scores[name]:.2f}")
        except Exception as e:
            print(f"   {name:18} failed ({type(e).__name__})")
    print()
    return scores

for question, reference in tests:
    await evaluate(question, reference)
''')

md("""
Now break the system on purpose and watch which score moves. With `k=1` the retrake question gets only one chunk, and it is the wrong one (the absence rule). Before you run it: which of the four scores should drop, and which should stay high?
""".replace("retrake", "retake"))

code('''
await evaluate(tests[0][0], tests[0][1], k=1)
''')

md("""
When we ran this with `gemma3:4b` as the judge, answer accuracy fell to 0.50 and faithfulness to 0.83, which is right: the answer is about illness, not failing. But context recall and context precision stayed at 1.00, which is **wrong**: the one retrieved chunk does not contain the resit rule. A small judge makes mistakes like this. Always read a few cases yourself, and use a stronger judge model when the numbers matter.
""")

md("""
Try your own experiments: change `k`, switch `hybrid_search` for `semantic_search` or `keyword_search` inside `evaluate`, or remove the "ONLY the context" rule from `build_prompt`. Change one thing at a time and compare the scores on the same questions.
""")

md("""
## Where to go next

- **Something wrong?** Print `show(hybrid_search(q, 5))` first. If the right chunk is not there, it is a retrieval problem (chunking, embeddings, search). If it is there, it is a generation problem (prompt, model).
- **Reranking:** retrieve 20 chunks, then let a stronger model pick the best 3.
- **Evaluation at scale:** write 20 to 50 questions with reference answers and run Part 8 on all of them after every change. For a judge you can trust more, use a stronger model than the one being judged.
- **Bigger documents:** keep the vectors in a vector database (Chroma, Qdrant, pgvector) instead of a NumPy array.
""")


# ------------------------------------------------------------------ build
def build():
    nb = new_notebook()
    for kind, text, opts in CELLS:
        if kind == "md":
            nb.cells.append(new_markdown_cell(text))
            continue
        meta = {"tags": []}
        if opts.get("form"):
            meta["cellView"] = "form"
        if opts.get("colab_only"):
            meta["tags"].append("colab-only")
        nb.cells.append(new_code_cell(text, metadata=meta))
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


if __name__ == "__main__":
    out = ROOT / "notebooks" / "rag_workshop.ipynb"
    nbformat.write(build(), out)
    print("wrote", out)
