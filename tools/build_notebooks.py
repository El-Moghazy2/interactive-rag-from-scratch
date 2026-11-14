"""Generate notebooks/rag_workshop.ipynb: the complete RAG pipeline and nothing else.

Every code cell is one component of the system and is labelled with its stage:
INDEXING, RETRIEVAL, AUGMENTATION, GENERATION or EVALUATION.

Usage: python tools/build_notebooks.py
"""
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
    CELLS.append(("code", text.strip("\n").replace("__PDF_URL__", PDF_URL), opts))


# ------------------------------------------------------------------ intro
md("""
# Chat with your Docs: build your first RAG

This notebook is the whole system and nothing else. Every cell is one component, labelled with the stage it belongs to:

| Stage | What it does | Parts |
|---|---|---|
| **INDEXING** | Done once, ahead of time: load the document, cut it into chunks, embed them, store them in two indexes | 2, 3, 4, 5 |
| **RETRIEVAL** | For each question: find the chunks that matter | 4, 5 |
| **AUGMENTATION** | Put those chunks into the prompt | 6 |
| **GENERATION** | Let the model write the answer | 1, 6 |
| **EVALUATION** | Score the result (optional) | 8 |

**Before anything else:** `Runtime → Change runtime type → T4 GPU`, then run the Setup cell. It takes about two minutes, so start it now.
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

# ------------------------------------------------------------------ part 1
md("""
## Part 1 · The model on its own  `GENERATION`

`ask_llm` sends a prompt to the model and returns its reply. It is the generation component, and the only thing the model ever sees is that prompt.

Ask about Moghazy University before the model has been given anything. The handbook's rules are invented, so it cannot know them.
""")

code('''
# GENERATION: send a prompt to the model, get text back
import ollama

def ask_llm(prompt):
    reply = ollama.generate(model=MODEL, prompt=prompt, options={"temperature": 0})
    return reply["response"].strip()

question = "Can I retake an exam I failed?"
print(ask_llm(f"You are a helpful assistant for Moghazy University students.\\n\\nQuestion: {question}\\nAnswer:"))
''')

# ------------------------------------------------------------------ part 2
md("""
## Part 2 · Load the document and cut it into chunks  `INDEXING`

**Load.** The handbook is a PDF. We download it, read the text page by page, tidy the spacing and drop the title and section headings.
""")

code('''
# INDEXING, step 1: load the document
import urllib.request
from pypdf import PdfReader

def load_pdf(path):
    pages = [page.extract_text(extraction_mode="layout") for page in PdfReader(path).pages]
    blocks = [" ".join(block.split()) for block in "\\n\\n".join(pages).split("\\n\\n")]   # one tidy block per paragraph
    return "\\n\\n".join(block for block in blocks if len(block) > 60)                    # drop title and headings

urllib.request.urlretrieve("__PDF_URL__", "handbook.pdf")
document = load_pdf("handbook.pdf")

print(len(document.split()), "words")
print(document[:300])
''')

md("""
**Chunk.** LangChain's recursive splitter cuts at the biggest natural break that keeps a chunk under `chunk_size`: blank lines first, then line breaks, full stops, spaces.
""")

code('''
# INDEXING, step 2: cut the document into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

def make_splitter(size=300, overlap=50):
    return RecursiveCharacterTextSplitter(
        chunk_size=size,                               # the longest a chunk may be, in characters
        chunk_overlap=overlap,                         # how much two neighbours share
        separators=["\\n\\n", "\\n", ". ", " ", ""],     # biggest break first
        keep_separator="end")                          # a full stop stays with its sentence

pieces = make_splitter().split_text(document)
chunks = [{"id": f"c{n + 1:02d}", "text": text} for n, text in enumerate(pieces)]
texts = [c["text"] for c in chunks]

print(len(chunks), "chunks")
print(chunks[19])
''')

# ------------------------------------------------------------------ part 3
md("""
## Part 3 · Embed the chunks  `INDEXING`

The embedding model turns each chunk into 384 numbers that capture its meaning. `normalize_embeddings=True` gives every vector length 1, so comparing two of them is a plain dot product (their cosine similarity).
""")

code('''
# INDEXING, step 3: turn every chunk into a vector
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(texts, normalize_embeddings=True)

print(emb.shape)   # (number of chunks, 384)
''')

# ------------------------------------------------------------------ part 4
md("""
## Part 4 · The vector index, and searching it  `INDEXING` `RETRIEVAL`

A vector index stores the chunk vectors and answers one question: which stored vectors are nearest to this one? `IndexFlatIP` compares with every vector, so it is exact.
""")

code('''
# INDEXING, step 4: store the vectors in a vector index
import faiss

index = faiss.IndexFlatIP(emb.shape[1])       # an empty index for 384-number vectors
index.add(emb)                                # store every chunk's vector

print(index.ntotal, "vectors in the index")
''')

code('''
# RETRIEVAL: search by meaning
def semantic_search(question, k=3):
    q = embedder.encode([question], normalize_embeddings=True)       # embed the question with the same model
    scores, ids = index.search(q, k)                                 # ask the index for the k nearest chunks
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

def show(results):
    for i, score in results:
        print(f"{chunks[i]['id']}  {score:6.3f}  {chunks[i]['text'][:90]}...")

show(semantic_search("Can I retake an exam I failed?"))
''')

# ------------------------------------------------------------------ part 5
md("""
## Part 5 · Keyword search and hybrid search  `INDEXING` `RETRIEVAL`

Search by meaning is weak on exact codes and names. BM25 is the classic keyword scorer: a chunk scores when it contains the question's words, and rare words count more. It needs its own index, built once like the vector index.
""")

code('''
# INDEXING, step 5: build the keyword index
import re
import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = set("a an the i my me you your is are am do does did can could what when where which who how "
                "if in on at to for of and or it be with from that this until much many there have has".split())

def tokenize(text):                      # lower-case words, keep codes like cs-240, drop filler words
    words = re.findall(r"[a-z]+-\\d+|[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]

bm25 = BM25Okapi([tokenize(t) for t in texts])
''')

code('''
# RETRIEVAL: search by keyword
def keyword_search(question, k=3):
    scores = bm25.get_scores(tokenize(question))
    top = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]   # only chunks that matched
''')

md("""
Each search wins on different questions, so the retriever uses both. **Reciprocal rank fusion** ignores the raw scores and uses positions: a chunk earns `1 / (60 + rank)` from every list it appears in.
""")

code('''
# RETRIEVAL: combine the two searches (this is the retriever the rest of the system calls)
def rrf(rankings, k=60):
    points = {}
    for ranking in rankings:                         # one ranked list per search method
        for rank, i in enumerate(ranking, start=1):
            points[i] = points.get(i, 0) + 1 / (k + rank)
    return sorted(points.items(), key=lambda item: -item[1])

def hybrid_search(question, k=3, pool=5):
    by_meaning = [i for i, _ in semantic_search(question, k=pool)]   # from the vector index
    by_keyword = [i for i, _ in keyword_search(question, k=pool)]    # from the keyword index
    return rrf([by_meaning, by_keyword])[:k]

for q in ["Which semester is CS-240 taught in?", "Can I retake an exam I failed?"]:
    print(q)
    show(hybrid_search(q))
    print()
''')

# ------------------------------------------------------------------ part 6
md("""
## Part 6 · Build the prompt and answer  `AUGMENTATION` `GENERATION`

**Augmentation** is plain text assembly: the retrieved chunks go into the prompt, each with its id, next to rules that tell the model to stay inside them.
""")

code('''
# AUGMENTATION: put the retrieved chunks and the question into one prompt
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
''')

md("""
**The whole system** is now three steps in one function: retrieve, augment, generate.
""")

code('''
# The RAG pipeline: RETRIEVAL -> AUGMENTATION -> GENERATION
def rag(question, k=3):
    results = hybrid_search(question, k)             # retrieve
    prompt = build_prompt(question, results)         # augment
    return ask_llm(prompt)                           # generate

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
## Part 7 · Chat with *your* document (optional)

Indexing is the only part that depends on the document. `build_index` runs its five steps again on any PDF; retrieval, augmentation and generation stay exactly as they are.
""")

code('''
# INDEXING, all five steps in one function
def build_index(path, strip_headings=True):
    global document, chunks, texts, emb, index, bm25
    document = load_pdf(path) if strip_headings else "\\n\\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    chunks = [{"id": f"c{n + 1:02d}", "text": " ".join(t.split())} for n, t in enumerate(make_splitter().split_text(document))]
    texts = [c["text"] for c in chunks]
    emb = embedder.encode(texts, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    print(path, "->", len(chunks), "chunks indexed")
''')

code('''
from google.colab import files
uploaded = files.upload()
pdf_path = next(iter(uploaded))
''', colab_only=True)

code('''
build_index(pdf_path, strip_headings=False)       # an arbitrary PDF: keep all of its text
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
## Part 8 · Measure it with Ragas (optional)  `EVALUATION`

Ragas turns "this answer looks right" into scores from 0 to 1, one per part of the system:

- **faithfulness**: is every claim in the answer backed by the retrieved chunks?
- **context recall**: did we retrieve everything the reference answer needs?
- **context precision**: are the useful chunks ranked above the useless ones?
- **answer accuracy**: does the answer agree with the reference answer?

Ragas needs an LLM as the judge. We use the same Ollama model through Ollama's OpenAI-compatible address, so no API key is needed. A small judge is lenient and sometimes wrong: read its scores as a signal, not a verdict.
""")

code('''
!pip -q install ragas "langchain-community<0.4"
''', colab_only=True)

code('''
# EVALUATION: the judge, and a tiny test set of questions with correct answers written by a person
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, ContextRecall, ContextPrecision, AnswerAccuracy

judge = llm_factory(MODEL, client=AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))

build_index("handbook.pdf")                       # back to the handbook, in case Part 7 replaced it

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
# EVALUATION: run the pipeline on one question and score each part of it
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
Now break the system on purpose and watch which score moves. With `k=1` the retake question gets only one chunk, and it is the wrong one (the absence rule). Before you run it: which of the four scores should drop?
""")

code('''
await evaluate(tests[0][0], tests[0][1], k=1)
''')

md("""
When we ran this with `gemma3:4b` as the judge, answer accuracy fell to 0.50 and faithfulness to 0.83, which is right: the answer is about illness, not failing. But context recall and context precision stayed at 1.00, which is **wrong**: the one retrieved chunk does not contain the resit rule. A small judge makes mistakes like this. Read a few cases yourself, and use a stronger judge model when the numbers matter.
""")

md("""
## Where to go next

- **Something wrong?** Print `show(hybrid_search(q, 5))` first. If the right chunk is not there, it is an indexing or retrieval problem. If it is there, it is an augmentation or generation problem.
- **Experiment:** change `make_splitter(size, overlap)` and call `build_index("handbook.pdf")` again, or change `k` in `rag`. Change one thing at a time.
- **Reranking:** retrieve 20 chunks, then let a stronger model pick the best 3.
- **Bigger collections:** keep the vectors in a vector database (Chroma, Qdrant, pgvector) instead of in memory.
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
    print("wrote", out, "with", len(CELLS), "cells")
