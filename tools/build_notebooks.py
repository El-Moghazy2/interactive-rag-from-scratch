"""Generate notebooks/rag_workshop.ipynb: the complete RAG pipeline and nothing else.

Every code cell is one component of the system and is labelled with its stage:
INDEXING, RETRIEVAL, AUGMENTATION, GENERATION or EVALUATION.

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


FIG_URL = "https://el-moghazy2.github.io/interactive-rag-from-scratch/notebooks/figures/"
FIGURES = []


def figure(match):
    """[[file|caption]] becomes a centred, numbered figure."""
    name, caption = match.group(1), match.group(2)
    FIGURES.append(name)
    return (f'<p align="center"><img src="{FIG_URL}{name}.svg" alt="Figure {len(FIGURES)}" width="900"></p>\n'
            f'<p align="center"><em><b>Figure {len(FIGURES)}.</b> {caption}</em></p>')


def md(text):
    CELLS.append(("md", re.sub(r"\[\[(\w+)\|(.+?)\]\]", figure, text.strip()), {}))


def code(text, **opts):
    CELLS.append(("code", text.strip("\n").replace("__PDF_URL__", PDF_URL), opts))


# ------------------------------------------------------------------ intro
md("""
# Chat with your Docs: build your first RAG

Imagine a chatbot that answers students' questions about their university: when the resit period is, what a late hand-in costs, which semester a course runs in. A large language model (LLM) writes fluent answers, but it only knows what was in its training data, and your university's handbook was not. Ask it anyway and it will often give an answer that sounds right and is invented.

**Retrieval-Augmented Generation (RAG)** fixes this without touching the model. Before the model answers, the system looks up the passages of your documents that matter for the question and puts them into the prompt. The model then answers from those passages, and can say where each fact came from.

In this notebook you build that system from scratch, one component at a time, and at the end you measure how well it works.

## Why not just fine-tune the model?

There are three ways to get an answer about your own documents out of an LLM.

[[fig1_three_ways|Three ways to answer a question about your own documents. Fine-tuning changes the model; RAG changes what the model reads.]]

- **Ask the model as it is.** There is nothing to build, but the model has never seen your documents, so it guesses.
- **Fine-tune it.** Training continues on your own data until the knowledge sits in the model's weights. That needs labelled examples, GPU time and experience, and it has to be repeated whenever a rule changes. The model still cannot tell you where an answer came from.
- **Use RAG.** The model stays as it is. The knowledge lives in the documents, so updating the system means editing a file, and every answer can point to its source.

For questions about facts in documents, RAG is the first thing to try. Fine-tuning remains the right tool for teaching a model a style or a skill rather than facts.

## What is RAG?

A RAG system works in two phases.

[[fig2_pipeline|The system you are about to build. The top row runs once; the bottom row runs for every question. Each stage keeps its colour in all the figures below.]]

**Indexing** happens once, before anyone asks anything: the document is loaded, cut into small passages called chunks, and stored in a form that can be searched quickly. **Answering** happens for every question, in the three steps that give RAG its name: **retrieve** the chunks that matter, **augment** the prompt with them, and let the model **generate** the answer.

Every code cell below is one component of this picture and starts with a comment naming its stage:

| Stage | What it does | Parts |
|---|---|---|
| **INDEXING** | Done once: load the document, cut it into chunks, embed them, store them in two indexes | 2, 3, 4, 5 |
| **RETRIEVAL** | For each question: find the chunks that matter | 4, 5 |
| **AUGMENTATION** | Put those chunks into the prompt | 6 |
| **GENERATION** | Let the model write the answer | 1, 6 |
| **EVALUATION** | Score the result (optional) | 8 |

## The use case: a student handbook

Our document is the handbook of **Moghazy University**: five pages of rules about exams, courses, fees, housing and the library. The university does not exist, and every rule in the handbook was invented for this workshop. That is deliberate. No model can have seen these rules during training, so when the system answers correctly, you know it was retrieval that did it.

**Before anything else:** `Runtime → Change runtime type → T4 GPU`, then run the Setup cell. It installs the libraries and downloads the language model, which takes about two minutes, so start it now.
""")

code('''
#@title 0 · Setup: run me first (about 2 minutes)
MODEL = "gemma3:4b"   # no GPU? use "gemma3:1b" instead

# every library is installed here, before anything is imported, so all versions match
!pip -q install sentence-transformers faiss-cpu rank_bm25 pypdf ollama langchain-text-splitters ragas "langchain-community<0.4"
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

We start at the end of the pipeline, with the component that writes the answer. The language model runs on this machine through [Ollama](https://ollama.com). `ask_llm` sends it a prompt and returns its reply; `temperature=0` makes the model choose its most likely next word every time, so the same prompt always gives the same answer.

That small function is the whole generation component, and it shows the fact everything else is built on: **the only thing the model ever sees is the prompt.** The rest of this notebook exists to put the right text into that prompt.

First, ask about Moghazy University with nothing but the question. Read the reply critically: is anything in it specific to this university?
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

The reply above is fluent and generic. It cannot contain the real rule, because the model has never read the handbook. So we give the system the handbook, and that begins **indexing**, the preparation that happens once.

### Step 1: load

The handbook is a PDF. `pypdf` reads its text page by page. We tidy the spacing and drop the title and the section headings, which leaves one paragraph per rule.
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
### Step 2: chunk

Why not hand the model the whole handbook with every question? Long prompts are slow and cost more, a small model loses facts that sit in the middle of a long text, and a real document collection would not fit at all. So we cut the document into **chunks**: short passages that can be found, and used, on their own.

Where to cut matters. A cut in the middle of a sentence leaves two halves that each mean little. LangChain's `RecursiveCharacterTextSplitter` avoids that by trying the biggest natural break first.

[[fig3_chunking|Recursive splitting. The splitter cuts at blank lines first; only a piece that is still longer than the chunk size is cut again, one level further down.]]

Two settings control it:

- `chunk_size` is the longest a chunk may be, in characters. Too large, and one chunk mixes several topics, so it matches every question a little and none well. Too small, and a rule is separated from the words that give it meaning.
- `chunk_overlap` is how many characters two neighbouring chunks share when a long passage has to be cut, so that a sentence at the border is not lost.

Every paragraph of our handbook is shorter than 300 characters, so each rule becomes exactly one chunk.
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

To find the chunks that matter for a question, the computer has to compare meanings, and it can only compare numbers. An **embedding model** turns a text into a list of numbers, a *vector*, in such a way that texts with similar meanings get vectors pointing in similar directions.

[[fig4_embeddings|An embedding turns text into a vector. The angles are drawn from this notebook's real scores: the resit rule points far closer to the question than the quiet-hours rule does. (The 384 dimensions are drawn as two, so read it as a sketch.)]]

Closeness is measured with **cosine similarity**, the cosine of the angle between two vectors: 1 means the same direction, 0 means unrelated. A score only means something next to the other chunks' scores; the best chunk here scores 0.49, and that is enough to win.

We use `all-MiniLM-L6-v2`, a small model that is fast even without a GPU and returns 384 numbers per text. `normalize_embeddings=True` scales every vector to length 1. For vectors of length 1 the cosine similarity is simply the dot product, which is what the index in Part 4 computes.
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

We now have 47 vectors. A **vector index** stores them and answers one question quickly: which of the stored vectors are closest to this one?

[[fig5_vector_index|Search by meaning for "Can I retake an exam I failed?". The index scores every stored vector against the question's vector and returns the three highest.]]

We use FAISS, an open-source library that runs on this machine. `IndexFlatIP` is its simplest index: *flat* means it compares the question with every stored vector, and *IP* stands for inner product, another name for the dot product. That is exact, and instant for 47 chunks. With millions of chunks you would move to an approximate index (IVF, HNSW), which looks at a small part of the vectors and accepts that it sometimes misses the best one, or to a vector database (Chroma, Qdrant, pgvector) that manages this for you.

Building the index is the last indexing step for search by meaning. Searching it is the first piece of **retrieval**: embed the question with the same model, then ask the index for the `k` nearest chunks.
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

Search by meaning has a blind spot: exact names and codes. To an embedding model, `CS-240` and `CS-204` look almost alike, so a question about one course can bring back the other. Keyword search has the opposite character. It knows nothing about meaning, but it never confuses two codes.

**BM25** is the standard keyword scorer. A chunk earns points for every word of the question that it contains, and rare words earn more than common ones. 

The **keyword index** is built during indexing, exactly like the vector index: once, from the same chunks, before any question arrives. `BM25Okapi(...)` reads every chunk and stores which words it contains, how often, and how rare each word is across the whole handbook. At question time only a lookup is left: the question is split into words, and each chunk's score is added up from those stored counts.

[[fig6_keyword|Keyword search for a course code. The word "taught" appears nowhere in the handbook and earns nothing; the rare code cs-240 decides the result.]]
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
### Hybrid search

Each search wins on different questions, so a good retriever uses both. Their scores cannot simply be added, because a cosine similarity and a BM25 score are on different scales. **Reciprocal rank fusion** (RRF) avoids the problem by ignoring the scores and using only the positions: a chunk earns `1 / (60 + rank)` points from every list it appears in.

[[fig7_fusion|Reciprocal rank fusion. Search by meaning alone puts the wrong course first; the chunk that both searches agree on wins.]]

`hybrid_search` is the retriever that the rest of the system calls. Each method nominates its five best chunks, and the fused top `k` go on to the prompt.
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

Retrieval has found the chunks. **Augmentation** hands them to the model, and it is nothing more than assembling text. The wording matters a great deal, though: the prompt is where you tell the model to stay inside the documents.

[[fig8_prompt|Anatomy of the augmented prompt for "Can I retake an exam I failed?", with the three chunks that hybrid search returned.]]

Each of the three rules prevents a specific failure:

- **Use only the context** stops the model from falling back on what it remembers from training.
- **Say "I don't know"** gives the model a way out when the handbook has no answer. Without it, a model would rather invent one.
- **Cite the chunk ids** lets a reader check every answer against its source.
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
### The whole system

Everything is in place. The complete RAG pipeline is three lines: retrieve, augment, generate.

The four questions test different things: a paraphrase (the handbook says *resit*, never *retake*), an exact course code, a rule that needs a little arithmetic, and a question the handbook cannot answer. For the last one, the right answer is "I don't know".
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
## Part 7 · Chat with the handbook (optional)

The pipeline is complete, so wrap it in a loop and ask your own questions. Try three: an easy one, one with an exact name or number, and one the handbook cannot know. When an answer is wrong, print `show(hybrid_search(q, 5))` and decide which part failed.
""")

code('''
while True:
    q = input("Ask the handbook (empty line to stop): ")
    if not q.strip():
        break
    print(rag(q), "\\n")
''', colab_only=True)

# ------------------------------------------------------------------ part 8
md("""
## Part 8 · Measure it with Ragas (optional)  `EVALUATION`

The answers in Part 6 look right. But "looks right on four questions" is not a measurement, and without one you cannot tell whether a change (another chunk size, a different prompt, a larger `k`) made the system better or worse.

There are two ways to evaluate a RAG system:

- **Human evaluation.** People read the answers and grade them. It is the most reliable way, and too slow to repeat after every change.
- **An LLM as the judge.** A language model does the grading. It is fast and repeatable, less reliable, and the usual choice while you are still building.

[Ragas](https://docs.ragas.io) is a library for the second way. You supply questions together with **reference answers** written by a person; Ragas asks a judge model to compare, and returns scores from 0 to 1. Each score looks at a different link in the system, and that is what makes them useful: a low score tells you where to look.

[[fig9_ragas|What each Ragas metric compares. The green metrics grade retrieval, the purple ones grade generation.]]

- **Faithfulness**: is every claim in the answer backed by the retrieved chunks? A low score means the model added things of its own.
- **Context recall**: did retrieval bring back everything the reference answer needs?
- **Context precision**: are the useful chunks ranked above the useless ones?
- **Answer accuracy**: does the answer agree with the reference answer?

The judge here is the same Ollama model, reached through Ollama's OpenAI-compatible address, so no API key is needed. A stronger judge gives more trustworthy scores. A small one is lenient and sometimes wrong: read its scores as a signal, not a verdict.
""")

code('''
# EVALUATION: the judge, and a tiny test set of questions with correct answers written by a person
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, ContextRecall, ContextPrecision, AnswerAccuracy

judge = llm_factory(MODEL, client=AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))

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
### An experiment: break retrieval on purpose

Evaluation earns its keep when you compare two versions of a system. With `k=1` the retake question gets only one chunk, and it is the wrong one (the rule about absence from an exam). Before you run the cell: which of the four scores should drop?
""")

code('''
await evaluate(tests[0][0], tests[0][1], k=1)
''')

md("""
When we ran this with `gemma3:4b` as the judge, answer accuracy fell to 0.50 and faithfulness to 0.83, which is right: the answer is about illness, not about failing. But context recall and context precision stayed at 1.00, which is **wrong**: the one retrieved chunk does not contain the resit rule. A small judge makes mistakes like this. Read a few cases yourself, and use a stronger judge model when the numbers matter.
""")

md("""
## Key takeaways

- **RAG gives a model knowledge it was never trained on by changing the prompt, not the model.** No fine-tuning, and updating the system means updating the documents.
- **Indexing runs once; retrieve, augment and generate run for every question.** Every component you built sits in one of those stages.
- **Each component has its own way of failing.** Chunks cut in the wrong place, a search that confuses two course codes, a prompt that lets the model improvise.
- **Measure, then change one thing at a time.** Scores that point at retrieval or at generation tell you which part to fix.

## Where to go next

- **Something wrong?** Print `show(hybrid_search(q, 5))` first. If the right chunk is not there, it is an indexing or retrieval problem. If it is there, it is an augmentation or generation problem.
- **Experiment:** change `make_splitter(size, overlap)` and run the indexing cells again (Parts 2 to 5), or change `k` in `rag`.
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
