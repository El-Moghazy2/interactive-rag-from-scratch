# Interactive RAG from Scratch

Learn retrieval-augmented generation from scratch, with interactive visual demos you can play with in the browser and a complete Colab notebook. Written for beginners: some Python is enough, no machine-learning background needed.

**[Open the site](https://el-moghazy2.github.io/interactive-rag-from-scratch/)** · **[90-minute deck](https://el-moghazy2.github.io/interactive-rag-from-scratch/site-short/)** · **[Full deck](https://el-moghazy2.github.io/interactive-rag-from-scratch/site/)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/El-Moghazy2/interactive-rag-from-scratch/blob/main/notebooks/rag_workshop.ipynb)

## What is inside

Each component is taught the same way: watch it fail, play with a live demo, then run the code.

| Part | You learn | Interactive demo |
|---|---|---|
| The idea | Why a model needs your documents | A model answering with and without the document |
| Chunking | Fixed size, overlap, recursive splitting | Sliders that recolour the text; a step-by-step recursive splitter |
| Embeddings | Text as numbers, cosine similarity | One query against six sentences; a map of every chunk |
| Vector index | Flat, clustered (IVF) and graph (HNSW) indexes | 600 points where a faster search can miss |
| Hybrid search | BM25 keywords, embeddings, rank fusion | Three rankings side by side with a trust slider |
| Generation | Grounded prompts, citations, "I don't know" | Build the prompt and watch recorded answers change |
| RAG vs long context | Cost, caching, when each one wins | Prompt caching, a decision guide |
| Evaluation | Five Ragas metrics | Draggable score bars with a diagnosis |

The notebook builds a complete prototype: PDF reading, chunking with LangChain's recursive splitter, embeddings (`all-MiniLM-L6-v2`), a FAISS vector index, BM25, rank fusion, and generation with a local model through Ollama. No API key is needed.

Everything on the slides is real: similarity scores, search results and model answers were computed or recorded, not written by hand. The handbook used in the examples is for "Moghazy University", which does not exist.

## Running it yourself

The decks are plain HTML: open `site/index.html` or `site-short/index.html` in a browser. The notebook runs as is in Google Colab on a free T4 GPU.

---

The rest of this page is the presenter's guide.

## Presenter's guide: Chat with your Docs

A 2-hour workshop: 90 minutes of content, 30 minutes of questions. Students from year 2 onward, some Python, no ML needed.

Every component follows the same rhythm: **watch it fail → predict → play with the demo → read the code → quick check → run it in Colab.**

## Two versions of the deck

| | `site/` (full) | `site-short/` (90 minutes) |
|---|---|---|
| Slides | 47 | 36 |
| Estimated time | about 110 minutes | about 90 minutes |
| Demo data | the 47-chunk handbook, the same data as the notebook | a toy FAQ of 16 one-line facts in four topics (exams, library, housing, rooms) |
| Numbers match Colab | yes | no: slides show the toy example, Colab uses the handbook |
| Left out | nothing | quizzes, the index code slide, the PDF upload slide, the cache comparison, cost calculator and benchmark slides, the Ragas code slide |

Both use the same notebook. The notebook is complete: there are no blanks to fill in. Students run each part after its slides, read the code, and change values to see what happens. In the short version Parts 7 and 8 are take-home.

### Run of show, short version

| Min | Part | Slides | Colab |
|---|---|---|---|
| 0–9 | 1 · The idea | title → colab1 | Setup at minute 0, then Part 1 (3 min) |
| 9–19 | 2 · Chunking | map-2 → chunk-code | Part 2 (4 min) |
| 19–35 | 3 · Embeddings | map-3 → emb-code | Part 3 (4 min) |
| 35–42 | 4 · Vector index | map-4 → idx-what | Part 4 (2 min) |
| 41–54 | 5 · Hybrid search | map-5 → hyb-code | Part 5 (5 min) |
| 54–67 | 6 · Generation | map-6 → gen-code | Part 6 (5 min) |
| 67–72 | 7 · Wrap-up | clinic, recap | take-home: Parts 7 and 8 |
| 72–82 | 8 · RAG vs long context | lc-question → lc-decide | none |
| 82–90 | 9 · Evaluation | ev-why, ev-bars | none |

These timings are estimates, not a timed run. Toy cases to use: *retake an exam* (keyword search returns nothing useful, embeddings find it), *room B-204* (embeddings rank B-240 first, keywords get it right), the index demo's *near a border* button with one cluster open (the index misses; it uses 600 made-up 2-D points, not the FAQ, so the saving is visible), parking (not in the documents).

## What is here

| Path | What it is |
|---|---|
| `site/index.html` | The full interactive slide deck (47 slides). Open it in a browser; no build, no server needed. |
| `site-short/index.html` | The 90-minute deck (36 slides) with toy examples. Its data comes from `tools/build_short.py`. |
| `site/data.js` | Real embeddings, map, clusters and recorded model answers used by the demos. Generated. |
| `notebooks/rag_workshop.ipynb` | The Colab notebook: complete, runnable code for every part, no blanks. Part 8 (optional) scores the pipeline with Ragas. |
| `data/handbook.md`, `data/handbook.pdf` | The workshop handbook for Moghazy University (all rules invented) that everything runs on. |
| `tools/` | Scripts that regenerate the data, the notebooks and the PDF. |

## Driving the deck

`→` / `Space` / clicker: next step, then next slide. `←`: back. `O`: all slides. `F`: fullscreen. `T`: light or dark. The bottom bar jumps between parts. Every slide has its own link (`index.html#hyb-demo`), so students can follow on their laptops.

The demos work offline. The "type your own" button on the two search demos downloads a small embedding model into the browser (about 25 MB, needs internet); without it the prepared questions still work.

## Run of show

| Min | Part | Slides | Ask the room before the demo | Colab |
|---|---|---|---|---|
| 0–8 | 1 · The idea | title → colab1 | "Will the model know our university's resit rule?" | Setup cell at minute 0, then Part 1 (3 min) |
| 8–17 | 2 · Chunking | chunk-why → chunk-quiz | "How big should a chunk be? What breaks at each extreme?" | Part 2 (5 min) |
| 17–33 | 3 · Embeddings | emb-why → emb-quiz | "One query, six sentences: rank them from closest to furthest." Then reveal the real scores. | Part 3 (5 min) |
| 33–42 | 4 · Vector index | idx-why → idx-quiz | "We skip most vectors. Can that give a wrong answer?" Use *feeling stressed* with 1 cluster, then 2. | Part 4 (5 min) |
| 42–55 | 5 · Hybrid search | hyb-why → hyb-quiz | "Which search wins on CS-240? Which on *retake an exam*?" Then drag the trust slider. | Part 5 (6 min) |
| 55–68 | 6 · Generation | gen-why → gen-quiz | "The handbook has nothing on parking. What will the model say with no documents? With documents and rules?" | Part 6 (6 min) |
| 68–78 | 7 · Your docs | clinic → recap | Three broken RAGs: which part do you fix? | Part 7: upload a PDF and chat (8 min) |
| 78–90 | 8 · RAG vs long context | lc-question → end | "Our handbook fits in any prompt. Was today a waste?" | none |
| extra | 9 · Evaluation with Ragas | ev-why → ev-code | "You changed the chunk size. Is the system better or worse?" On the bars slide, click *Faithful but wrong* and ask which part to fix. | Part 8 (optional) |
| 90–120 | Questions | press `O` to jump back to any demo | | |

If you run late: shorten Part 4 (show the demo, skip its code slide), and let Part 7's upload continue into the question time.

## Before the day

1. Everything is hosted from this repository: the decks on GitHub Pages and the notebook through the Colab link above. The title slide shows the address.
2. The notebook downloads `data/handbook.pdf` from this repository, so the repository must stay public and keep its name.
3. Run the notebook once on a free Colab **T4** runtime and time the Setup cell. It installs Ollama and pulls `gemma3:4b` (3.3 GB). If the pull takes more than 3 minutes on the venue network, change `MODEL` to `gemma3:1b` in the Setup cell.
4. Add your benchmark chart as `site/benchmark.png`; it appears on the "measured" slide of Part 8.
5. Do one timed dry run.

## Regenerating

```bash
python -m venv .venv
```

```bash
.venv/Scripts/pip install sentence-transformers faiss-cpu rank_bm25 scikit-learn pypdf fpdf2 nbformat ollama langchain-text-splitters qrcode ragas "langchain-community<0.4"
```

```bash
.venv/Scripts/python tools/build_data.py --gen
```

```bash
.venv/Scripts/python tools/build_notebooks.py
```

```bash
.venv/Scripts/python tools/test_notebook.py
```

`build_short.py` (same flags) rebuilds `site-short/data.js` from the toy corpus defined inside it. `build_data.py` rebuilds `site/data.js` from `data/handbook.md`. With `--gen` it also records real answers from a local Ollama (`gemma3:4b`; set `RAG_LOCAL_MODEL` if your local tag differs) into `data/gen_cache.json`. Delete that file to record again. If you edit the handbook, rerun all three scripts so the slides, the notebook and the checks stay in step.

## Design notes worth knowing when you present

- Chunking is taught in three steps: the fixed-size demo (with overlap), the recursive-splitting demo, then the code. The recursive demo has Play and Step buttons and a maximum-size slider; it animates the rounds of LangChain's `RecursiveCharacterTextSplitter` on a three-paragraph passage. It is my own step-by-step version of the algorithm, checked in the page against the library's real output at every slider position (`RECDEMO.check()` in the browser console returns an empty list when they agree). The "recursive (LangChain)" mode in the first chunking demo shows the library's real output directly. In the notebook, `chunk_text` (fixed size) comes first to show the bad cuts, then `make_splitter` wraps the recursive splitter. The handbook is still split on blank lines (one rule per paragraph, the splitter's first separator), so chunk ids match the slides; the student's own PDF in Part 7 is cut with `make_splitter(500, 100)`.
- The notebook is a complete prototype: chunks, embeddings, a FAISS vector index (`IndexFlatIP`, exact), a BM25 keyword index, rank fusion of the two, and generation. `semantic_search` queries the FAISS index; the three-line NumPy version is shown once as "what the index does inside". The clustered (IVF) index is a side experiment and is not used by the pipeline, so later parts never fail because of the index.
- The handbook is for "Moghazy University", which does not exist; its first paragraph says so and that every rule in it is invented. Money amounts are in dollars and are invented like everything else. The handbook is written in `data/handbook.md`, turned into `data/handbook.pdf` by `tools/build_pdf.py`, and the PDF is what the notebook and `tools/build_data.py` read, with the same few lines of code, so the 47 chunks on the slides are the 47 chunks in Colab.
- The overview picture groups the steps by component: Retrieval (indexing ahead of time, then search per question), Augmentation, Generation.
- Notebook Part 8 runs real Ragas (faithfulness, context recall, context precision, answer accuracy) with the Ollama model as the judge, through Ollama's OpenAI-compatible address. Ragas 0.4.3 failed to import next to langchain-community 0.4, so the install cell pins `langchain-community<0.4`. Response relevancy is left out: with gemma3:4b as the judge it returned 0 for a good answer in testing.
- The invented rules mean no model can know the handbook. Planted cases: *retake an exam* shares no keywords with the resit rule (keyword search misses, embeddings find it); *CS-240* is confused with CS-204 by embeddings (keyword search finds it); the late-work rule breaks across fixed-size chunks; parking is not in the handbook at all.
- Hybrid search here lets each method nominate its top 5 and fuses them with reciprocal rank fusion. With these questions the right chunk is always in the hybrid top 3, though not always first. That is worth saying: hybrid is a safety net, not a guarantee.
- Keyword search drops a short list of filler words ("can", "I", "the"). Without that, filler words drown the signal on short questions.
- Recorded answers (gemma3:4b, temperature 0) showed two real failures, and the slides use them as they happened: with no retrieval the model invents parking prices; with only one retrieved chunk (the wrong one) it answers the resit question confidently from the absence rule. With three chunks it answered correctly and said "I don't know" for parking, even when the strict rule was switched off. Smaller models are less careful, so keep the rule.
- `tools/make_pipeline_svg.py` redraws the pictogram diagram on the "whole system" slide.
- The 2D map is a t-SNE projection, so screen distances are approximate. All scores shown are real cosine similarities from `all-MiniLM-L6-v2`, the same model the notebook uses.
- Each of Parts 2 to 6 opens with a "where we are" slide: the overview picture with the current step lit.
- Part 9 (evaluation, 3 slides, about 6 minutes) comes on top of a full 90 minutes together with Part 8. Choose: drop Part 8 or Part 9, shorten Part 7, or let one run into the question time. Its middle slide has five draggable score bars with example levels; nothing on it is measured.
- Part 8 has two caching slides: how prompt caching reuses an identical prefix (and what breaks it), and a comparison of the KV cache, prompt cache, embeddings and index, and answer cache. Part 8 is now 7 slides in 12 minutes; skip the benchmark slide or the calculator if you run late.
- Part 8 summarises Martin Keen's IBM Technology video "Is RAG Still Needed?" and the benchmark set-up from your LinkedIn post. The cost calculator uses OpenAI's list prices for four GPT-6 models (Standard tier, read from the pricing page on 12 Nov 2026) and shows the formula under each bar. The prices live in the `PRICES` table in `site/demos.js`; check them against the pricing page before the workshop.
