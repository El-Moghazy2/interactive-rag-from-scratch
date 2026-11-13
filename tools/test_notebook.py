"""Run the notebook's code locally, skipping Colab-only cells.

Needs a local Ollama with the model below. Usage: python tools/test_notebook.py [notebook] [first_cell]
"""
import ast
import asyncio
import inspect
import os
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
which = sys.argv[1] if len(sys.argv) > 1 else "rag_workshop.ipynb"
only_from_md = sys.argv[2] if len(sys.argv) > 2 else None      # e.g. "Part 8": run setup cells, then only that part
nb = nbformat.read(ROOT / "notebooks" / which, as_version=4)

env = {"MODEL": os.environ.get("RAG_LOCAL_MODEL", "gemma3:4b"), "pdf_path": str(ROOT / "data" / "handbook.pdf")}
loop = asyncio.new_event_loop()
QUIET = ("ask_llm(", "rag(q", "show(")                          # slow or noisy calls skipped when jumping to a part
reached = only_from_md is None
for n, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        if os.environ.get("RAG_STOP_AT") and os.environ["RAG_STOP_AT"] in cell.source:
            print("stopped before:", os.environ["RAG_STOP_AT"])
            break
        if only_from_md and only_from_md in cell.source:
            reached = True
        continue
    tags = cell.metadata.get("tags", [])
    if "colab-only" in tags:
        continue
    src = cell.source
    if not reached:                                             # keep definitions, drop the demo calls
        src = "\n".join(l for l in src.splitlines() if not (l.startswith(("print(", "for q in", "    print(", "    show(", "show(", "check(")) and any(k in l for k in QUIET + ("check(", "print(", "for q in"))))
    print(f"\n===== cell {n} =====", flush=True)
    try:
        result = eval(compile(src, f"cell{n}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), env)
        if inspect.iscoroutine(result):
            loop.run_until_complete(result)
    except SyntaxError:
        if reached:
            raise
