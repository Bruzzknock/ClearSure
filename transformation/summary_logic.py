import os
import json
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from LLMs import one_sentence_summary, propose_split_spans, remove_think_block

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

"""
Recursive one‑sentence summarizer with **LLM‑directed splitting**
================================================================
Version ▸ July 2025

• Uses the atomic helper‑prompts you keep in *LLMs.py* (`one_sentence_summary`,
  `propose_split_spans`).
• Dual back‑end: **OpenAI** *or* **Ollama‑deepseek‑r1:14b** – auto‑detected or
  forced via `--backend` CLI flag.
• Writes the summary tree to `output.json` ➜ later consumed by KG importer.

───────────────────────────────────────────────────────────────────────────────
Path layout
───────────
BASE_DIR/structured/
    ├─ final_kg.json      (future)
    ├─ import_kg.cypher   (future)
    └─ output.json        (summary tree – this run)

Run example
───────────
$ python recursive_summarizer.py docs/paper.txt \
      --backend ollama --max-depth 6 --max-children 5
"""

# ───────────────────────── Paths & constants ────────────────────────── #

BASE_DIR = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = BASE_DIR / "structured"
STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)

FINAL_KG_PATH = STRUCTURED_DIR / "final_kg.json"   # reserved for future KG export
OUT_PATH = STRUCTURED_DIR / "import_kg.cypher"     # reserved for Cypher import script
OUTPUT_JSON_PATH = STRUCTURED_DIR / "output.json"  # human‑readable tree dump

MODEL_DEFAULT = "gpt-4o-mini"  # only used for OpenAI; Ollama model comes from env / CLI

# ─────────────────────────── LLM back‑end ───────────────────────────── #

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore

try:
    from langchain_community.llms import Ollama  # type: ignore
except ImportError:
    Ollama = None  # type: ignore


class LLMWrapper:
    """Unified interface for OpenAI ChatCompletion *or* Ollama."""

    def __init__(self, backend: str, model_name: str):
        self.backend = backend
        self.model_name = model_name

        if backend == "openai":
            if openai is None:
                raise RuntimeError("openai package not installed")
            # Nothing else required; ChatCompletion will read OPENAI_API_KEY.
            self._system = {"role": "system", "content": "You are a helpful assistant."}

        elif backend == "ollama":
            if Ollama is None:
                raise RuntimeError("langchain-community not installed for Ollama backend")

            host = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_HOST_PC")
            if not host:
                raise EnvironmentError("Set OLLAMA_HOST or OLLAMA_HOST_PC for Ollama backend")
            self._ollama_model = Ollama(
                model=model_name,
                base_url=host,
                temperature=0.0,
                num_ctx=8192,
            )
        else:
            raise ValueError("backend must be 'openai' or 'ollama'")

    # ------------------------------------------------------------------
    def invoke(self, prompt: str) -> str:
        """Minimalist single‑prompt call so helper functions can do `model.invoke()`."""
        if self.backend == "openai":
            resp = openai.ChatCompletion.create(
                model=self.model_name,
                temperature=0.0,
                messages=[
                    self._system,
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message["content"].strip()

        # Ollama: `self._ollama_model` already supports direct prompt.
        return self._ollama_model.invoke(prompt).strip()


# Global default wrapper – initialised in __main__
GLOBAL_LLM: Optional[LLMWrapper] = None


# ─────────────────────── Summarisation helpers ──────────────────────── #


def summarize(text: str) -> str:
    """Compress *text* to one sentence using helper‑prompt in LLMs.py"""
    return one_sentence_summary(text, GLOBAL_LLM)  # type: ignore[arg-type]


def propose_splits(text: str):
    """Delegate split decision to helper‑prompt in LLMs.py"""
    return propose_split_spans(text, GLOBAL_LLM)  # type: ignore[arg-type]


# ────────────────────────── Data classes ────────────────────────────── #

@dataclass
class Node:
    text: str
    summary: str
    children: List["Node"] = field(default_factory=list)

    def to_dict(self):
        return {
            "summary": self.summary,
            "children": [c.to_dict() for c in self.children],
        }


# ───────────────────────── Recursion driver ─────────────────────────── #


def build_tree(
    text: str,
    depth: int = 0,
    max_depth: int = 6,
    min_chars: int = 400,
    max_children: int = 10,
) -> Node:
    """Recursively build a summary tree for *text*."""

    #summary_sentence = summarize(text)
    node = Node(text=text, summary=None)

    # -- Stopping conditions --
    if depth >= max_depth or len(text) <= min_chars:
        return node

    spans = propose_splits(text)
    if not spans:  # LLM opted not to split
        return node

    for span in spans:
        sub_text = text[span["start"] : span["end"] + 1]  # end inclusive ➜ +1
        child = build_tree(
            sub_text,
            depth=depth + 1,
            max_depth=max_depth,
            min_chars=min_chars,
            max_children=max_children,
        )
        node.children.append(child)

    return node


# ────────────────────────────── CLI ─────────────────────────────────── #

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Recursive LLM summarizer with LLM‑directed splitting"
    )
    parser.add_argument("path", type=Path, help="Path to plain‑text file to summarize")
    parser.add_argument("--backend", choices=["auto", "openai", "ollama"], default="auto",
                        help="Which back‑end to use (default: auto‑detect)")
    parser.add_argument("--model", default=MODEL_DEFAULT, help="Model name (OpenAI or Ollama)")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum recursion depth")
    parser.add_argument("--max-children", type=int, default=5, help="Maximum children per node")
    parser.add_argument("--min-chars", type=int, default=400, help="Min chars before split")
    args = parser.parse_args()

    # -------------------- Initialise LLM wrapper --------------------- #
    if args.backend == "auto":
        if os.getenv("OPENAI_API_KEY") and openai is not None:
            backend = "openai"
            model_name = args.model
        else:
            backend = "ollama"
            model_name = "deepseek-r1:14b" if args.model == MODEL_DEFAULT else args.model
    else:
        backend = args.backend
        model_name = args.model

    GLOBAL_LLM = LLMWrapper(backend=backend, model_name=model_name)

    # ------------------------- Read file ----------------------------- #
    full_text = args.path.read_text(encoding="utf-8")

    # --------------------- Build summary tree ------------------------ #
    root = build_tree(
        full_text,
        max_depth=args.max_depth,
        max_children=args.max_children,
        min_chars=args.min_chars,
    )

    tree_dict = root.to_dict()

    # 1) Print to stdout for immediate inspection
    json.dump(tree_dict, sys.stdout, indent=2, ensure_ascii=False)
    print()

    # 2) Save human & machine‑readable dump
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(tree_dict, fh, indent=2, ensure_ascii=False)
    print(f"\n[✓] Summary tree written to {OUTPUT_JSON_PATH.relative_to(BASE_DIR)}")

    # (Optional) hooks for future KG export – uncomment when needed
    # with open(FINAL_KG_PATH, "w", encoding="utf-8") as kg:
    #     json.dump(tree_dict, kg, ensure_ascii=False)

    # with open(OUT_PATH, "w", encoding="utf-8") as cypher:
    #     cypher.write("// TODO: generate Cypher import script\n")
