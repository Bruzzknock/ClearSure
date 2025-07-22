#!/usr/bin/env python3
"""Build a hierarchical topic tree from a text or PDF document using an LLM."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from uuid import uuid4

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass

os.environ["OLLAMA_HOST"] = os.environ.get(
    "OLLAMA_HOST_PC", os.environ.get("OLLAMA_HOST", "")
)

import nltk
import pdfplumber
from neo4j import GraphDatabase
from langchain_ollama.llms import OllamaLLM
from LLMs import label_text, sentence_topic_same, remove_think_block

VERBOSE = False


def log(msg: str) -> None:
    if VERBOSE:
        print(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_context_window(model) -> int:
    known = {
        "gpt-3.5-turbo": 16384,
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
    }
    if isinstance(model, str):
        key = model
    else:
        key = getattr(model, "model", "")
    return known.get(key, 8192)


def extract_text(path: Path) -> str:
    """Return plain text from *path*.

    Supports PDF via ``pdfplumber`` or reads the file as UTF-8 text otherwise.
    """
    if path.suffix.lower() == ".pdf":
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)

    return path.read_text(encoding="utf-8")



# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

@dataclass
class Node:
    name: str
    char_start: int
    char_end: int
    parent: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
    children: List["Node"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "children": [c.to_dict() for c in self.children],
        }


# ---------------------------------------------------------------------------
# Topic discovery
# ---------------------------------------------------------------------------

TOKENIZER = nltk.data.load("tokenizers/punkt/english.pickle")


def _ensure_length(text: str, limit: int) -> str:
    tokens = text.split()
    if len(tokens) <= limit:
        return text
    return " ".join(tokens[:limit])




def phase2(
    text: str,
    parent: Node,
    offset: int,
    model,
    ctx_limit: int,
) -> None:
    spans = list(TOKENIZER.span_tokenize(text))
    if not spans:
        return
    first_start, first_end = spans[0]
    first_sentence = text[first_start:first_end]
    raw_label = label_text(_ensure_length(first_sentence, ctx_limit), model)
    if "<think>" in raw_label and VERBOSE:
        print(raw_label)
    label = remove_think_block(raw_label).strip()
    end = first_end
    for i in range(1, len(spans)):
        sent = text[spans[i][0] : spans[i][1]]
        sent_trunc = _ensure_length(sent, ctx_limit)
        if sentence_topic_same(label, sent_trunc, model):
            end = spans[i][1]
        else:
            node = Node(
                name=label,
                char_start=offset + first_start,
                char_end=offset + end - 1,
                parent=parent.id,
            )
            log(f"🪧 New node: {label} [{node.char_start}-{node.char_end}]")
            parent.children.append(node)
            phase2(text[end:], parent, offset + end, model, ctx_limit)
            return
    node = Node(
        name=label,
        char_start=offset + first_start,
        char_end=offset + spans[-1][1] - 1,
        parent=parent.id,
    )
    log(f"🪧 New node: {label} [{node.char_start}-{node.char_end}]")
    parent.children.append(node)


def build_tree(text: str, model) -> Node:
    ctx = get_context_window(model)
    log("🌳 Building topic tree")
    raw_root = label_text(_ensure_length(text, ctx // 2), model)
    if "<think>" in raw_root and VERBOSE:
        print(raw_root)
    root_name = remove_think_block(raw_root).strip()
    log(f"🌲 Root topic: {root_name}")
    root = Node(name=root_name, char_start=0, char_end=len(text) - 1, parent=None)
    phase2(text, root, 0, model, ctx // 2)
    return root


# ---------------------------------------------------------------------------
# Neo4j integration
# ---------------------------------------------------------------------------



def push_to_neo4j(root: Node, uri: str, user: str, password: str) -> None:
    log(f"🔗 Connecting to Neo4j at {uri}")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        def _create(node: Node, parent_id: str | None):
            log(f"➡️  Pushing node '{node.name}' ({node.id})")
            session.run(
                "MERGE (n:Topic {id:$id}) SET n.name=$name, n.char_start=$cs, n.char_end=$ce",
                id=node.id,
                name=node.name,
                cs=node.char_start,
                ce=node.char_end,
            )
            if parent_id:
                session.run(
                    "MATCH (p:Topic {id:$pid}),(c:Topic {id:$cid}) "
                    "MERGE (p)-[:HAS_CHILD]->(c)",
                    pid=parent_id,
                    cid=node.id,
                )
            for child in node.children:
                _create(child, node.id)

        _create(root, None)
    driver.close()
    log("✅ Finished pushing to Neo4j")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("..\structured\input.txt"),
        help="PDF or UTF-8 text document",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "deepseek-r1:14b"),
        help="Model name for the provider",
    )
    p.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-pass", default="neo4j")
    p.add_argument("--out", default="topic_tree.json")
    p.add_argument("--verbose", action="store_true", help="Print progress")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global VERBOSE
    VERBOSE = args.verbose
    model = OllamaLLM(
        model=args.model,
        base_url=os.environ["OLLAMA_HOST"],
        options={"num_ctx": 8192},
        temperature=0.0,
    )
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    log(f"📄 Reading {args.input}")
    text = extract_text(Path(args.input))
    tree = build_tree(text, model)
    log(f"💾 Writing tree to {args.out}")
    Path(args.out).write_text(json.dumps(tree.to_dict(), indent=2), encoding="utf-8")
    push_to_neo4j(tree, args.neo4j_uri, args.neo4j_user, args.neo4j_pass)


if __name__ == "__main__":
    main()
