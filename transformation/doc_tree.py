#!/usr/bin/env python3
"""Build a hierarchical topic tree from a text or PDF document using an LLM."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
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
try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None
import requests

VERBOSE = False


def log(msg: str) -> None:
    if VERBOSE:
        print(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    return len(text.split())


def get_context_window(model: str) -> int:
    known = {
        "gpt-3.5-turbo": 16384,
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
    }
    return known.get(model, 8192)


def extract_text(path: Path) -> str:
    """Return plain text from *path*.

    Supports PDF via ``pdfplumber`` or reads the file as UTF-8 text otherwise.
    """
    if path.suffix.lower() == ".pdf":
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)

    return path.read_text(encoding="utf-8")


def call_with_backoff(
    messages: List[dict],
    model: str,
    provider: str,
    base_url: str | None,
    max_attempts: int = 6,
) -> str:
    """Send *messages* to the chosen provider with retries."""

    delay = 1
    for attempt in range(max_attempts):
        log(f"🤖 LLM call attempt {attempt + 1}/{max_attempts}")
        try:
            if provider == "openai":
                if openai is None:
                    raise ImportError("openai package not installed")
                kwargs = {"model": model, "messages": messages}
                if base_url:
                    # support both openai>=1 and legacy clients
                    if hasattr(openai, "OpenAI"):
                        client = openai.OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY"))
                        res = client.chat.completions.create(**kwargs)
                        return res.choices[0].message.content.strip()
                    else:
                        openai.api_base = base_url
                        res = openai.ChatCompletion.create(**kwargs)
                        return res["choices"][0]["message"]["content"].strip()
                res = openai.ChatCompletion.create(**kwargs)
                return res["choices"][0]["message"]["content"].strip()
            else:  # ollama
                url = base_url or "http://localhost:11434/api/chat"
                url = url.rstrip("/")
                if not url.endswith(("api/chat", "api/generate")):
                    url += "/api/chat"

                payload = {"model": model}
                if url.endswith("api/chat"):
                    payload["messages"] = messages
                    payload["stream"] = False
                else:  # api/generate expects a single prompt string
                    prompt = "\n".join(m["content"] for m in messages)
                    payload.update({"prompt": prompt, "stream": False})

                resp = requests.post(url, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                # openai-compatible or ollama native
                if "message" in data:
                    return data["message"]["content"].strip()
                if "choices" in data:
                    return data["choices"][0]["message"]["content"].strip()
                if "response" in data:
                    return data["response"].strip()
                if "error" in data:
                    raise RuntimeError(str(data["error"]))
                raise KeyError("Unexpected response from backend")
        except Exception:
            if attempt == max_attempts - 1:
                raise
            time.sleep(delay + random.random())
            delay *= 2
    raise RuntimeError("Exceeded retry attempts")


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


def label_text(text: str, model: str, ctx_limit: int, provider: str, base_url: str | None) -> str:
    text = _ensure_length(text, ctx_limit)
    snippet = text.replace("\n", " ")[:60]
    log(f"✏️  Labeling text: {snippet}...")
    prompt = (
        "Give a concise node label (<= 12 words) describing the following text:"\
        f"\n\n{text}"
    )
    messages = [{"role": "user", "content": prompt}]
    label = call_with_backoff(messages, model, provider, base_url)
    log(f"🏷️  Labeled as: {label}")
    return label


def sentence_topic_same(topic: str, sentence: str, model: str, ctx_limit: int, provider: str, base_url: str | None) -> bool:
    prompt = (
        f"Topic: {topic}\nSentence: {sentence}\n" "Does the sentence elaborate on this topic? Answer yes or no."
    )
    prompt = _ensure_length(prompt, ctx_limit)
    log(f"🤔 Checking if sentence relates to '{topic}'")
    reply = call_with_backoff([{"role": "user", "content": prompt}], model, provider, base_url)
    result = reply.lower().startswith("yes")
    log(f"✅ Relation result: {result}")
    return result


def phase2(
    text: str,
    parent: Node,
    offset: int,
    model: str,
    ctx_limit: int,
    provider: str,
    base_url: str | None,
) -> None:
    spans = list(TOKENIZER.span_tokenize(text))
    if not spans:
        return
    first_start, first_end = spans[0]
    first_sentence = text[first_start:first_end]
    label = label_text(first_sentence, model, ctx_limit, provider, base_url)
    end = first_end
    for i in range(1, len(spans)):
        sent = text[spans[i][0] : spans[i][1]]
        if sentence_topic_same(label, sent, model, ctx_limit, provider, base_url):
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
            phase2(text[end:], parent, offset + end, model, ctx_limit, provider, base_url)
            return
    node = Node(
        name=label,
        char_start=offset + first_start,
        char_end=offset + spans[-1][1] - 1,
        parent=parent.id,
    )
    log(f"🪧 New node: {label} [{node.char_start}-{node.char_end}]")
    parent.children.append(node)


def build_tree(text: str, model: str, provider: str, base_url: str | None) -> Node:
    ctx = get_context_window(model)
    log("🌳 Building topic tree")
    root_name = label_text(text, model, ctx // 2, provider, base_url)
    log(f"🌲 Root topic: {root_name}")
    root = Node(name=root_name, char_start=0, char_end=len(text) - 1, parent=None)
    phase2(text, root, 0, model, ctx // 2, provider, base_url)
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
    p.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default=os.environ.get("LLM_PROVIDER", "ollama"),
        help="Which backend to use for completions",
    )
    p.add_argument(
        "--api-base",
        default=os.environ.get("OLLAMA_HOST")
        or os.environ.get("OLLAMA_HOST_PC"),
        help="Override API base URL (for OpenAI-compatible endpoints)",
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
    if args.provider == "openai" and openai is not None:
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        if args.api_base:
            if hasattr(openai, "OpenAI"):
                openai_client = openai.OpenAI(base_url=args.api_base, api_key=os.getenv("OPENAI_API_KEY"))
                openai_client  # quiet linter
            else:
                openai.api_base = args.api_base
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    log(f"📄 Reading {args.input}")
    text = extract_text(Path(args.input))
    tree = build_tree(text, args.model, args.provider, args.api_base)
    log(f"💾 Writing tree to {args.out}")
    Path(args.out).write_text(json.dumps(tree.to_dict(), indent=2), encoding="utf-8")
    push_to_neo4j(tree, args.neo4j_uri, args.neo4j_user, args.neo4j_pass)


if __name__ == "__main__":
    main()
