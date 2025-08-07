from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
import re


# ---------------------------------------------------------------------------
# Configuration paths -------------------------------------------------------
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = BASE_DIR / "structured"
CYTHER_PATH = STRUCTURED_DIR / "import_kg.cypher"


@dataclass
class Topic:
    """Simple representation of a Topic extracted from ``import_kg.cypher``."""

    id: str
    label: str
    embedding: List[float] | None = None


# ---------------------------------------------------------------------------
# Step 1: extract Topic objects ---------------------------------------------
# ---------------------------------------------------------------------------

def extract_topics(path: Path = CYTHER_PATH) -> List[Topic]:
    """Parse *path* and return ``Topic`` objects present in the Cypher file."""
    topics: List[Topic] = []
    if not path.exists():
        return topics

    pattern = re.compile(
        r'CREATE\s+\(:Entity\s+\{id: "(?P<id>[^"]+)",\s+label: "(?P<label>[^"]+)"[^}]*type: "Topic"'
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            topics.append(Topic(id=match.group("id"), label=match.group("label")))
    return topics


# ---------------------------------------------------------------------------
# Step 2: create embeddings -------------------------------------------------
# ---------------------------------------------------------------------------

def _build_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def _embed(texts: Iterable[str], tokenizer, model) -> List[List[float]]:
    import torch

    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # mean pool the token embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    # normalise to unit vectors
    embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
    return embeddings.cpu().tolist()


def embed_topics(topics: List[Topic]) -> None:
    tokenizer, model = _build_embedder()
    vectors = _embed([t.label for t in topics], tokenizer, model)
    for topic, vec in zip(topics, vectors):
        topic.embedding = vec


# ---------------------------------------------------------------------------
# Step 3: query Neo4j vector index -----------------------------------------
# ---------------------------------------------------------------------------

def query_similar_topics(
    topics: List[Topic],
    *,
    index_name: str = "topic-embeddings",
    top_k: int = 5,
    uri: str = "bolt://localhost:7687",
    auth: Tuple[str, str] = ("neo4j", "12345678"),
) -> List[Tuple[Topic, List[dict]]]:
    """Return a list of ``(topic, matches)`` pairs."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=auth)
    results: List[Tuple[Topic, List[dict]]] = []
    with driver.session() as session:
        # Ensure the vector index exists.  Determine the dimensionality from
        # the first available topic embedding and create the index on demand.
        dim = next((len(t.embedding) for t in topics if t.embedding), None)
        if dim:
            exists = session.run(
                "SHOW INDEXES YIELD name WHERE name = $name RETURN name",
                name=index_name,
            ).single()
            if not exists:
                session.run(
                    "CALL db.index.vector.createNodeIndex("
                    "$name, 'Topic', 'embedding', $dim, 'cosine')",
                    name=index_name,
                    dim=dim,
                )
        for topic in topics:
            if topic.embedding is None:
                continue
            res = session.run(
                "CALL db.index.vector.queryNodes($index, $k, $embedding) "
                "YIELD node, score RETURN node.id AS id, node.label AS label, score",
                index=index_name,
                k=top_k,
                embedding=topic.embedding,
            )
            matches = [r.data() for r in res]
            results.append((topic, matches))
    driver.close()
    return results


# ---------------------------------------------------------------------------
# Step 4: present results ---------------------------------------------------
# ---------------------------------------------------------------------------

def show_results(results: List[Tuple[Topic, List[dict]]]) -> None:
    for topic, matches in results:
        print(f"Topic from KG: {topic.id} – {topic.label}")
        for hit in matches:
            score = hit.get("score")
            print(
                f"  similar: {hit.get('id')} – {hit.get('label')} (score={score:.4f})"
            )
        if not matches:
            print("  no similar topics found")


def main() -> None:
    topics = extract_topics()
    if not topics:
        print("No Topic nodes found in import_kg.cypher")
        return
    embed_topics(topics)
    results = query_similar_topics(topics)
    show_results(results)


if __name__ == "__main__":
    main()
