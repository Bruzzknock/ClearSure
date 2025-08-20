import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer


def clean_relation(s: str) -> str:
    """Normalize relation and label names for Cypher.

    Neo4j relationship types and labels may only contain alphanumeric
    characters and underscores. Any other character (including punctuation
    like curly apostrophes) is replaced with an underscore and the result is
    uppercased.
    """

    return re.sub(r"[^0-9A-Z_]", "_", s.upper())


def escape(s: str) -> str:
    """Escape quotes inside labels/properties."""

    return s.replace('"', '\\"')


BASE_DIR = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = BASE_DIR / "structured"
DEFAULT_INPUT = STRUCTURED_DIR / "final_kg.json"
DEFAULT_OUTPUT = STRUCTURED_DIR / "import_kg.cypher"

# Initialize the sentence embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def final_to_cypher(input: Path = DEFAULT_INPUT, output: Path = DEFAULT_OUTPUT) -> None:
    """Convert ``final_kg.json`` to ``import_kg.cypher`` with embeddings."""

    with open(input, "r", encoding="utf-8") as f:
        kg = json.load(f)

    cypher_nodes: list[str] = []
    cypher_edges: list[str] = []

    # Use a set to track all created nodes (avoid duplicate CREATEs)
    node_ids: set[str] = set()

    for node in kg["nodes"]:
        nid = node["id"]
        # Not all nodes are guaranteed to carry a `label` field.  If it's
        # missing we fall back to an empty string so the conversion still
        # succeeds rather than failing with a KeyError.
        label = escape(node.get("label", ""))

        # Avoid duplicating the same node
        if nid in node_ids:
            continue
        node_ids.add(nid)

        props = {k: v for k, v in node.items() if k not in ["id", "label", "type"]}
        # Also handle nested attributes
        if "attributes" in props:
            attrs = props.pop("attributes")
            props.update(attrs)

        node_label = node.get("type", "Entity")

        # Add embedding for Topic nodes
        if node_label == "Topic":
            embedding = model.encode(node.get("label", "")).tolist()
            props["embedding"] = embedding

        prop_str = ""
        if props:
            prop_pairs = [
                f'{k}: "{v}"' if isinstance(v, str) else f'{k}: {v}'
                for k, v in props.items()
            ]
            prop_str = ", " + ", ".join(prop_pairs)

        cypher_nodes.append(
            f'CREATE (:{clean_relation(node_label)} {{id: "{nid}", label: "{label}"{prop_str}}});'
        )

    for edge in kg["edges"]:
        src = edge["source"]
        tgt = edge["target"]
        rel = clean_relation(escape(edge["relation"]))
        attr_str = ""
        if "attributes" in edge and edge["attributes"]:
            attr_pairs = [
                f'{k}: "{v}"' if isinstance(v, str) else f'{k}: {v}'
                for k, v in edge["attributes"].items()
            ]
            attr_str = " { " + ", ".join(attr_pairs) + " }"

        cypher_edges.append(
            f"""
MATCH (a {{id: \"{src}\"}}), (b {{id: \"{tgt}\"}})
CREATE (a)-[:{rel}{attr_str}]->(b);
"""
        )

    # Write Cypher script to file
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(cypher_nodes + cypher_edges))


if __name__ == "__main__":
    final_to_cypher()
