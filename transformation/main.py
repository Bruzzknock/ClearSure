import os
import json
import argparse
from pathlib import Path

from langchain_ollama.llms import OllamaLLM

from pipeline import (
    prepare_input_file,
    process_document,
    extract_text,
    reset_final_kg,
    FINAL_KG_PATH,
    OUT_PATH,
    STRUCTURED_DIR,
    SENTENCE_KGS_PATH,
    INPUT_PATH,
)
from run_pipeline import load_and_push, clear_database
import doc_tree
from kg_utils import update_kg, clean_kg

VERBOSE = False


def log(msg: str) -> None:
    if VERBOSE:
        print(msg)

# Set to False if you want to keep existing Neo4j data
RESET_DB = True

TOPIC_TREE_PATH = STRUCTURED_DIR / "topic_tree.json"


def ensure_ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_HOST_PC")
    if not host:
        raise EnvironmentError("Set OLLAMA_HOST or OLLAMA_HOST_PC")
    return host


def _find_topic(node: doc_tree.Node, start: int, end: int) -> doc_tree.Node | None:
    """Return the deepest topic ``node`` covering the character span."""
    if start < node.char_start or end > node.char_end:
        return None
    for child in node.children:
        match = _find_topic(child, start, end)
        if match is not None:
            return match
    return node


def phase1_sentence_kg(text_path: Path) -> None:
    """Run phase 1: sentence-level KG extraction."""
    host = ensure_ollama_host()
    model = OllamaLLM(
        model="deepseek-r1:14b",
        base_url=host,
        options={"num_ctx": 8192},
        temperature=0.0,
    )
    input_basename = prepare_input_file(text_path)
    process_document(model, input_file=input_basename)


def phase2_summary(text_path: Path) -> None:
    """Build topic tree and merge sentence KGs into ``final_kg.json``."""

    host = ensure_ollama_host()
    model = OllamaLLM(
        model="deepseek-r1:14b",
        base_url=host,
        options={"num_ctx": 8192},
        temperature=0.0,
    )

    text = extract_text(text_path)

    tree = doc_tree.build_tree(text, model)
    TOPIC_TREE_PATH.write_text(json.dumps(tree.to_dict(), indent=2), encoding="utf-8")

    reset_final_kg(FINAL_KG_PATH, backup=False, verbose=VERBOSE)

    # Insert Topic nodes into the KG before linking statements to them
    topic_nodes, child_edges = doc_tree.flatten_tree(tree)
    if topic_nodes or child_edges:
        update_kg({"nodes": topic_nodes, "edges": child_edges}, kg_path=FINAL_KG_PATH)

    sentence_kgs = json.loads(SENTENCE_KGS_PATH.read_text(encoding="utf-8"))

    for sent in sentence_kgs:
        kg_patch = sent.get("kg", {})
        kg, id_map = update_kg(kg_patch, kg_path=FINAL_KG_PATH, return_id_map=True)

        topic = _find_topic(tree, sent.get("char_start", 0), sent.get("char_end", 0))
        edges = []
        if topic:
            for node in kg_patch.get("nodes", []):
                if node.get("type") == "Statement":
                    new_id = id_map.get(node["id"], node["id"])
                    edges.append({
                        "source": topic.id,
                        "relation": "HAS_STATEMENT",
                        "target": new_id,
                    })
        if edges:
            clean_kg({"edges_patch": edges}, kg_path=FINAL_KG_PATH)

    return None


def push_to_neo4j() -> None:
    if RESET_DB:
        clear_database(drop_meta=True)
    load_and_push(save_to=OUT_PATH)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the ClearSure pipeline")
    p.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Document to process (default: structured/input.txt)",
    )
    p.add_argument("--no-reset-db", action="store_true", help="Keep existing Neo4j data")
    p.add_argument("-v", "--verbose", action="store_true", help="Print progress")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global RESET_DB, VERBOSE
    RESET_DB = not args.no_reset_db
    VERBOSE = args.verbose
    doc_tree.VERBOSE = VERBOSE

    log(f"📄 Using input file: {args.input}")
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    log("🚀 Phase 1: Sentence KG extraction")
    phase1_sentence_kg(args.input)
    log("🚀 Phase 2: Build summary and merge KGs")
    phase2_summary(args.input)
    log("🚀 Pushing KG to Neo4j")
    push_to_neo4j()

    print(f"KG saved to {FINAL_KG_PATH}")


if __name__ == "__main__":
    main()
