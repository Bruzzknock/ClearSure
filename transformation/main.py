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
)
from run_pipeline import load_and_push, clear_database
import doc_tree
from kg_utils import update_kg, clean_kg

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

    reset_final_kg(FINAL_KG_PATH, backup=False, verbose=False)

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
    p.add_argument("--input", type=Path, required=True, help="Document to process")
    p.add_argument("--no-reset-db", action="store_true", help="Keep existing Neo4j data")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global RESET_DB
    RESET_DB = not args.no_reset_db

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    phase1_sentence_kg(args.input)
    phase2_summary(args.input)
    push_to_neo4j()

    print(f"KG saved to {FINAL_KG_PATH}")


if __name__ == "__main__":
    main()
