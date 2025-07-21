import os
import json
from pathlib import Path

from langchain_ollama.llms import OllamaLLM

from transformation.pipeline import (
    prepare_input_file,
    process_document,
    FINAL_KG_PATH,
    OUT_PATH,
    INPUT_PATH
)
from transformation.summary_logic import (
    LLMWrapper,
    GLOBAL_LLM,
    build_tree,
    OUTPUT_JSON_PATH,
)
from transformation.merge_summary import merge_summary_into_kg
from transformation.run_pipeline import load_and_push, clear_database

# Set to False if you want to keep existing Neo4j data
RESET_DB = True


def ensure_ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_HOST_PC")
    if not host:
        raise EnvironmentError("Set OLLAMA_HOST or OLLAMA_HOST_PC")
    return host


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
    """Run phase 2: build summary tree and merge into KG."""
    llm = LLMWrapper(backend="ollama", model_name="deepseek-r1:14b")
    # register globally for helper functions in summary_logic
    globals()['GLOBAL_LLM'] = llm

    full_text = text_path.read_text(encoding="utf-8")
    root = build_tree(full_text)
    tree = root.to_dict()
    OUTPUT_JSON_PATH.write_text(json.dumps(tree, indent=2), encoding="utf-8")

    merge_summary_into_kg(OUTPUT_JSON_PATH, FINAL_KG_PATH, FINAL_KG_PATH)


def push_to_neo4j() -> None:
    if RESET_DB:
        clear_database(drop_meta=True)
    load_and_push(save_to=OUT_PATH)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    phase1_sentence_kg(INPUT_PATH)
    phase2_summary(INPUT_PATH)
    push_to_neo4j()

    print(f"KG saved to {FINAL_KG_PATH}")
    print(f"Summary tree saved to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
