import os
import json
from pathlib import Path

from langchain_ollama.llms import OllamaLLM

from pipeline import (
    prepare_input_file,
    process_document,
    FINAL_KG_PATH,
    OUT_PATH,
    INPUT_PATH
)
from run_pipeline import load_and_push, clear_database

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
    # TODO
    return None


def push_to_neo4j() -> None:
    if RESET_DB:
        clear_database(drop_meta=True)
    load_and_push(save_to=OUT_PATH)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    #phase1_sentence_kg(INPUT_PATH)
    phase2_summary(INPUT_PATH)
    push_to_neo4j()

    print(f"KG saved to {FINAL_KG_PATH}")


if __name__ == "__main__":
    main()
