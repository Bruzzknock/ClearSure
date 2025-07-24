import os
from pathlib import Path
from langchain_ollama.llms import OllamaLLM

from summary_process import process_document
from run_pipeline import load_and_push, clear_database

STRUCTURED_DIR = Path(__file__).resolve().parents[1] / "structured"
OUT_PATH = STRUCTURED_DIR / "import_kg.cypher"


def main() -> None:
    os.environ["OLLAMA_HOST"] = os.environ.get("OLLAMA_HOST_PC", os.environ.get("OLLAMA_HOST", ""))
    model_name = os.environ.get("OLLAMA_MODEL", "deepseek-r1:14b")
    model = OllamaLLM(
        model=model_name,
        base_url=os.environ["OLLAMA_HOST"],
        options={"num_ctx": 8192},
        temperature=0.0,
    )

    final_kg = process_document(model)
    print(f"✅ Done. final KG has {len(final_kg['edges'])} edges.")
    clear_database(drop_meta=True)
    load_and_push(save_to=OUT_PATH)
    print("✅ Graph ingested and written to", OUT_PATH)


if __name__ == "__main__":
    main()
