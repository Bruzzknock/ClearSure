# ------------------------------------------------------------------
# 0.  utilities ----------------------------------------------------
# ------------------------------------------------------------------
import json, os
from pathlib import Path
from typing import Dict, Any
from langchain_ollama.llms import OllamaLLM
from kg_utils import (
    clean_kg,
    update_kg,
    save_file,
    load_file,
    split_into_sentences,
    ensure_final_kg_exists,
    reset_final_kg,
)
from sentence_process import process_sentence

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE_DIR       = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = BASE_DIR / "structured"
FINAL_KG_PATH  = STRUCTURED_DIR / "final_kg.json"
OUT_PATH       = STRUCTURED_DIR / "import_kg.cypher" 

# helper wrappers kept for backwards compatibility
def save(text: str, file: str) -> str:
    return save_file(text, STRUCTURED_DIR / file)

def load_text(file_name: str) -> str:
    return load_file(STRUCTURED_DIR / file_name)

# ------------------------------------------------------------------
# 1.  sentence splitter --------------------------------------------
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 2.  core loop ----------------------------------------------------
# ------------------------------------------------------------------
def process_document(model, input_file: str = "output.json") -> Dict[str, Any]:
    """
    Run the three-stage pipeline *per sentence* and return the final KG dict.
    """
    ensure_final_kg_exists()                     # create empty KG if needed
    reset_final_kg(backup = False)

    raw_text  = load_text(input_file)
    sentences = list(split_into_sentences(raw_text))
    print(f"🟢 {len(sentences)} sentences queued\n")

    for sentence in sentences:
        process_sentence(sentence, model, FINAL_KG_PATH)
        
    # return the final KG object for convenience
    return json.loads(FINAL_KG_PATH.read_text())

# ------------------------------------------------------------------
# 3.  run it --------------------------------------------------------
# ------------------------------------------------------------------
if __name__ == "__main__":
    os.environ["OLLAMA_HOST"] = os.environ["OLLAMA_HOST_PC"]
    model = OllamaLLM(
        model="deepseek-r1:14b",
        base_url=os.environ["OLLAMA_HOST"],
        options={"num_ctx": 8192},
        temperature=0.0,
    )

    final_kg = process_document(model, input_file="output.json")
    save(final_kg,"final_kg.json")
    print("\n✅ Done. final_kg.json now contains", len(final_kg["edges"]), "edges.")

    from run_pipeline import load_and_push, clear_database
    clear_database(drop_meta=True)           # wipe
    load_and_push(save_to=OUT_PATH)          # reload + save copy
    print("FINAL_KG_PATH =", FINAL_KG_PATH.resolve())
    print("OUT_PATH =", OUT_PATH.resolve())
    print("✅ Graph ingested and written")