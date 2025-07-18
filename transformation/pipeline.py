# ------------------------------------------------------------------
# 0.  utilities ----------------------------------------------------
# ------------------------------------------------------------------
import json, os
from pathlib import Path
from typing import Dict, Any, Iterable
import spacy, warnings
from langchain_ollama.llms import OllamaLLM
from kg_utils import clean_kg, update_kg       
from LLMs import (
    simplify_text,
    remove_think_block,
    create_knowledge_ontology,
    clean_up_1st_phase,
    one_sentence_summary,
)

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
CURRENT_SENTENCE_KG_PATH = STRUCTURED_DIR / "current_sentence_kg.json"

def save(text: str, file: str) -> str:
    output_file = STRUCTURED_DIR / file
    if isinstance(text, (dict, list)):
        text_str = json.dumps(text, ensure_ascii=False, indent=2)
    else:
        text_str = str(text)
    with output_file.open("w", encoding="utf-8") as f:
        f.write(text_str)
    return text_str

def load_text(file_name: str) -> str:
    return (STRUCTURED_DIR / file_name).read_text(encoding="utf-8")

def ensure_final_kg_exists() -> None:
    """
    Make sure final_kg.json exists and has the minimal structure.
    """
    if not FINAL_KG_PATH.exists():
        FINAL_KG_PATH.write_text(json.dumps({"nodes": [], "edges": []}, indent=2))

# ------------------------------------------------------------------
# 1.  sentence splitter --------------------------------------------
# ------------------------------------------------------------------
# use the tiny model – plenty for sentence boundary detection
try:
    _nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
    _nlp.add_pipe("sentencizer")          # ← sets .is_sent_start flags
except OSError:
    # model missing → fall back to a blank pipeline
    warnings.warn("en_core_web_sm not found; using blank 'en' + sentencizer")
    _nlp = spacy.blank("en")
    _nlp.add_pipe("sentencizer")

def split_into_sentences(text: str) -> Iterable[str]:
    doc = _nlp(text)
    for sent in doc.sents:
        t = sent.text.strip()
        if t:
            yield t

import shutil, time

def reset_final_kg(
    path: Path = FINAL_KG_PATH,
    backup: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Ensure a *clean/empty* KG file at `path`.

    - Creates parent directory if missing.
    - Optionally backs up an existing file (timestamped).
    - Writes {"nodes": [], "edges": []}.
    - Returns the empty dict written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and backup:
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup_path = path.with_suffix(path.suffix + f".bak.{ts}")
        shutil.copy2(path, backup_path)
        if verbose:
            print(f"📦 Backed up existing KG to {backup_path}")

    empty = {"nodes": [], "edges": []}
    path.write_text(json.dumps(empty, indent=2), encoding="utf-8")
    if verbose:
        print(f"🧹 Reset KG at {path}")
    return empty


# ------------------------------------------------------------------
# 2.  core loop ----------------------------------------------------
# ------------------------------------------------------------------
def process_document(model, input_file: str = "output.json") -> Dict[str, Any]:
    """Run the three-stage pipeline sentence-by-sentence."""
    ensure_final_kg_exists()                     # create empty KG if needed
    reset_final_kg(backup=False)

    raw_text = load_text(input_file)
    sentences = list(split_into_sentences(raw_text))
    print(f"🟢 {len(sentences)} sentences queued\n")

    summary = ""
    for idx, sentence in enumerate(sentences, start=1):
        print(f"—— Sentence {idx}/{len(sentences)} ——")
        print(f"—— Sentence —— {sentence}")

        # ------------------------------------------------------
        # (A) simplify
        # ------------------------------------------------------
        simplified_txt = remove_think_block(simplify_text(sentence, model))
        print("✅✅✅✅✅✅ Simplified text:", simplified_txt)

        # ------------------------------------------------------
        # (B) ontology generation
        # ------------------------------------------------------
        kg_patch_txt = remove_think_block(
            create_knowledge_ontology(simplified_txt, model)
        )
        print("✅✅✅✅✅✅ Ontology:", kg_patch_txt)

        _, id_map = update_kg(
            kg_patch_txt,
            kg_path=FINAL_KG_PATH,
            save=True,
            return_id_map=True,
        )

        # ------------------------------------------------------
        # (C) clean-up first pass
        # ------------------------------------------------------
        kg_patch_dict = json.loads(kg_patch_txt)
        cleaned_patch = remove_think_block(
            clean_up_1st_phase(kg_patch_dict, model)
        )
        print("✅✅✅✅✅✅ Cleaned Edges:", cleaned_patch)

        # ------------------------------------------------------
        # (D) merge into the growing master KG
        # ------------------------------------------------------
        clean_kg(
            cleaned_patch,
            kg_path=FINAL_KG_PATH,
            save=True,
            id_map=id_map,
            reassign_edge_ids=True,
            drop_missing=True,
        )

        current_kg = json.loads(FINAL_KG_PATH.read_text())
        save(current_kg, CURRENT_SENTENCE_KG_PATH.name)

        # ------------------------------------------------------
        # (E) update running summary
        # ------------------------------------------------------
        summary_input = f"{summary} {sentence}".strip() if summary else sentence
        summary = remove_think_block(one_sentence_summary(summary_input, model))
        print("✅✅✅✅✅✅ Summary:", summary)

        # ------------------------------------------------------
        # (F) merge using summary context
        # ------------------------------------------------------
        summary_kg_txt = remove_think_block(create_knowledge_ontology(summary, model))
        update_kg(summary_kg_txt, kg_path=FINAL_KG_PATH, save=True)
        summary_kg_dict = json.loads(summary_kg_txt)
        summary_patch = remove_think_block(
            clean_up_1st_phase(summary_kg_dict, model)
        )
        clean_kg(
            summary_patch,
            kg_path=FINAL_KG_PATH,
            save=True,
            reassign_edge_ids=True,
            drop_missing=True,
        )

    final_kg = json.loads(FINAL_KG_PATH.read_text())
    save(final_kg, CURRENT_SENTENCE_KG_PATH.name)
    return final_kg

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
