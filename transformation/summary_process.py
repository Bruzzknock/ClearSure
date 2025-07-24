import json
from pathlib import Path
from typing import Iterable

from sentence_process import process_sentence
from kg_utils import (
    split_into_sentences,
    ensure_final_kg_exists,
    reset_final_kg,
    load_file,
)


STRUCTURED_DIR = Path(__file__).resolve().parents[1] / "structured"
FINAL_KG_PATH = STRUCTURED_DIR / "final_kg.json"


def process_document(model, input_file: str = "output.json", kg_path: Path = FINAL_KG_PATH) -> dict:
    """Run the sentence pipeline over a whole document."""
    ensure_final_kg_exists(kg_path)
    reset_final_kg(kg_path, backup=False)
    raw_text = load_file(STRUCTURED_DIR / input_file)
    sentences: Iterable[str] = list(split_into_sentences(raw_text))
    for sentence in sentences:
        process_sentence(sentence, model, kg_path)
    return json.loads(kg_path.read_text())
