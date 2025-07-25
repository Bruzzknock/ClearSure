# ------------------------------------------------------------------
# 0.  utilities ----------------------------------------------------
# ------------------------------------------------------------------
import json, os
from pathlib import Path
from typing import Dict, Any, Iterable
import spacy, warnings
import argparse
import pdfplumber
from langchain_ollama.llms import OllamaLLM
from kg_utils import _extract_json_block, update_kg, clean_kg
from LLMs import (
    simplify_text,
    remove_think_block,
    create_knowledge_ontology,
    clean_up_1st_phase,
    label_text,
    clean_label,
)

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = BASE_DIR / "structured"
FINAL_KG_PATH = STRUCTURED_DIR / "final_kg.json"
INPUT_PATH = STRUCTURED_DIR / "input.txt"
OUT_PATH = STRUCTURED_DIR / "import_kg.cypher"
SENTENCE_KGS_PATH = STRUCTURED_DIR / "sentence_kgs.json"


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


def extract_text(path: Path) -> str:
    """Return plain text extracted from *path*.

    Currently supports PDF via pdfplumber; other files are read as UTF-8.
    """
    if path.suffix.lower() == ".pdf":
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)

    return path.read_text(encoding="utf-8")


def prepare_input_file(src: Path, dest: Path = STRUCTURED_DIR / "output.json") -> str:
    """Extract text from *src* and write it to *dest*.

    Returns the basename of the written file for `process_document`.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = extract_text(src)
    dest.write_text(text, encoding="utf-8")
    return dest.name


# ------------------------------------------------------------------
# 1.  sentence splitter --------------------------------------------
# ------------------------------------------------------------------
# use spaCy's small English model for sentence boundary detection
try:
    # keep the parser so abbreviations like "Dr." don't trigger splits
    _nlp = spacy.load("en_core_web_sm", disable=["tagger", "ner"])
except OSError:
    # model missing → fall back to a blank pipeline + simple sentencizer
    warnings.warn("en_core_web_sm not found; using blank 'en' + sentencizer")
    _nlp = spacy.blank("en")

# ensure there is a component that sets `is_sent_start`
if "parser" not in _nlp.pipe_names and "senter" not in _nlp.pipe_names:
    _nlp.add_pipe("sentencizer")


def split_into_sentences(text: str) -> Iterable[tuple[str, int, int]]:
    """Yield ``(sentence, start, end)`` triples for *text*.

    ``start`` and ``end`` are character offsets (inclusive) referring to the
    original text.  Leading/trailing whitespace is stripped from the returned
    sentence and offsets adjusted accordingly.
    """

    doc = _nlp(text)
    for sent in doc.sents:
        raw = sent.text
        if not raw.strip():
            continue

        start = sent.start_char
        end = sent.end_char - 1  # spaCy end_char is exclusive

        lstrip = len(raw) - len(raw.lstrip())
        rstrip = len(raw) - len(raw.rstrip())

        start += lstrip
        end -= rstrip
        yield raw.strip(), start, end


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


def _max_topic_index(nodes: list[dict]) -> int:
    idx = [int(n["id"][1:]) for n in nodes if n.get("id", "").startswith("t")]
    return max(idx, default=0)


def build_topic_tree(kg: dict, model) -> dict:
    """Return topic nodes and edges linking statements to them."""
    statements = [
        (n["id"], n.get("label", ""))
        for n in kg.get("nodes", [])
        if n.get("type") == "Statement"
    ]
    if not statements:
        return {"nodes": [], "edges": []}

    groups = [statements[i : i + 5] for i in range(0, len(statements), 5)]
    next_idx = _max_topic_index(kg.get("nodes", []))
    nodes: list[dict] = []
    edges: list[dict] = []

    for grp in groups:
        text = " ".join(lbl for _, lbl in grp)
        raw = label_text(text, model)
        label = clean_label(raw)
        next_idx += 1
        tid = f"t{next_idx}"
        nodes.append({"id": tid, "label": label, "type": "Topic"})
        root_sid = grp[0][0]
        edges.append({"source": root_sid, "relation": "BELONGS_TO_TOPIC", "target": tid})

    return {"nodes": nodes, "edges": edges}


# ------------------------------------------------------------------
# 2.  core loop ----------------------------------------------------
# ------------------------------------------------------------------
def process_document(model, input_file: str = "output.json") -> list[dict]:
    """Extract a KG for each sentence and store all of them in ``sentence_kgs.json``."""

    raw_text = load_text(input_file)
    sentences = list(split_into_sentences(raw_text))
    print(f"🟢 {len(sentences)} sentences queued\n")

    SENTENCE_KGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    sentence_kgs: list[dict] = []
    SENTENCE_KGS_PATH.write_text("[]", encoding="utf-8")
    ensure_final_kg_exists()

    for idx, (sentence, start_pos, end_pos) in enumerate(sentences, start=1):
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

        # ------------------------------------------------------
        # (C) clean-up first pass
        # ------------------------------------------------------
        kg_patch_dict = json.loads(_extract_json_block(kg_patch_txt))
        cleaned_patch_txt = remove_think_block(clean_up_1st_phase(kg_patch_dict, model))
        print("✅✅✅✅✅✅ Cleaned Edges:", cleaned_patch_txt)

        edges_patch = json.loads(_extract_json_block(cleaned_patch_txt)).get("edges_patch", [])

        sentence_kg = {
            "sentence": sentence,
            "char_start": start_pos,
            "char_end": end_pos,
            "kg": {
                "nodes": kg_patch_dict.get("nodes", []),
                "edges": kg_patch_dict.get("edges", []) + edges_patch,
            },
        }

        sentence_kgs.append(sentence_kg)
        SENTENCE_KGS_PATH.write_text(
            json.dumps(sentence_kgs, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        kg_patch = sentence_kg["kg"]
        update_kg(kg_patch, kg_path=FINAL_KG_PATH)

    # Build topics after all sentences are processed
    current_kg = json.loads(FINAL_KG_PATH.read_text(encoding="utf-8"))
    topic_patch = build_topic_tree(current_kg, model)
    if topic_patch["nodes"] or topic_patch["edges"]:
        clean_kg({"nodes": topic_patch["nodes"], "edges_patch": topic_patch["edges"]}, kg_path=FINAL_KG_PATH)

    return sentence_kgs


# ------------------------------------------------------------------
# 3.  run it --------------------------------------------------------
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a document into a KG and push to Neo4j"
    )
    parser.add_argument("path", type=Path, help="Path to a PDF or text file to process")
    args = parser.parse_args()

    os.environ["OLLAMA_HOST"] = os.environ.get(
        "OLLAMA_HOST_PC", os.environ.get("OLLAMA_HOST", "")
    )

    model = OllamaLLM(
        model="deepseek-r1:14b",
        base_url=os.environ["OLLAMA_HOST"],
        options={"num_ctx": 8192},
        temperature=0.0,
    )

    input_file = prepare_input_file(args.path)

    results = process_document(model, input_file=input_file)
    save(results, SENTENCE_KGS_PATH.name)
    print(f"\n✅ Done. {SENTENCE_KGS_PATH.name} contains {len(results)} KGs.")
