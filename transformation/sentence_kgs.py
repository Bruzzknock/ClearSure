import json
from pathlib import Path
import argparse

from llm import build_llm

from LLMs import simplify_text, remove_think_block, create_knowledge_ontology
from kg_utils import _extract_json_block
from pipeline import split_into_sentences, extract_text

BASE_DIR = Path(__file__).resolve().parents[1]
STRUCTURED_DIR = BASE_DIR / "structured"
STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)


def sentence_kgs(text: str, model) -> list[dict]:
    """Return list of {{"sentence": str, "kg": dict}} for *text*."""
    results = []
    for sentence in split_into_sentences(text):
        simplified = remove_think_block(simplify_text(sentence, model))
        kg_text = remove_think_block(create_knowledge_ontology(simplified, model))
        kg_json = json.loads(_extract_json_block(kg_text))
        results.append({"sentence": sentence, "kg": kg_json})
    return results


def main(path: Path, out: Path) -> None:
    text = extract_text(path)
    model = build_llm()
    kgs = sentence_kgs(text, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kgs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(kgs)} sentence KGs to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per-sentence knowledge graphs")
    parser.add_argument("path", type=Path, help="PDF or text file to process")
    parser.add_argument(
        "--out",
        type=Path,
        default=STRUCTURED_DIR / "sentence_kgs.json",
        help="Where to store the resulting JSON list",
    )
    args = parser.parse_args()
    main(args.path, args.out)
