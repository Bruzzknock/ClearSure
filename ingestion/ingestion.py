#!/usr/bin/env python
"""
ingestion.py — minimal entry‑point to process a document with the
`unstructured` library.

It loads a PDF (or any of the 60+ formats unstructured recognises),
optionally chunks the parsed Elements, and writes the result to JSON so the
next step in the ClearSure pipeline (fact extraction → graph loading) can
consume it.

Usage (inside the activated virtual‑env):

    python ingestion.py docs/sample.pdf --out structured/sample.json --chunk

The script defaults to the high‑resolution parsing strategy (Detectron2 +
Tesseract) but you can switch to the faster heuristics with
    --strategy fast

Add -h / --help to see all options.
"""

from __future__ import annotations
import logging
import time
import argparse
import json
from pathlib import Path
from typing import List

from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import Element

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Partition a document with unstructured and emit JSON", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=Path, help="Path to the document to ingest")
    parser.add_argument("--out", type=Path, default=None, help="Where to write JSON output (directory will be created if missing)")
    parser.add_argument("--strategy", choices=["hi_res", "auto","fast"], default="hi_res", help="Partition strategy. hi-res gives best layout fidelity; fast is CPU-light")
    parser.add_argument("--chunk", action="store_true", help="Chunk consecutive Elements for downstream embedding / graph nodes")
    parser.add_argument("--max-chars", type=int, default=800, help="Maximum characters per chunk (only if --chunk is set)")
    return parser.parse_args()

def partition_document(path: Path, strategy: str) -> List[Element]:
    """
    Parse *path* with Unstructured using *strategy* and return its Elements.

    Parameters
    ----------
    path : Path
        The document to ingest (PDF, DOCX, PPTX, image, HTML, …).
    strategy : {"hi_res", "auto", "fast"}
        Controls how much layout analysis and OCR Unstructured performs.

    Returns
    -------
    List[Element]
        Ordered list of Element objects representing structural blocks.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *strategy* is not one of the accepted options.
    Any exception raised by `partition()` is propagated unchanged.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    
    if strategy not in {"hi_res", "auto", "fast"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    logger.debug("Partitioning %s using %s strategy", path, strategy)
    t0 = time.perf_counter()

    elements: List[Element] = partition(filename=str(path), strategy=strategy)

    logger.info(
        "Partitioned %s – %d elements in %.2f s",
        path, len(elements), time.perf_counter() - t0,
    )
    return elements

def maybe_chunk(elems: List[Element], do_chunk: bool, max_chars: int) -> List[Element]:
    """Return elems unchanged unless the user passed --chunk."""
    if not do_chunk:
        return elems
    return chunk_elements(elems, max_characters=max_chars)

def write_output(elements: List[Element], out: Path | None, src: Path) -> None:
    """Either pretty-print a preview or dump full JSON to *out*."""
    if out is None:
        # console preview = first few elements
        for el in elements[:8]:
            kind    = getattr(el, "category", el.__class__.__name__)  # future-proof
            snippet = (el.text or "").replace("\n", " ")[:80]
            print(f"{kind:<15} | {snippet!r}")
        print(f"... total: {len(elements)} elements")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    records = [el.to_dict() for el in elements]
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d elements ➜ %s", len(records), out)

def main() -> None:
    args = _parse_args()

    # 1) parse
    elements = partition_document(args.input, args.strategy)

    # 2) optional chunk
    elements = maybe_chunk(elements, args.chunk, args.max_chars)

    # 3) output
    write_output(elements, args.out, args.input)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()