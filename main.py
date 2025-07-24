#!/usr/bin/env python
"""Entry point to run ingestion and transformation pipeline."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from ingestion.ingestion import ingest_directory

STRUCTURED_DIR = Path("structured")
INPUT_FILE = STRUCTURED_DIR / "input.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ClearSure pipeline")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Directory with PDF files to ingest",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    ingest_directory(args.input_dir, INPUT_FILE, verbose=args.verbose)

    cmd = [sys.executable, str(Path("transformation/main.py")), "--input", str(INPUT_FILE)]
    if args.verbose:
        cmd.append("--verbose")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
