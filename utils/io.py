from pathlib import Path
import pdfplumber


def extract_text(path: Path) -> str:
    """Return plain text extracted from *path*.

    Supports PDF via pdfplumber; other files are read as UTF-8.
    """
    if path.suffix.lower() == ".pdf":
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)

    return path.read_text(encoding="utf-8")
