from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader


def parse_file(file_path: str, file_type: str) -> tuple[str, dict]:
    path = Path(file_path)
    parser = {
        "txt": _parse_plain_text,
        "md": _parse_plain_text,
        "pdf": _parse_pdf,
        "docx": _parse_docx,
    }.get(file_type)

    if parser is None:
        raise ValueError(f"unsupported file type: {file_type}")

    return parser(path)


def _parse_plain_text(path: Path) -> tuple[str, dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text, {}


def _parse_pdf(path: Path) -> tuple[str, dict]:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages), {"page_count": len(reader.pages)}


def _parse_docx(path: Path) -> tuple[str, dict]:
    doc = DocxDocument(str(path))
    paragraphs = [paragraph.text for paragraph in doc.paragraphs]
    return "\n".join(paragraphs), {"paragraph_count": len(doc.paragraphs)}
