import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    section_title: str | None = None


def split_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    return [item.text for item in split_text_with_metadata(text, chunk_size=chunk_size, overlap=overlap)]


def split_text_with_metadata(text: str, chunk_size: int = 600, overlap: int = 80) -> list[TextChunk]:
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if not paragraphs:
        return []

    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_length = 0
    current_section_title: str | None = None
    pending_heading: str | None = None

    for paragraph in paragraphs:
        if _is_heading(paragraph):
            if current_parts:
                chunk = _merge_parts(current_parts)
                if _is_meaningful_chunk(chunk):
                    chunks.append(TextChunk(text=chunk, section_title=current_section_title))
                current_parts = []
                current_length = 0
            pending_heading = paragraph
            current_section_title = _normalize_heading(paragraph)
            continue

        if pending_heading:
            paragraph = f"{pending_heading}\n\n{paragraph}"
            pending_heading = None

        if len(paragraph) > chunk_size:
            if current_parts:
                chunk = _merge_parts(current_parts)
                if _is_meaningful_chunk(chunk):
                    chunks.append(TextChunk(text=chunk, section_title=current_section_title))
                current_parts = []
                current_length = 0

            prefix = None
            if current_section_title and not paragraph.startswith("#"):
                heading_line = _format_heading_prefix(current_section_title)
                if paragraph.startswith(heading_line):
                    prefix = None
                else:
                    prefix = heading_line
            for piece in _split_long_paragraph(paragraph, chunk_size, overlap, prefix=prefix):
                if _is_meaningful_chunk(piece):
                    chunks.append(TextChunk(text=piece, section_title=current_section_title))
            continue

        projected = current_length + len(paragraph) + (2 if current_parts else 0)
        if projected <= chunk_size:
            current_parts.append(paragraph)
            current_length = projected
            continue

        chunk = _merge_parts(current_parts)
        if _is_meaningful_chunk(chunk):
            chunks.append(TextChunk(text=chunk, section_title=current_section_title))

        current_parts = [paragraph]
        current_length = len(paragraph)

    if current_parts:
        chunk = _merge_parts(current_parts)
        if _is_meaningful_chunk(chunk):
            chunks.append(TextChunk(text=chunk, section_title=current_section_title))

    return chunks


def _split_long_paragraph(paragraph: str, chunk_size: int, overlap: int, prefix: str | None = None) -> list[str]:
    pieces: list[str] = []
    start = 0

    while start < len(paragraph):
        end = min(start + chunk_size, len(paragraph))
        if end < len(paragraph):
            split_at = max(
                paragraph.rfind("。", start, end),
                paragraph.rfind("！", start, end),
                paragraph.rfind("？", start, end),
                paragraph.rfind("\n", start, end),
                paragraph.rfind(" ", start, end),
            )
            if split_at > start + int(chunk_size * 0.6):
                end = split_at + 1

        piece = paragraph[start:end].strip()
        if prefix and not pieces and piece and not piece.startswith("#"):
            piece = f"{prefix}\n\n{piece}"
        if piece:
            pieces.append(piece)
        if end >= len(paragraph):
            break
        start = max(start + 1, end - overlap)

    return pieces


def _merge_parts(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part.strip()).strip()


def _is_heading(text: str) -> bool:
    return bool(re.match(r"^#{1,6}\s+", text.strip()))


def _normalize_heading(text: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", text.strip()).strip()


def _format_heading_prefix(section_title: str) -> str:
    return f"### {section_title}"


def _is_meaningful_chunk(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 40:
        return False

    cjk_count = sum(1 for char in stripped if "\u4e00" <= char <= "\u9fff")
    alpha_num_count = sum(1 for char in stripped if char.isalnum())
    slash_count = stripped.count("/") + stripped.count("\\")
    if cjk_count + alpha_num_count < 20:
        return False
    if slash_count > max(20, len(stripped) // 8):
        return False
    return True
