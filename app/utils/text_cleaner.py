import re


def clean_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(\w)-\n(\w)", r"\1\2", normalized)
    normalized = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", normalized)
    normalized = re.sub(r"https?://\S+", "", normalized)
    normalized = re.sub(r"\*\*([^*]+)\*\*", r"\1", normalized)
    normalized = re.sub(r"\\\[(\d+)\]", r"[\1]", normalized)
    normalized = re.sub(r"[（(]\s*\d+\s*[)）]", "", normalized)
    normalized = re.sub(r"(?m)^\s*(?:第?\s*\d+\s*页|page\s*\d+|\d+)\s*$", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[ \t]+", " ", normalized)

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", normalized) if part.strip()]
    cleaned_paragraphs: list[str] = []
    for paragraph in paragraphs:
        cleaned = re.sub(r"\s+", " ", paragraph).strip()
        if not cleaned:
            continue
        if _is_reference_paragraph(cleaned):
            continue
        cleaned_paragraphs.append(cleaned)

    normalized = "\n\n".join(cleaned_paragraphs)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"\(\s*\)", "", normalized)
    return normalized.strip()


def _is_reference_paragraph(text: str) -> bool:
    compact = text.strip()
    if compact.startswith("## Page "):
        return False
    if re.match(r"^\[?\d+\]?\s+", compact):
        return True
    if re.match(r"^\\?\[\d+\]\s+", compact):
        return True
    if "http" in compact and len(compact) < 180:
        return True
    if compact.startswith("参考资料") or compact.startswith("参考文献"):
        return True
    citation_markers = re.findall(r"\[\d+\]", compact)
    if len(citation_markers) >= 2 and len(compact) < 240:
        return True
    return False
