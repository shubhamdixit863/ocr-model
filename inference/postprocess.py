import unicodedata

DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F


def postprocess_text(text: str) -> str:
    """Normalize and filter text to Devanagari block and whitespace."""
    normalized = unicodedata.normalize("NFC", text)
    cleaned = []
    for ch in normalized:
        code = ord(ch)
        if ch.isspace() or DEVANAGARI_START <= code <= DEVANAGARI_END:
            cleaned.append(ch)
    return "".join(cleaned).strip()
