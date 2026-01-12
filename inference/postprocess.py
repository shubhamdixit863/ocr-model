import unicodedata

KAITHI_START = 0x11080
KAITHI_END = 0x110CF


def postprocess_text(text: str) -> str:
    """Normalize and filter text to Kaithi block and whitespace."""
    normalized = unicodedata.normalize("NFC", text)
    cleaned = []
    for ch in normalized:
        code = ord(ch)
        if ch.isspace() or KAITHI_START <= code <= KAITHI_END:
            cleaned.append(ch)
    return "".join(cleaned).strip()
