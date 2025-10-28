import re


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Replace common encoding artifacts
    s = (
        s.replace("\x92", "'")
        .replace("", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace("…", "...")
        .replace("\x96", "-")
        .replace("\x97", "-")
        .replace("\xa0", " ")
    )
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s
