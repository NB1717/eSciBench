import re
import unicodedata

def normalize_string(text):
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    # Remove LaTeX \text{...}
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)

    # x_{q_1} -> x_q_1
    text = re.sub(r'([A-Za-z0-9])_\{([^{}]+)\}', r'\1_\2', text)

    # \cdots -> ⋯
    text = text.replace(r"\cdots", "⋯")

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()
