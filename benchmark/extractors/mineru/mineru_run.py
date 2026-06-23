import os
import json
from typing import Tuple, List
import re


def looks_like_author_name(text):
    text = re.sub(r"<sup>.*?</sup>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 3:
        return False

    bad_words = [
        "university", "department", "institute", "school",
        "laboratory", "college", "faculty", "email", "@"
    ]

    if any(w in text.lower() for w in bad_words):
        return False

    return bool(re.search(r"[A-Z][a-z]+", text))


AFFILIATION_RE = re.compile(
    r"university|department|institute|school|college|faculty|laboratory|centre|center",
    re.I
)

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)
MINERU_DIR = "/tmp/mineru_101"


# =========================================================
# TEXT EXTRACTION (SAFE)
# =========================================================
def get_text(block):
    parts = []

    for line in block.get("lines", []):
        for span in line.get("spans", []):

            if "content" in span:
                parts.append(span["content"])

            elif "html" in span:
                parts.append(span["html"])

    return " ".join(parts).strip()


# =========================================================
# EQUATION EXTRACTION (ROBUST)
# =========================================================
def extract_equations(block):
    eqs = []

    btype = block.get("type", "")

    # direct equation block
    if btype == "interline_equation":
        eqs.append(get_text(block))

    # span-level equations
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            txt = span.get("content", "")
            if txt and ("=" in txt or "\\" in txt):
                eqs.append(txt)

    # HTML table equations (very important in MinerU)
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if "html" in span:
                html = span["html"]
                if "=" in html:
                    eqs.append(html)

    # recursive blocks
    for sub in block.get("blocks", []):
        eqs.extend(extract_equations(sub))

    return eqs


# =========================================================
# DEDUP CLEANER (IMPORTANT)
# =========================================================
def clean_list(items):
    seen = set()
    out = []

    for x in items:
        x = x.strip()
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)

    return out


def normalize_equation(text):

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"_\s*\{\s*", "_{", text)
    text = re.sub(r"\^\s*\{\s*", "^{", text)

    text = re.sub(r"\s*\}", "}", text)

    text = re.sub(
        r"\\operatorname\s*\{\s*([A-Za-z\s]+)\s*\}",
        lambda m: "\\operatorname{" +
        m.group(1).replace(" ", "") + "}",
        text
    )

    text = text.replace(
        r"\cdot \cdot \cdot",
        r"\cdots"
    )

    return text.strip()

# =========================================================
# FIND FILE
# =========================================================
def find_middle_json(pdf_name):
    base = os.path.splitext(pdf_name)[0]

    path = os.path.join(
        MINERU_DIR,
        base,
        base,
        "auto",
        f"{base}_middle.json"
    )

    return path if os.path.exists(path) else None


# =========================================================
# MAIN WRAPPER
# =========================================================
def extract_raw(base_dir: str, label: str, pdf) -> Tuple[bool, List]:

    path = find_middle_json(pdf.pdf_name)

    if not path:
        return False, []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    outputs = []
    
    front_blocks = []

    
    if data.get("pdf_info"):
        first_page = data["pdf_info"][0]

        for block in first_page.get("para_blocks", []):

            txt = get_text(block)

            front_blocks.append(
                (block.get("type", ""), txt)
            )
    for page in data.get("pdf_info", []):

        page_num = page.get("page_idx", 0) + 1

        for block in page.get("para_blocks", []):

            btype = block.get("type", "")

            # =====================================================
            # TITLE
            # =====================================================
            if label == "title" and btype == "title":
                if outputs:
                    continue
                outputs.append(
                    (pdf.pdf_name, page_num, "title", get_text(block))
                )

            # =====================================================
            # ABSTRACT
            # =====================================================
            elif label == "abstract" and btype == "abstract":
                outputs.append(
                    (pdf.pdf_name, page_num, "abstract", get_text(block))
                )

            # =====================================================
            # REFERENCE
            # =====================================================
            elif label == "reference" and btype == "ref_text":
                outputs.append(
                    (pdf.pdf_name, page_num, "reference", get_text(block))
                )

            # =====================================================
            # EQUATION (FINAL FIX)
            # =====================================================
            elif label == "equation":
                
                eqs = extract_equations(block)

                
                    
                eqs = clean_list(eqs)
                for e in eqs:
                    e = normalize_equation(e)
                    outputs.append(
                        (pdf.pdf_name, page_num, "equation", e)
                    )
            # =====================================================
            # CAPTION (ROBUST)
            # =====================================================
            elif label == "caption":
                for sub in block.get("blocks", []):
                    stype = sub.get("type", "")
                    if "caption" not in stype:
                        continue
                    txt = get_text(sub)
                    txt = re.sub(
                        r"^(Figure|Fig\.?|Table)\s*\d+\s*[:.]?\s*",
                        "",
                        txt,
                        flags=re.I
                    )
                    txt = txt.strip()
                    if txt:
                        outputs.append(
                            (pdf.pdf_name, page_num, "caption", txt)
                        )
            # =====================================================
            # TABLE (CLEAN HTML)
            # ===================================================== 
            elif label == "table":
                if "table" in btype:

                    txt = ""

                    for sub in block.get("blocks", []):

                        for line in sub.get("lines", []):

                            for span in line.get("spans", []):

                                txt += span.get("html", "") + " "

                    txt = txt.strip()

                    if txt:
                        outputs.append(
                            (pdf.pdf_name, page_num, "table", txt)
                        )
    # =====================================================
    # AUTHOR / AFFILIATION / EMAIL / SECTION
    # =====================================================

    if label == "email":

        for _, txt in front_blocks:

            for em in EMAIL_RE.findall(txt):

                outputs.append(
                    (pdf.pdf_name, 1, "email", em)
                )

    elif label == "affiliation":

        for _, txt in front_blocks:

            if AFFILIATION_RE.search(txt):

                outputs.append(
                    (pdf.pdf_name, 1, "affiliation", txt)
                )

    elif label == "author":

        for btype, txt in front_blocks[:5]:

            if btype not in ["text", "list"]:
                continue

            txt = re.sub(
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                " ",
                txt
            )

            if AFFILIATION_RE.search(txt):
                continue

            txt = re.sub(r"<sup>.*?</sup>", " ", txt)

            parts = re.split(
                r",|;|\band\b|\&",
                txt
            )

            for p in parts:

                p = p.strip()
 
                if looks_like_author_name(p):

                    outputs.append(
                        (pdf.pdf_name, 1, "author", p)
                    )
 
    elif label == "section":

        first_title = True

        for page in data.get("pdf_info", []):

            page_num = page.get("page_idx", 0) + 1

            for block in page.get("para_blocks", []):

                if block.get("type") != "title":
                    continue

                txt = get_text(block)

                if not txt:
                    continue

                if first_title:
                    first_title = False
                    continue

                outputs.append(
                    (pdf.pdf_name, page_num, "section", txt)
                )



    elif label == "keyword":

        for _, txt in front_blocks:

            low = txt.lower()

            if "keyword" not in low:
                continue

            txt = re.sub(
                r"(?i).*keywords?\s*[:.]?\s*",
                "",
                txt
            )

            txt = re.sub(
                r"<sup>.*?</sup>",
                " ",
                txt
            )

            parts = re.split(
                r"\s*·\s*|\s*;\s*|\s*,\s*",
                txt
            )

            for p in parts:

                p = p.strip(" .")

                if len(p) < 2:
                    continue

                        

                outputs.append(
                    (pdf.pdf_name, 1, "keyword", p)
                )

    


    return True, outputs
