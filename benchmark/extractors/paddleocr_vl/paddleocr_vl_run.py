import json
import re
import html
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path

from ...normalisation import normalize_string



OUTPUT_ROOT = Path(
    "/scratch/nasimb/escibench_project/paddleocr_vl_workspace/paddleocr_vl_outputs"
)


def output_dir(pdf) -> Path:
    """Return the PaddleOCR-VL output directory for a PDF."""
    return OUTPUT_ROOT / Path(pdf.pdf_name).stem


def load_pages(pdf):
    """Load all PaddleOCR-VL page JSON files for a PDF."""

    pdf_dir = output_dir(pdf)

    if not pdf_dir.exists():
        return []

    pages = []

    for path in sorted(pdf_dir.glob("*_res.json")):
        with open(path, "r", encoding="utf-8") as f:
            pages.append(json.load(f))

    return pages


def page_blocks(page):
    """Return all parsed blocks from one page."""
    return page.get("parsing_res_list", [])


def block_text(block):
    """Return normalized block text."""
    return normalize_string(block.get("block_content", ""))

def collect_blocks(pages, label):
    """
    Collect blocks with a specific PaddleOCR-VL label.
    Returns a list of (page_index, text).
    """

    results = []

    for page in pages:

        page_index = page["page_index"]

        for block in page_blocks(page):

            if block.get("block_label") != label:
                continue

            text = block_text(block)

            if text:
                results.append((page_index, text))

    return results


def collect_reference_blocks(pages):
    """
    Collect reference blocks.

    Primary source:
        reference_content

    Fallback:
        On pages immediately preceding the first reference_content page,
        accept text blocks that look like numbered references.
    """

    results = []

    first_ref_page = None

    # -----------------------------
    # Locate first page containing reference_content
    # -----------------------------

    for page in pages:

        if any(
            b.get("block_label") == "reference_content"
            for b in page_blocks(page)
        ):
            first_ref_page = page["page_index"]
            break

    if first_ref_page is None:
        return results

    # -----------------------------
    # Normal reference_content
    # -----------------------------

    for page in pages:

        page_index = page["page_index"]

        for block in page_blocks(page):

            if block.get("block_label") != "reference_content":
                continue

            text = block_text(block)

            if text:
                results.append((page_index, text))

    # -----------------------------
    # Fallback pages immediately before bibliography
    # -----------------------------

    for page in pages:

        page_index = page["page_index"]

        if page_index >= first_ref_page:
            continue

        if first_ref_page - page_index > 8:
            continue

        labels = {
            b.get("block_label")
            for b in page_blocks(page)
        }

        # only bibliography-like pages
        if labels - {"text", "number"}:
            continue

        for block in page_blocks(page):

            if block.get("block_label") != "text":
                continue

            text = block_text(block)

            if not text:
                continue

            if re.match(r"^\[\d+\]", text):
                results.append((page_index, text))

    results.sort(key=lambda x: (x[0], x[1]))

    return results


def extract_title(pdf):

    blocks = collect_blocks(load_pages(pdf), "doc_title")

    if not blocks:
        return False, []

    page, text = blocks[0]

    return True, [
        (
            pdf.pdf_name,
            page,
            "title",
            text,
        )
    ]

def normalize_caption(text):

    text = normalize_string(text)

    text = re.sub(
        r"^(?:Fig(?:ure)?\.?\s*\d+[A-Za-z]?[.:]?\s*)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()

def extract_caption(pdf):

    pages = load_pages(pdf)

    if not pages:
        return False, []

    captions = []

    seen = set()

    for page in pages:

        page_index = page["page_index"]

        blocks = [
            b for b in page_blocks(page)
            if b.get("block_label") == "figure_title"
        ]

        i = 0

        while i < len(blocks):

            text = block_text(blocks[i])

            if not text:
                i += 1
                continue

            # ---------- Figure caption ----------

            if re.match(r"^Fig(?:ure)?\.?", text, re.I):

                caption = normalize_caption(text)

                if caption and caption not in seen:

                    seen.add(caption)

                    captions.append(
                        (
                            pdf.pdf_name,
                            page_index,
                            "caption",
                            caption,
                        )
                    )

                i += 1
                continue

            # ---------- Table caption ----------

            if re.match(r"^TABLE\b", text, re.I):

                m = re.match(
                    r"^(TABLE\s+[A-Za-z0-9IVX]+[.:]?)\s*(.*)$",
                    text,
                    flags=re.I,
                )

                caption = ""

                if m:
                    caption = m.group(2).strip()

                if not caption and i + 1 < len(blocks):
                    caption = block_text(blocks[i + 1])

                caption = normalize_caption(caption)

                if caption and caption not in seen:
                    seen.add(caption)

                    captions.append(
                        (
                            pdf.pdf_name,
                            page_index,
                            "caption",
                            caption,
                        )
                    )
                if caption and m and not m.group(2).strip():
                    i += 2
                else:
                    i += 1
                continue

            i += 1

    if not captions:
        return False, []

    return True, captions

def extract_abstract(pdf):

    pages = load_pages(pdf)

    blocks = []

    for page in pages:
        page_blocks = collect_blocks([page], "abstract")
        if page_blocks:
            blocks = page_blocks
            break



    if not blocks:
        return False, []

    page = blocks[0][0]

    abstract = normalize_string(
        " ".join(text for _, text in blocks)
    )

    return True, [
        (
            pdf.pdf_name,
            page,
            "abstract",
            abstract,
        )
    ]

_MONTHS = {
    "jan": "January",
    "feb": "February",
    "mar": "March",
    "apr": "April",
    "may": "May",
    "jun": "June",
    "jul": "July",
    "aug": "August",
    "sep": "September",
    "sept": "September",
    "oct": "October",
    "nov": "November",
    "dec": "December",
}


def normalize_pub_date(text):

    text = normalize_string(text).strip()

    # remove common prefixes
    text = re.sub(
        r"^(received|accepted|published|available online|online published|dated)\s*[:\-]?\s*",
        "",
        text,
        flags=re.I,
    )

    # ----------------------------
    # ISO format
    # 2024-05-17
    # ----------------------------

    m = re.search(
        r"\b(\d{4})-(\d{2})-(\d{2})\b",
        text,
    )

    if m:
        y, mon, d = m.groups()
        try:
            dt = datetime.strptime(
                f"{y}-{mon}-{d}",
                "%Y-%m-%d",
            )
            return dt.strftime("%B %-d, %Y")
        except ValueError:
            pass

    # ----------------------------
    # 23 Nov 2016
    # 23 Nov. 2016
    # ----------------------------

    m = re.search(
        r"(\d{1,2})\s+"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
        r"[a-z]*\s+"
        r"(\d{4})",
        text,
        re.I,
    )

    if m:
        d, mon, y = m.groups()
        return f"{_MONTHS[mon.lower()]} {int(d)}, {y}"

    # ----------------------------
    # Nov. 23, 2016
    # November 23, 2016
    # ----------------------------

    m = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
        r"January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\.?\s+"
        r"(\d{1,2}),?\s+"
        r"(\d{4})",
        text,
        re.I,
    )

    if m:

        mon, d, y = m.groups()

        key = mon.lower().rstrip(".")

        if key in _MONTHS:
            mon = _MONTHS[key]
        else:
            mon = mon.capitalize()

        return f"{mon} {int(d)}, {y}"

    # ----------------------------
    # 22 June 2022
    # ----------------------------

    m = re.search(
        r"(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{4})",
        text,
        re.I,
    )

    if m:
        d, mon, y = m.groups()
        return f"{mon.capitalize()} {int(d)}, {y}"

    # ----------------------------
    # April 2021
    # Apr. 2021
    # ----------------------------

    m = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
        r"January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\.?\s+"
        r"(\d{4})",
        text,
        re.I,
    )

    if m:

        mon, y = m.groups()

        key = mon.lower().rstrip(".")

        if key in _MONTHS:
            mon = _MONTHS[key]
        else:
            mon = mon.capitalize()

        return f"{mon} {y}"

    return None

def iter_candidate_blocks(blocks, max_blocks=18, max_len=120):

    allowed_labels = {
        "header",
        "text",
        "footer",
        "footnote",
    }

    inspected = 0

    for block in blocks:

        label = block.get("block_label", "")

        if label not in allowed_labels:
            continue

        inspected += 1

        if inspected > max_blocks:
            break

        text = block_text(block)

        if not text:
            continue

        if len(text) > max_len:
            continue

        yield (
            block,
            text,
        )
        
def extract_pub_date(pdf):

    pages = load_pages(pdf)

    if not pages:
        return False, []

    page = pages[0]
    blocks = page_blocks(page)
    
    PRIORITY = {
        "published": 100,
        "available": 90,
        "accepted": 70,
        "revised": 60,
        "received": 50,
        "submitted": 40,
        "preprint": 30,
        "arxiv": 20,
        "generic": 0,
    }

    candidates = []

    allowed = {
        "header",
        "text",
        "footer",
        "footnote",
        "title",
        "caption",
    }

    for block in blocks:

        label = block.get("block_label", "").lower()

        if label not in allowed:
            continue

        txt = block_text(block)
        txt = re.sub(r"\s+", " ", txt)

        if not txt:
            continue

        txt = txt.replace("\r", "\n")

        txt = txt.replace("•", " ")
        txt = txt.replace("|", " ")
        txt = txt.replace("·", " ")
        txt = re.sub(r"\s+", " ", txt)

        pieces = []

        for line in re.split(r"[;\n]", txt):
            line = line.strip()

            if not line:
                continue

            line = line.strip(".,;:()[]{}")

            parts = re.split(
                r"""(?ix)
                (?=
                    published|
                    publication|
                    accepted|
                    received|
                    revised|
                    submitted|
                    available\s+online|
                    online|
                    first\s+published|
                    published\s+online|
                    date\s+of\s+publication|
                    preprint|
                    this\s+version|
                    dated:|
                    arxiv:
                )
                """,
                line,
            )

            for p in parts:
                p = p.strip()
                if p:
                    pieces.append(p)

        for piece in pieces:
            piece = piece.strip()

            if len(piece) > 120:
                continue

            piece = re.sub(r"\s+", " ", piece)

            date = normalize_pub_date(piece)

            if pdf.pdf_name == "topologicalAnalysisTruncated.pdf":
                print("PIECE :", repr(piece))
                print("DATE  :", date)

            if not date:
                continue

            low = piece.lower()

            score = PRIORITY["generic"]

            if "published" in low:
                score = PRIORITY["published"]

            elif (
                "available online" in low
                or "published online" in low
                or "first published" in low
                or "date of publication" in low
            ):
                score = PRIORITY["available"]

            elif "accepted" in low:
                score = PRIORITY["accepted"]

            elif "revised" in low:
                score = PRIORITY["revised"]

            elif "received" in low:
                score = PRIORITY["received"]

            elif "submitted" in low:
                score = PRIORITY["submitted"]

            elif "preprint" in low:
                score = PRIORITY["preprint"]

            elif low.startswith("arxiv"):
                score = PRIORITY["arxiv"]

            candidates.append(
                {
                    "score": score,
                    "date": date,
                    "page": page["page_index"],
                    "text": piece,
                }
            )

    if not candidates:
        return False, []

    candidates.sort(
        key=lambda x: (
            -x["score"],
            len(x["text"]),
        )
    )

    best = candidates[0]

    return True, [
        (
            pdf.pdf_name,
            best["page"],
            "pub_date",
            best["date"],
        )
    ]

def split_compact_author_names(text):

    import re

    text = re.sub(r"\s+", " ", text).strip()

    # If separators already exist, leave unchanged.
    if "," in text or " and " in text.lower():
        return [text]

    words = text.split()

    # One author with multiple initials:
    # Hamish A. S. Reid
    # José C. M. Bermudez
    # Luciano C. Ayres

    import re

    initials = sum(
        1
        for w in words
        if re.fullmatch(r"[A-Z]\.", w)
    )

    if initials >= 2:
        return [text]

    if len(words) < 4:
        return [text]

    # Don't split names containing surname particles.
    # Examples:
    # Arjan van der Schaft
    # Jacobo Ruiz de Elvira

    particles = {"van", "von", "de", "del", "der", "di", "da"}

    if any(w.lower() in particles for w in words):
        return [text]

    # Typical pattern:
    # First Last First Last
    if len(words) % 2 == 0:
        ok = True
        for w in words:
            if len(w) == 1:
                ok = False
                break
            if len(w) == 2 and w.endswith("."):
                ok = False
                break
        if ok:
            names = []
            for i in range(0, len(words), 2):
                names.append(words[i] + " " + words[i + 1])
            return names

    return [text]

def split_author_block(text):

    import re

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove LaTeX affiliation markers such as
    # $^{1}$, ${}^{1}$, $^{1,2,*}$, etc.
    text = re.sub(r"\$\s*\{?\s*\}\?\s*\^\{[^}]*\}\s*\$", ", ", text)
    text = re.sub(r"\$\s*\^\{[^}]*\}\s*\$", ", ", text)
    text = re.sub(r"\^\{[^}]*\}", ", ", text)

    # Remove remaining dollar signs
    text = text.replace("$", " ")

    # Remove ORCID identifiers such as [0000-0002-1825-0097]
    text = re.sub(
        r"\[\d{4}-\d{4}-\d{4}-\d{3}[\dX]\]",
        "",
        text,
    )

    # Replace author footnote markers with commas.
    text = re.sub(
        r"\s*[*†‡§¶]+\s*",
        ", ",
        text,
    )

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove everything starting from the first email.
    # Emails never belong to author names.
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b.*",
        "",
        text,
    )

    # Normalize separators
    text = re.sub(r"\band\b", ",", text, flags=re.I)

    parts = [
        p.strip(" ,;")
        for p in text.split(",")
        if p.strip(" ,;")
    ]

    final_parts = []

    for part in parts:
        final_parts.extend(split_compact_author_names(part))

    return final_parts


def is_author_candidate(text):

    import re

    low = text.lower().strip()

    BAD_PHRASES = [
        "index terms",
        "keywords",
        "key words",
        "student member",
        "senior member",
        "member ieee",
        "on behalf of",
        "these authors contributed",
        "contributed equally",
        "equal contribution",
        "corresponding author",
        "supervisor",
        "thesis",
        "degree of",
        "accepted",
        "received",
        "date of receipt",
        "acceptance should",
        "pacs numbers",
        "mathematics subject classification",
    ]

    for phrase in BAD_PHRASES:
        if phrase in low:
            return False

    #
    # Reject obvious affiliation/institution lines.
    #
    AFFILIATION_HINTS = [
        "university",
        "université",
        "universitat",
        "università",
        "universidad",
        "institute",
        "institut",
        "department",
        "departament",
        "departamento",
        "faculty",
        "school of",
        "college",
        "laboratory",
        "laboratoire",
        "centre for",
        "center for",
        "research center",
        "research centre",
        "academy",
        "hospital",
        "clinic",
        "google llc",
        "iqvia",
    ]

    if any(hint in low for hint in AFFILIATION_HINTS):
         return False

    #
    # Reject obvious running text.
    #
    if len(text.split()) > 8:
        return False

    if re.search(
        r"\b(our|this|we|suggesting|observe|based|approach)\b",
        low,
    ):
        return False

    return True

def extract_keyword(pdf):

    import re

    pages = load_pages(pdf)

    if not pages:
        return False, []

    page = pages[0]
    blocks = page_blocks(page)
   

    keywords = []
    waiting = False

    def parse_keyword_text(text):

        text = text.split("|")[0]

        text = re.sub(
            r'^(keywords?|key\s*words?|key\s*words?\s*and\s*phrases?|index\s*terms?)\s*[:.\-—]*\s*',
            '',
            text,
            flags=re.I,
        )

        text = re.split(
            r'JEL\s+Classification|Mathematics\s+Subject\s+Classification|MSC',
            text,
            flags=re.I,
        )[0]

        text = text.replace("—", ",")
        text = text.replace("–", ",")
        text = text.replace("·", ",")
        text = text.replace(";", ",")

        out = []

        for kw in text.split(","):

            kw = normalize_string(kw)
            
            

            if kw:
                out.append(kw)

        return out

    for i, block in enumerate(blocks):

        label = block.get("block_label", "")
        text = block.get("block_content", "").strip()

        if not text:
            continue

        low = text.lower()


        # paragraph_title -> next text block
        if label == "paragraph_title" and (
            "keyword" in low
            or "key word" in low
            or "index term" in low
        ):
            waiting = True
            continue

        if waiting:

            if label != "text":
                continue

            for kw in parse_keyword_text(text):
                keywords.append(
                    (
                        pdf.pdf_name,
                        page["page_index"],
                        "keyword",
                        kw,
                    )
                )
            break

        # inline keyword block
        if (
            low.startswith("keywords")
            or low.startswith("keyword")
            or low.startswith("key words")
            or low.startswith("index terms")
            or low.startswith("index term")
            or low.startswith("key words and phrases")
        ):

            for kw in parse_keyword_text(text):
                keywords.append(
                    (
                        pdf.pdf_name,
                        page["page_index"],
                        "keyword",
                        kw,
                    )
                )

            break

    if not keywords:
        return False, []

    return True, keywords

    


def extract_author(pdf):

    import re

    BAD_WORDS = {
        "university",
        "department",
        "institute",
        "laboratory",
        "school",
        "faculty",
        "college",
        "road",
        "street",
        "keywords",
        "classification",
        "preprint",
        "abstract",
        "telephone",
        "phone",
        "e-mail",
        "email",
        "@",
        "group",
        "lab",
        "centre",
        "center",
        "research",
        "science",
        "engineering",
        "technology",
        "technische",
        "universität",
        "universite",
        "università",
        "universitat",
        "neuroimaging",
        "processing",
    }

    pages = load_pages(pdf)

    if not pages:
        return False, []

    page = pages[0]

    authors = []

    blocks = page_blocks(page)
    
    title_seen = False
    blocks_after_title = 0

    for block in blocks:

        # Ignore everything before the title
        if not title_seen:
            if block.get("block_label") == "doc_title":
                title_seen = True
            continue

        blocks_after_title += 1


        if blocks_after_title > 15:
            break

        if block.get("block_label") in (
            "paragraph_title",
            "abstract",
        ):
            break


        if block.get("block_label") != "text":
            continue

        raw_text = block.get("block_content", "")
        lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

        merged_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            if i + 1 < len(lines):

                nxt = lines[i + 1]

                import re

                # Merge:
                # Hamish A.  +  S. Reid
                # Arjan van  +  der Schaft
                # Jacobo Ruiz + de Elvira

                if (
                    re.search(r"(?:[A-Z]\.|van|von|de|del|der|di|da)$", line, re.I)
                    or re.match(r"^(?:[A-Z]\.|van|von|de|del|der|di|da)\b", nxt, re.I)
                ):
                    merged_lines.append(line + " " + nxt)
                    i += 2
                    continue

            merged_lines.append(line)
            i += 1

        lines = merged_lines

        if not lines:
            continue


        for text in lines:

            low = text.lower()
            

            if any(word in low for word in BAD_WORDS):
                
                continue
            
            check_text = re.sub(
                r"\[\d{4}-\d{4}-\d{4}-\d{3}[\dX]\]",
                "",
                text,
            )

            if re.search(r"\b\d{4,}\b", check_text):
                
                continue

            
            if len(text.split()) > 20:
                candidates = [
                    a
                    for a in split_author_block(text)
                    if len(a.split()) >= 2
                    and is_author_candidate(a)
                ]

                if len(candidates) < 2:
                    continue

            seen = set()

            for author in split_author_block(text):

                author = normalize_string(author)
                # Remove common affiliation markers at the end of author names.
                author = re.sub(r'[\{\}\[\]\(\)\*\†\‡\§\♣\♥\▲\△\◆\◇\^]+$', '', author).strip()

                # Remove trailing single-letter affiliation labels (e.g., "a", "b", "c").
                author = re.sub(r'\s+[a-z]$', '', author).strip()

                if author in seen:
                    continue
                seen.add(author)

                if len(author.split()) < 2:
                    continue

                if not is_author_candidate(author):
                    continue

                authors.append(
                    (
                        pdf.pdf_name,
                        page["page_index"],
                        "author",
                        normalize_string(author),
                    )
                )

    if not authors:
        return False, []

    return True, authors

def extract_affiliation(pdf):

    
    pages = load_pages(pdf)

    if not pages:
        return False, []

    page = pages[0]
    blocks = page_blocks(page)

    affiliations = []

    title_seen = False
    author_seen = False

    current = []

    for block in blocks:

        label = block.get("block_label")
        text = normalize_string(block.get("block_content", ""))

        if not text:
            continue

        if not title_seen:
            if label == "doc_title":
                title_seen = True
            continue

        if label in ("abstract", "paragraph_title"):
            break

        if label != "text":
            continue

        low = text.lower()


        # first text block after title usually contains authors,
        # but sometimes also contains affiliations.
        if not author_seen:
            
            author_seen = True

        

            if not re.search(
                r"\b("
                r"university|college|school|department|faculty|"
                r"institute|laboratory|laboratoire|lab|"
                r"center|centre|academy|hospital|"
                r"caltech|technion|google|meta|nvidia|"
                r"engineering|computer\s+science|physics|mathematics"
                r")\b",
                low,
                re.I,
            ):
                continue

            m = re.search(
                r"\b("
                r"university|college|school|department|faculty|"
                r"institute|laboratory|laboratoire|lab|"
                r"center|centre|academy|hospital|"
                r"caltech|technion|google|meta|nvidia|"
                r"engineering|computer\s+science|physics|mathematics"
                r")\b",
                low,
                re.I,

            )


            if not m:
                continue

        # Remove email addresses but keep the affiliation text.
        text = re.sub(r"\S+@\S+", "", text).strip()
        low = text.lower()

        text = re.split(
            r"(?i)\*?\s*(corresponding\s+author|corresponding\s+authors|correspondence|e-?mail)\b",
            text,
            maxsplit=1,
        )[0].strip()

        low = text.lower()

        if not text or low.startswith("email"):
            continue

        # Split affiliations only when explicit superscript markers exist.
        if re.search(r"\$\s*\^\{\d+\}\s*\$", text):

            parts = re.split(r"\$\s*\^\{\d+\}\s*\$", text)

            for part in parts:

                part = part.strip()

                if not part:
                     continue

                part = re.sub(
                    r"\(dated:.*?\)",
                    "",
                    part,
                    flags=re.I | re.S,
                ).strip()

                if part:
                    current.append(part)
                    

        else:
            
            if (
                current
                and current[-1].rstrip().endswith(",")
                and not re.search(r"\$\s*\^\{", text)
                and "@" not in text
                and not re.search(
                    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
                    text,
                    re.I,
                )
            ):
                current[-1] += " " + text
            else:

                affiliation_hint = re.search(
                    r"\b("
                    r"university|institute|institutes|college|school|academy|"
                    r"laboratory|laboratoire|lab|"
                    r"department|centre|center|"
                    r"technion|caltech|dipartimento|"
                    r"mathematics|physics|computer\s+science|"
                    r"engineering|signaux|systemes|systèmes|"
                    r"hospital|faculty|"
                    r"google|deepmind|meta|"
                    r"microsoft|amazon|apple|ibm|nvidia|intel|"
                    r"adobe|bytedance|huawei|qualcomm|"
                    r"salesforce|oracle"
                    r")\b",
                    low,
                    re.I,
                )

                if not affiliation_hint:
                    continue
                    
                current.append(
                    re.sub(
                         r"(?i)\s*(?:\$\s*\^\{\*+\}\s*)?\*?\s*corresponding authors?:.*$",
                         "",
                         text,
                    ).strip()
                )

      
            
    for aff in current:

        affiliations.append(
            (
                pdf.pdf_name,
                page["page_index"],
                "affiliation",
                normalize_string(aff),
            )
        )

    # fallback: affiliations stored in footnotes
    for block in blocks:


        if block.get("block_label") != "footnote":
            continue

        text = block.get("block_content", "")

        if not text:
            continue

        # Remove author name if it occupies the first line.
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) >= 2:
            text = " ".join(lines[1:])
        else:
            text = lines[0]

        text = normalize_string(text)
        
        low = text.lower()

        # Skip editorial notes.
        if "date of receipt" in low or "acceptance should be inserted" in low:
            continue

        # Remove e-mail instead of dropping the affiliation.
        text = re.sub(r"e-?mail\s*:.*$", "", text, flags=re.I).strip()
        text = re.sub(r"\S+@\S+", "", text).strip()


        if not text:
            continue

        # obvious non-affiliation footnotes
        if re.fullmatch(r"[\$\^\{\}\*\+\-†‡§¶\s]+", text):
            continue

        if re.search(
            r"(corresponding author|equal contribution|equal contributions|"
            r"contributed equally|speaker|email address|email addresses|"
            r"dated:|received useful feedback|we are grateful|"
            r"funding|supported by|grant|deceased|"
            r"now affiliated with|originally submitted|"
            r"work done during internship)",
            low,
        ):
            continue

        if re.fullmatch(
            r"\(?dated:\s*.*\)?|"
            r"(january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+\d{1,2},\s+\d{4}\.?",
            low,
        ):
            continue

        affiliations.append(
            (
                pdf.pdf_name,
                page["page_index"],
                "affiliation",
                text,
            )
        )



    if not affiliations:
        return False, []

    return True, affiliations


def extract_email(pdf):
    

    pages = load_pages(pdf)

    if not pages:
        return False, []

    emails = []
    seen = set()

    email_pattern = re.compile(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    )

    group_email_pattern = re.compile(
        r"[\{\[]([^\{\}\[\]]+)[\}\]]@([A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    )


    for page in pages:

        page_index = page["page_index"]

        for block in page_blocks(page):

            text = block.get("block_content", "")

            if not text:
                continue

            for local_part, domain in group_email_pattern.findall(text):

                local_part = re.sub(
                    r"([A-Za-z0-9_.-]+)\s+([A-Za-z0-9_.-]+)",
                    r"\1,\2",
                    local_part,
                )

                for name in local_part.split(","):

                    name = name.strip()

                    if not name:
                        continue

                    email = normalize_string(f"{name}@{domain}").lower()

                    if email in seen:
                        continue

                    seen.add(email)

                    emails.append(
                        (
                            pdf.pdf_name,
                            page_index,
                            "email",
                            email,
                        )
                    )

                    
            for email in email_pattern.findall(text):

                email = normalize_string(email).lower()

                if email in seen:
                    continue

                seen.add(email)

                emails.append(
                    (
                        pdf.pdf_name,
                        page_index,
                        "email",
                        email,
                    )
                )

    if not emails:
        return False, []

    return True, emails 

def remove_display_math(text):

    text = text.replace("$$", "")
    text = text.replace("\\[", "")
    text = text.replace("\\]", "")

    return text

def remove_displaystyle(text):

    return re.sub(
        r'\\displaystyle\b',
        '',
        text,
    )

def remove_spacing_commands(text):

    return re.sub(
        r'\\(?:quad|qquad|,|;|!|:|enspace|thinspace|medspace|thickspace)\b',
        '',
        text,
    )

def remove_alignment(text):

    return text.replace("&", " ")

def remove_linebreaks(text):

    return re.sub(
        r'\\\\+',
        ' ',
        text,
    )


def preprocess_equation(text):

    text = remove_display_math(text)

    # remove LaTeX environments
    text = re.sub(
        r'\\(?:begin|end)\{(?:aligned|align|array|cases|matrix|bmatrix|pmatrix|vmatrix|Vmatrix|smallmatrix|split|gathered?)\}',
        '',
        text,
    )

    # remove displaylimits
    text = re.sub(
        r'\\displaylimits\b',
        '',
        text,
    )

    text = remove_displaystyle(text)

    text = remove_spacing_commands(text)

    text = remove_alignment(text)

    text = remove_linebreaks(text)

    text = text.replace("∈fty", "∞")

    text = re.sub(r'begin\{?align\*?\}?', '', text)
    text = re.sub(r'end\{?align\*?\}?', '', text)

    text = re.sub(r'beginalign\*?', '', text)
    text = re.sub(r'endalign\*?', '', text)

    text = text.replace("parallel", "∥")
    text = text.replace("perp", "⊥")

    text = text.replace("sqrt2", "√2")

    return text

def unwrap_command(text, command):

    pattern = rf'\\{command}\{{([^{{}}]*)\}}'

    while True:

        new = re.sub(pattern, r'\1', text)

        if new == text:
            break

        text = new

    return text

def unwrap_latex(text):

    commands = [

        "mathrm",
        "mathbb",
        "mathcal",
        "mathbf",
        "mathit",
        "mathsf",
        "mathtt",
        "boldsymbol",
        "bm",
        "operatorname",
        "text",

    ]

    for cmd in commands:

        text = unwrap_command(text, cmd)

    return text

def normalize_accents(text):

    accent_commands = [
        "widehat",
        "hat",
        "widetilde",
        "tilde",
        "overline",
        "underline",
        "bar",
        "vec",
        "dot",
        "ddot",
        "acute",
        "grave",
        "breve",
        "check",
    ]

    for cmd in accent_commands:
        text = unwrap_command(text, cmd)

    text = re.sub(
        r'\\?(?:widehat|hat|widetilde|tilde|overline|underline|bar|vec|dot|ddot|acute|grave|breve|check)(?=[A-Za-zΑ-Ωα-ω])',
        '',
        text,
    )

    return text

def normalize_operatorname(text):

    text = unwrap_command(text, "operatorname")

    text = re.sub(
        r'\\?operatorname\*?',
        '',
        text,
    )

    keyword_map = {
        r'\bl\s*i\s*m\b': 'lim',
        r'\bs\s*u\s*p\b': 'sup',
        r'\bi\s*n\s*f\b': 'inf',
        r'\bm\s*i\s*n\b': 'min',
        r'\bm\s*a\s*x\b': 'max',
        r'\ba\s*r\s*g\b': 'arg',
    }

    for pattern, repl in keyword_map.items():
        text = re.sub(pattern, repl, text)
        
    return text

def normalize_arrows(text):

    arrow_replacements = {
        r'\\xrightarrow': '→',
        r'\\xleftarrow': '←',
        r'\\rightarrow': '→',
        r'\\leftarrow': '←',
        r'\\Rightarrow': '⇒',
        r'\\Leftarrow': '⇐',
        r'\\leftrightarrow': '↔',
        r'\\Leftrightarrow': '⇔',
        r'\\uparrow': '↑',
        r'\\downarrow': '↓',
    }

    for old, new in arrow_replacements.items():
        text = text.replace(old, new)

    arrow_words = {
        "xrightarrow": "→",
        "xleftarrow": "←",
        "rightarrow": "→",
        "leftarrow": "←",
        "Rightarrow": "⇒",
        "Leftarrow": "⇐",
        "leftrightarrow": "↔",
        "Leftrightarrow": "⇔",
        "uparrow": "↑",
        "downarrow": "↓",
    }

    for old, new in arrow_words.items():
        text = text.replace(old, new)

    return text

def normalize_floorceil(text):

    floorceil = {
        r'\\lfloor': '⌊',
        r'\\rfloor': '⌋',
        r'\\lceil': '⌈',
        r'\\rceil': '⌉',
    }

    for old, new in floorceil.items():
        text = text.replace(old, new)

    floorceil_words = {
        "lfloor": "⌊",
        "rfloor": "⌋",
        "lceil": "⌈",
        "rceil": "⌉",
    }

    for old, new in floorceil_words.items():
        text = text.replace(old, new)

    return text

def apply_replacements(text, rules):

    for old, new in rules.items():
        text = text.replace(old, new)

    return text

REPLACEMENT_RULES = {
    r'\to': '→',
    r'\rightarrow': '→',
    r'\leftarrow': '←',
    r'\mapsto': '↦',
    r'\cong': '≅',
    r'\in': '∈',
    r'\notin': '∉',
    r'\leq': '≤',
    r'\geq': '≥',
    r'\neq': '≠',
    r'\times': '×',
    r'\cdot': '·',
    r'\pm': '±',

    r'\int': '∫',
    r'\prod': '∏',
    r'\cup': '∪',
    r'\cap': '∩',
    r'\subseteq': '⊆',
    r'\supseteq': '⊇',
    r'\forall': '∀',
    r'\exists': '∃',
    r'\otimes': '⊗',
    r'\oplus': '⊕',
    r'\sim': '∼',
    r'\approx': '≈',
    r'\propto': '∝',
    r'\iff': '⇔',
    r'\implies': '⇒',
    r'\Longrightarrow': '⇒',
    r'\Longleftrightarrow': '⇔',
    r'\equiv': '≡',
    r'\subset': '⊂',
    r'\supset': '⊃',
    r'\emptyset': '∅',
    r'\varnothing': '∅',
    r'\infty': '∞',
}

GREEK_RULES = {
    r'\chi': 'χ',
    r'\xi': 'ξ',
    r'\mu': 'μ',
    r'\Lambda': 'Λ',
    r'\Omega': 'Ω',
    r'\Delta': 'Δ',
    r'\Phi': 'Φ',
    r'\Psi': 'Ψ',
    r'\Gamma': 'Γ',
    r'\pi': 'π',
    r'\Pi': 'Π',
    r'\sigma': 'σ',
    r'\Sigma': 'Σ',
    r'\phi': 'φ',
    r'\Phi': 'Φ',
    r'\omega': 'ω',
    r'\Omega': 'Ω',
    r'\lambda': 'λ',
    r'\Lambda': 'Λ',
    r'\alpha': 'α',
    r'\beta': 'β',
    r'\gamma': 'γ',
    r'\delta': 'δ',
    r'\epsilon': 'ε',
    r'\varepsilon': 'ε',
    r'\rho': 'ρ',
    r'\tau': 'τ',
    r'\zeta': 'ζ',
    r'\eta': 'η',
    r'\theta': 'θ',
    r'\vartheta': 'ϑ',
    r'\kappa': 'κ',
    r'\nu': 'ν',
    r'\upsilon': 'υ',
    r'\psi': 'ψ',
}

def normalize_frac(text):

    # frac12 -> 1/2
    text = re.sub(
        r'frac([0-9]+)([0-9]+)',
        r'\1/\2',
        text,
    )

    # fracxy -> x/y
    text = re.sub(
        r'frac([A-Za-zα-ωΑ-Ω])([A-Za-zα-ωΑ-Ω])',
        r'\1/\2',
        text,
    )

    # frac∂x -> ∂/x
    text = re.sub(
        r'frac(∂)([A-Za-zα-ωΑ-Ω])',
        r'\1/\2',
        text,
    )

    # fracdx -> d/x
    text = re.sub(
        r'frac(d)([A-Za-zα-ωΑ-Ω])',
        r'\1/\2',
        text,
    )

    return text

def normalize_equation(text):

    text = normalize_string(text)

    text = preprocess_equation(text)

    text = unwrap_latex(text)

    text = normalize_accents(text)

    text = normalize_operatorname(text)

    text = normalize_arrows(text)

    text = text.replace("backslash", "\\")

    # OCR often removes the backslash from arrow commands

    arrow_words = {
        "xrightarrow": "→",
        "xleftarrow": "←",
        "rightarrow": "→",
        "leftarrow": "←",
        "Rightarrow": "⇒",
        "Leftarrow": "⇐",
        "leftrightarrow": "↔",
        "Leftrightarrow": "⇔",
        "uparrow": "↑",
        "downarrow": "↓",
    }

    for old, new in arrow_words.items():
        text = text.replace(old, new)

    text = normalize_floorceil(text)

    # remove begin/end LaTeX environments
    environment_words = [
        "beginalign*",
        "endalign*",
        "beginalign",
        "endalign",
        "beginarray",
        "endarray",
        "beginmatrix",
        "endmatrix",
        "begincases",
        "endcases",
    ]

    for word in environment_words:
        text = text.replace(word, "")

    # normalize mathrm
    text = text.replace("mathrmd", "d")
    text = text.replace("\\mathrm{d}", "d")
    text = text.replace("\\mathrm", "")

    text = normalize_frac(text)
    
    text = apply_replacements(text, REPLACEMENT_RULES)

    text = apply_replacements(text, GREEK_RULES)

    # normalize common LaTeX commands to Unicode
    text = re.sub(r'\bsum\b', '∑', text)
    text = re.sub(r'\bpartial\b', '∂', text)
    text = re.sub(r'\bnabla\b', '∇', text)
    text = re.sub(r'\binfty\b', '∞', text)

    text = re.sub(r'\bleft\b', '', text)
    text = re.sub(r'\bright\b', '', text)

    text = re.sub(r'\blangle\b', '⟨', text)
    text = re.sub(r'\brangle\b', '⟩', text)

    text = re.sub(r'\bprime\b', "′", text)
    text = re.sub(r"\s+'\b", "'", text)

    text = re.sub(r'\bmathbf\b', '', text)
    text = re.sub(r'\s+', ' ', text)

    
    # remove xlongequal wrapper
    text = re.sub(r'xlongequal\s*\([^)]*\)', '', text)
    text = re.sub(r'\bxlongequal\b', '=', text) 
    
    # remove LaTeX delimiter sizing commands
    text = re.sub(
        r'\\(?:big|Big|bigl|bigr|Bigl|Bigr|bigg|Bigg|biggl|biggr|Biggl|Biggr)\b',
        '',
        text,
    )

    # remove spacing commands
    text = re.sub(
        r'\\(?:quad|qquad|,|;|!|:|enspace|thinspace|medspace|thickspace)\b',
        '',
        text,
    )

    # remove phantom commands
    text = re.sub(
        r'\\(?:phantom|hphantom|vphantom)\{[^{}]*\}',
        '',
        text,
    )


    # remove displaystyle
    text = re.sub(r'\\displaystyle\b', '', text)

    # normalize ldots
    text = re.sub(r'\\ldots\b', '...', text)

    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'\s+', ' ', text)

    # remove braces around simple subscripts/superscripts
    text = re.sub(r'_\{([^{}]+)\}', r'_\1', text)
    text = re.sub(r'\^\{([^{}]+)\}', r'^\1', text)

    # remove remaining simple braces
    text = re.sub(r'\{([^{}]+)\}', r'\1', text)

    # remove remaining backslashes
    text = text.replace("\\", "")

    # normalize spaces
    text = re.sub(r'\s+', ' ', text)

    
    return text.strip()

def merge_multiline_equations(equations):

    if not equations:
        return equations

    merged = [equations[0]]

    continuation_re = re.compile(
        r"""^(
            [+\-] |
            [=] |
            [)\]}] |
            [&] |
            ε|ζ|π|
            o\(|O\(
        )""",
        re.X,
    )

    for eq in equations[1:]:

        prev = merged[-1]

        prev_pdf, prev_page, _, prev_text = prev
        pdf, page, _, text = eq

        # never merge across PDFs/pages
        if pdf != prev_pdf or page != prev_page:
            merged.append(eq)
            continue

        if continuation_re.match(text):

            merged[-1] = (
                prev_pdf,
                prev_page,
                "equation",
                prev_text + " " + text,
            )

        else:
            merged.append(eq)

    return merged

def extract_equation(pdf):
    
    pages = load_pages(pdf)
       
    if not pages:
        return False, []

    equations = []

      
    for page in pages:

        page_index = page["page_index"]

        for block in page_blocks(page):

            if block.get("block_label") != "display_formula":
                continue

            text = normalize_equation(
                block.get("block_content", "")
            )

            if not text:
                continue

            equations.append(
                (
                    pdf.pdf_name,
                    page_index,
                    "equation",
                    text,
                )
            )

    equations = merge_multiline_equations(equations)

    if not equations:
        return False, []

    return True, equations

def extract_reference(pdf):

    pages = load_pages(pdf)

    blocks = collect_reference_blocks(pages)

    # Find first page containing reference_content
    first_ref_page = None

    for page in pages:
        if any(
            b.get("block_label") == "reference_content"
            for b in page_blocks(page)
        ):

            first_ref_page = page["page_index"]
            break

    if first_ref_page is not None:
        for page in pages:
            page_index = page["page_index"]

            # Only inspect at most 8 pages before the bibliography
            if page_index >= first_ref_page or first_ref_page - page_index > 8:
                continue

            for block in page_blocks(page):
                if block.get("block_label") != "text":
                    continue

                text = block_text(block)

                if text and re.match(r"^\[\d+\]", text):
                    blocks.append((page_index, text))

    blocks.sort(key=lambda x: x[0])

    if not blocks:
        return False, []

    results = []

    i = 0

    while i < len(blocks):

        page, text = blocks[i]

        while (
            text.rstrip().endswith("-")
            and i + 1 < len(blocks)
        ):
            next_page, next_text = blocks[i + 1]

            stripped = next_text.lstrip()

            if stripped.startswith("["):
                break

            text = text.rstrip()[:-1] + stripped
            i += 1

        results.append(
            (
                pdf.pdf_name,
                page,
                "reference",
                text,
            )
        )

        i += 1

    return True, results
    
def extract_section(pdf):

    pages = load_pages(pdf)

    if not pages:
        return False, []

    results = []

    for page in pages:

        page_index = page["page_index"]

        for block in page_blocks(page):

            if block.get("block_label") != "paragraph_title":
                continue

            text = block_text(block)

            if not text:
                continue

            if text.lower() == "contents":
                continue

            text = re.sub(
                r'^\s*(?:\d+(?:\.\d+)*|[A-Z]|[ivxlcdmIVXLCDM]+)[\.\)]?\s+',
                '',
                text,
            )

            text = normalize_string(text)

            
            if not text:
                continue

            results.append(
                (
                    pdf.pdf_name,
                    page_index,
                    "section",
                    text,
                )
            )

    if not results:
        return False, []

    return True, results

def extract_footer(pdf):

    pages = load_pages(pdf)

    if not pages:
        return False, []

    footer_labels = {
        "footnote",
        "vision_footnote",
        "footer",
    }

    footers = []
    seen = set()

    for page in pages:

        page_index = page["page_index"]

        for block in page_blocks(page):

            if block.get("block_label") not in footer_labels:
                continue

            text = block_text(block)

            if not text:
                continue


            if text in seen:
                continue

            seen.add(text)

            footers.append(
                (
                    pdf.pdf_name,
                    page_index,
                    "footer",
                    text
                )
            )

    if not footers:
        return False, []

    return True, footers



def extract_header(pdf):
    
    pages = load_pages(pdf)
    
    if not pages:
        
        return False, []

    headers = []
    

    for page in pages:
        page_index = page.get("page_index", "?")

        blocks = page_blocks(page)

        page_headers = []

        for block in blocks:

            label = block.get("block_label")
            
            text = block_text(block)

            
            if label != "header":
                continue

            if not text:
                
                continue


            # Skip obvious template/running headers
            skip_patterns = [
                "journal of l",
                "aps/123-qed",
                "chapter ",
                "appendix ",
                "references",
                "figures & tables",
                "the following material supplements the paper",
                "professorship program.",
            ]

            lower_text = text.lower()

            if any(p in lower_text for p in skip_patterns):
                
                continue

            page_headers.append(text)

        if page_headers:
            merged = " ".join(" ".join(page_headers).split())

            headers.append(
                (
                    pdf.pdf_name,
                    page_index,
                    "header",
                    merged,
                )
            )

    
    if not headers:
        return False, []

    return True, headers

def normalize_table_text(text: str) -> str:
    """
    Normalize PaddleOCR-VL table text to better match eSciBench GT.
    """

    if not text:
        return text

    text = html.unescape(text)

    replacements = {
        r"\pm": "±",
        r"\times": "×",
        r"\cdot": "·",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\neq": "≠",
        r"\approx": "≈",
        r"\propto": "∝",
        r"\infty": "∞",

        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\epsilon": "ε",
        r"\varepsilon": "ε",
        r"\theta": "θ",
        r"\lambda": "λ",
        r"\mu": "μ",
        r"\nu": "ν",
        r"\pi": "π",
        r"\rho": "ρ",
        r"\sigma": "σ",
        r"\tau": "τ",
        r"\phi": "φ",
        r"\omega": "ω",

        r"\Gamma": "Γ",
        r"\Delta": "Δ",
        r"\Theta": "Θ",
        r"\Lambda": "Λ",
        r"\Pi": "Π",
        r"\Sigma": "Σ",
        r"\Phi": "Φ",
        r"\Omega": "Ω",
    }

    #
    # remove inline math delimiters
    #
    text = text.replace("$", "")

    #
    # replace common latex commands
    #
    for latex, uni in replacements.items():
        text = text.replace(latex, uni)

    #
    # remove escaped braces
    #
    text = text.replace(r"\{", "{")
    text = text.replace(r"\}", "}")

    #
    # remove escaped underscore
    #
    text = text.replace(r"\_", "_")

    #
    # collapse whitespace
    #
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def merge_multipage_tables(tables):
    if not tables:
        return tables

    merged = []
    i = 0

    while i < len(tables):
        current = tables[i]

        if i + 1 < len(tables):

            current_text = current[3]
            next_text = tables[i + 1][3]

            if "continued on next page" in current_text.lower():

                current_text = re.sub(
                    r"continued on next page",
                    "",
                    current_text,
                    flags=re.I,
                ).strip()

                merged.append(
                    (
                        current[0],
                        current[1],
                        current[2],
                        current_text + " " + next_text,
                    )
                )

                i += 2
                continue

        merged.append(current)
        i += 1

    return merged

def extract_table(pdf):

    pages = load_pages(pdf)

    if not pages:
        return False, []

    tables = []

    for page in pages:

        page_index = page.get("page_index", "?")

        for block in page_blocks(page):

            if block.get("block_label") != "table":
                continue

            html_content = block.get("block_content", "")

            if not html_content:
                continue

            soup = BeautifulSoup(html_content, "html.parser")
            text = soup.get_text(" ", strip=True)
            text = normalize_table_text(text)

            
            if not text:
                continue

            tokens = text.split()

            binary_tokens = sum(
                1
                for t in tokens
                if re.fullmatch(r"[01]{6,}", t)
            )

            # Skip binary lookup tables (not GT tables in eSciBench)
            if binary_tokens >= 5:
                continue

            tables.append(
                (
                    pdf.pdf_name,
                    page_index,
                    "table",
                    text,
                )
            )

    if not tables:
        return False, []

    tables = merge_multipage_tables(tables)

    return True, tables

def extract_raw(base_dir, label, pdf):

    handlers = {
        "title": extract_title,
        "abstract": extract_abstract,
        "author": extract_author,
        "keyword": extract_keyword,
        "pub_date": extract_pub_date,
        "affiliation": extract_affiliation,
        "caption": extract_caption,
        "email": extract_email,
        "equation": extract_equation,
        "section": extract_section,
        "reference": extract_reference,
        "footer": extract_footer,
        "header": extract_header,
        "table": extract_table,
        
    }

    if label not in handlers:
        return False, []

    return handlers[label](pdf)
