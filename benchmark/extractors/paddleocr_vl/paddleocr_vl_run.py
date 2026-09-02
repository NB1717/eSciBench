import json
import re
import subprocess
import unicodedata
from pathlib import Path
from datetime import datetime
from benchmark.normalisation import normalize_string
from bs4 import BeautifulSoup


OUTPUT_ROOT = Path(
    "/scratch/nasimb/escibench_project/paddleocr_vl_workspace/"
    "paddleocr_vl_outputs"
)


def _raw_output_root_for_pdf(pdf):
    pdf_name = Path(pdf.pdf_name).stem

    pdf_dir = OUTPUT_ROOT / pdf_name

    if pdf_dir.is_dir():
        return OUTPUT_ROOT

    return None


def normalize_text(text):
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r'([A-Za-z0-9])_\{([^{}]+)\}', r'\1_\2', text)
    text = text.replace(r"\cdots", "⋯")

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def load_first_page(pdf):
    pdf_name = Path(pdf.pdf_name).stem
    raw_root = _raw_output_root_for_pdf(pdf)

    if raw_root is None:
        return None

    pdf_dir = raw_root / pdf_name
    page_file = pdf_dir / f"{pdf_name}_0_res.json"

    if not page_file.is_file():
        return None

    with open(page_file, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_reference_blocks(pages):

    results = []

    first_ref_page = None


    for page in pages:

        if any(
            b.get("block_label") == "reference_content"
            for b in page.get("parsing_res_list", [])
        ):
            first_ref_page = page["page_index"]
            break

    if first_ref_page is None:
        return results


    for page in pages:

        page_index = page["page_index"]

        for block in page.get("parsing_res_list", []):

            if block.get("block_label") != "reference_content":
                continue

            text = normalize_text(block.get("block_content", ""))

            if text:
                results.append((page_index, text))


    for page in pages:

        page_index = page["page_index"]

        if page_index >= first_ref_page:
            continue

        if first_ref_page - page_index > 8:
            continue

        labels = {
            b.get("block_label")
            for b in page.get("parsing_res_list", [])
        }

        if labels - {"text", "number"}:
            continue

        for block in page.get("parsing_res_list", []):

            if block.get("block_label") != "text":
                continue

            text = normalize_text(block.get("block_content", ""))

            if not text:
                continue

            if re.match(r"^\[\d+\]", text):
                results.append((page_index, text))

    results.sort(key=lambda x: (x[0], x[1]))

    return results

def extract_title(pdf):

    page = load_first_page(pdf)

    if not page:
        return False, []

    blocks = page.get("parsing_res_list", [])

    doc_titles = [
        b for b in blocks
        if b.get("block_label") == "doc_title"
        and normalize_text(b.get("block_content"))
    ]

    doc_titles = sorted(
        doc_titles,
        key=lambda b: (
            b.get("block_order") is None,
            b.get("block_order")
            if b.get("block_order") is not None
            else float("inf"),
        ),
    )

    if doc_titles:
        title = normalize_text(doc_titles[0].get("block_content"))

        return True, [
            (
                pdf.pdf_name,
                page.get("page_index", 0),
                "title",
                title,
            )
        ]

    headers = [
        b for b in blocks
        if b.get("block_label") == "header"
        and normalize_text(b.get("block_content"))
    ]

    if headers:
        title = normalize_text(headers[0].get("block_content"))

        return True, [
            (
                pdf.pdf_name,
                page.get("page_index", 0),
                "title",
                title,
            )
        ]

    ordered_blocks = sorted(
        blocks,
        key=lambda b: (
            b.get("block_order") is None,
            b.get("block_order")
            if b.get("block_order") is not None
            else float("inf"),
        ),
    )

    for block in ordered_blocks:
        text = normalize_text(block.get("block_content"))

        if not text:
            continue

        return True, [
            (
                pdf.pdf_name,
                page.get("page_index", 0),
                "title",
                text,
            )
        ]

    return False, []



EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)

GROUP_EMAIL_RE = re.compile(
    r"[\{\[]([^\{\}\[\]]+)[\}\]]@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})"
)


def load_all_pages(pdf):
    pdf_name = Path(pdf.pdf_name).stem
    raw_root = _raw_output_root_for_pdf(pdf)

    if raw_root is None:
        return []

    pdf_dir = raw_root / pdf_name

    if not pdf_dir.is_dir():
        return []

    page_files = list(pdf_dir.glob(f"{pdf_name}_*_res.json"))

    def page_number(path):
        name = path.name
        prefix = f"{pdf_name}_"
        suffix = "_res.json"
        return int(name[len(prefix):-len(suffix)])

    page_files.sort(key=page_number)

    pages = []

    for page_file in page_files:
        with open(page_file, "r", encoding="utf-8") as f:
            pages.append(json.load(f))

    return pages




CAPTION_MARKER_RE = re.compile(
    r"(?ix)^\s*"
    r"(?:figure|fig\.?|tableau|table|sch[ée]ma)"
    r"\s*\d+[A-Za-z]?"
    r"\s*[\.:;\-–—]?\s*"
)

CAPTION_MARKER_ONLY_RE = re.compile(
    r"(?ix)^\s*"
    r"(?:figure|fig\.?|tableau|table|sch[ée]ma)"
    r"\s*\d+[A-Za-z]?"
    r"\s*[\.:;\-–—]?\s*$"
)

CAPTION_FR_WORDS = {
    "de","du","des","la","le","les","un","une","en","et",
    "dans","sur","pour","par","avec","aux","au","d","l",
    "évolution","effet","masse","eaux","traitées","fonction",
    "schéma","courbe"
}

CAPTION_EN_WORDS = {
    "the","of","and","in","for","to","with","from","on","by",
    "according","effect","evolution","water","treated","mass",
    "diagram","curve","plot","study","results"
}


def caption_language_score(text, lexicon):
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text.lower())
    return sum(w in lexicon for w in words)


def caption_split_one_line_bilingual(text):
    if "\n" in text:
        return [text]

    boundaries = [
        m.end()
        for m in re.finditer(r"\.\s+(?=[A-ZÀ-Ý])", text)
    ]

    best = None

    for pos in boundaries:
        left = text[:pos].strip()
        right = text[pos:].strip()

        if len(left.split()) < 4 or len(right.split()) < 4:
            continue

        fr_left = caption_language_score(
            left,
            CAPTION_FR_WORDS,
        )
        en_left = caption_language_score(
            left,
            CAPTION_EN_WORDS,
        )
        fr_right = caption_language_score(
            right,
            CAPTION_FR_WORDS,
        )
        en_right = caption_language_score(
            right,
            CAPTION_EN_WORDS,
        )

        if (
            fr_left >= 2
            and fr_left > en_left
            and en_right >= 2
            and en_right > fr_right
        ):
            strength = (
                (fr_left - en_left)
                + (en_right - fr_right)
            )

            if best is None or strength > best[0]:
                best = (strength, left, right)

    if best:
        return [best[1], best[2]]

    return [text]


def caption_clean_piece(text):
    text = CAPTION_MARKER_RE.sub(
        "",
        text,
        count=1,
    )

    return normalize_string(text.strip())


def extract_caption(pdf):
    pages = load_all_pages(pdf)

    if not pages:
        return False, []

    out = []
    seen = set()

    def add(page_index, text):
        candidate = caption_clean_piece(text)

        if not candidate or candidate in seen:
            return

        seen.add(candidate)

        out.append(
            (
                pdf.pdf_name,
                page_index,
                "caption",
                candidate,
            )
        )

    for page in pages:
        page_index = page.get("page_index", 0)
        blocks = page.get("parsing_res_list", [])

        for i, block in enumerate(blocks):
            label = block.get("block_label")

            raw = str(
                block.get("block_content", "") or ""
            ).strip()

            if not raw:
                continue

            if (
                label == "figure_title"
                and CAPTION_MARKER_RE.match(raw)
            ):

                if CAPTION_MARKER_ONLY_RE.fullmatch(raw):

                    for j in range(
                        i + 1,
                        min(i + 5, len(blocks)),
                    ):
                        nxt = blocks[j]
                        nxt_label = nxt.get("block_label")

                        if nxt_label in {
                            "image",
                            "chart",
                            "number",
                        }:
                            continue

                        if nxt_label != "figure_title":
                            break

                        nxt_raw = str(
                            nxt.get(
                                "block_content",
                                "",
                            ) or ""
                        ).strip()

                        if (
                            nxt_raw
                            and not CAPTION_MARKER_ONLY_RE.fullmatch(
                                nxt_raw
                            )
                        ):
                            add(page_index, nxt_raw)

                        break

                    continue

                line_parts = [
                    p.strip()
                    for p in re.split(r"\n+", raw)
                    if p.strip()
                ]

                for part in line_parts:
                    cleaned = CAPTION_MARKER_RE.sub(
                        "",
                        part,
                        count=1,
                    ).strip()

                    for piece in caption_split_one_line_bilingual(
                        cleaned
                    ):
                        add(page_index, piece)

            elif (
                label == "header"
                and CAPTION_MARKER_ONLY_RE.fullmatch(raw)
            ):
                for j in range(
                    i + 1,
                    min(i + 5, len(blocks)),
                ):
                    nxt = blocks[j]
                    nxt_label = nxt.get("block_label")

                    if nxt_label in {
                        "image",
                        "chart",
                        "number",
                    }:
                        continue

                    if nxt_label != "figure_title":
                        break

                    nxt_raw = str(
                        nxt.get(
                            "block_content",
                            "",
                        ) or ""
                    ).strip()

                    if nxt_raw:
                        add(page_index, nxt_raw)

                    break

    if not out:
        return False, []

    return True, out


def extract_email(pdf):
    pages = load_all_pages(pdf)

    if not pages:
        return False, []

    results = []
    seen = set()

    for page in pages:
        page_index = page.get("page_index", 0)

        for block in page.get("parsing_res_list", []):
            content = str(block.get("block_content", "") or "")

            for local_part, domain in GROUP_EMAIL_RE.findall(content):
                local_part = re.sub(
                    r"([A-Za-z0-9_.-]+)\s+([A-Za-z0-9_.-]+)",
                    r"\1,\2",
                    local_part,
                )

                for name in local_part.split(","):
                    name = name.strip()

                    if not name:
                        continue

                    email = normalize_text(f"{name}@{domain}")

                    if not email or email in seen:
                        continue

                    seen.add(email)

                    results.append(
                        (
                            pdf.pdf_name,
                            page_index,
                            "email",
                            email,
                        )
                    )

            for match in EMAIL_RE.findall(content):
                email = normalize_text(match)

                if not email or email in seen:
                    continue

                seen.add(email)

                results.append(
                    (
                        pdf.pdf_name,
                        page_index,
                        "email",
                        email,
                    )
                )

    if not results:
        return False, []

    return True, results



KEYWORD_PREFIX_RE = re.compile(
    r"(?i)^\s*"
    r"(?:"
    r"mots?\s*[-‐-‒–—]?\s*cl[ée]s?"
    r"|key\s*words?\s+and\s+phrases?"
    r"|key\s*words?"
    r"|keywords?"
    r"|index\s+terms?"
    r"|palabras\s+clave"
    r")"
    r"\s*(?:[:：.•·\-–—]\s*)?"
)




def _parse_keyword_payload(text):
    text = str(text or "").split("|")[0]

    text = re.split(
        r"(?i)\b(?:"
        r"JEL\s+Classification|"
        r"Mathematics\s+Subject\s+Classification|"
        r"MSC"
        r")\b",
        text,
        maxsplit=1,
    )[0]

    text = re.sub(r"[—–·;]", ",", text)

    out = []

    for raw_keyword in text.split(","):
        value = raw_keyword.strip()
        value = re.sub(r"[.;:•·]+$", "", value).strip()
        value = normalize_text(value)

        if value:
            out.append(value)

    return out


def extract_keyword(pdf):
    pages = load_all_pages(pdf)

    if not pages:
        return False, []

    results = []

    for page in pages:
        page_index = page.get("page_index", 0)
        blocks = page.get("parsing_res_list", [])
        waiting = False

        for block in blocks:
            label = str(block.get("block_label", "") or "")
            content = str(block.get("block_content", "") or "").strip()

            if not content:
                continue

            if label == "paragraph_title":
                low = content.lower()

                if (
                    "keyword" in low
                    or "key word" in low
                    or "index term" in low
                    or "mot clé" in low
                    or "mots clé" in low
                    or "palabras clave" in low
                ):
                    waiting = True
                    continue

            if waiting:
                if label != "text":
                    continue

                for value in _parse_keyword_payload(content):
                    results.append(
                        (pdf.pdf_name, page_index, "keyword", value)
                    )

                waiting = False
                continue

            if label != "text":
                continue

            match = KEYWORD_PREFIX_RE.match(content)

            if not match:
                continue

            payload = content[match.end():].strip()

            for value in _parse_keyword_payload(payload):
                results.append(
                    (pdf.pdf_name, page_index, "keyword", value)
                )

    if not results:
        return False, []

    return True, results



AFF_INST_RE = re.compile(
    r"(?i)\b("
    r"universit(?:y|é|e|ät)|"
    r"école|ecole|school|"
    r"faculté|faculte|faculty|"
    r"département|departement|department|"
    r"institut|institute|"
    r"laboratoire|laboratory|"
    r"centre|center|"
    r"chaire|"
    r"cnrs|umr\b|ehess\b|"
    r"cégep|cegep|college|collège|"
    r"google|deepmind|meta|microsoft|amazon|apple|ibm|"
    r"nvidia|intel|adobe|bytedance|huawei|qualcomm|"
    r"salesforce|oracle|caltech|technion|"
    r"academy|hospital|dipartimento|lab"
    r")"
)

AFF_FIELD_RE = re.compile(
    r"(?i)\b("
    r"engineering|"
    r"computer\s+science|"
    r"physics|"
    r"mathematics"
    r")\b"
)


AFF_ROLE_RE = re.compile(
    r"(?i)\b("
    r"professeur|professeure|professor|"
    r"étudiant|étudiante|etudiant|etudiante|student|"
    r"candidate au doctorat|"
    r"boursier|boursière|boursiere|"
    r"chercheur|chercheure|researcher|"
    r"coordonnateur|coordonnatrice|"
    r"spécialiste|specialiste|"
    r"animatrice|animateur|"
    r"auxiliaire de recherche|"
    r"avocat-conseil|"
    r"forensic auditor"
    r")"
)

AFF_DEGREE_RE = re.compile(
    r"(?i)\b("
    r"ll\.\s*b\.|ll\.\s*m\.|"
    r"m\.\s*st\.|d\.\s*phil\.|"
    r"ph\.?\s*d\.?|"
    r"b\.\s*sc\.?|m\.\s*sc\.?|"
    r"b\.\s*a\.?|m\.\s*a\.?|"
    r"b\.\s*ed\.?|m\.\s*s\.\s*s\."
    r")"
)

AFF_LATEX_MARKER_RE = re.compile(
    r"\$\s*\^\s*\{([^{}]+)\}\s*\$"
)

AFF_UNICODE_MARKERS = {
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
}

AFF_STOP_RE = re.compile(
    r"(?i)\s+(?="
    r"une version antérieure|"
    r"une version anterieure|"
    r"l['’]auteur(?:e)?\b|"
    r"the author\b|"
    r"cet article\b"
    r")"
)


def _ordered_blocks(page):
    return sorted(
        page.get("parsing_res_list", []),
        key=lambda b: (
            b.get("block_order") is None,
            b.get("block_order")
            if b.get("block_order") is not None
            else float("inf"),
        ),
    )


def _strip_email(text):
    return EMAIL_RE.sub(" ", str(text or ""))


def _clean_affiliation(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = _strip_email(text)

    text = AFF_STOP_RE.split(text, maxsplit=1)[0]

    text = AFF_LATEX_MARKER_RE.sub(" ", text)

    for char in AFF_UNICODE_MARKERS:
        text = text.replace(char, " ")

    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n,;:.–—-")

    return normalize_text(text)


def _first_affiliation_cue(text):
    matches = []

    m = AFF_ROLE_RE.search(text)
    if m:
        matches.append(m)

    m = AFF_INST_RE.search(text)
    if m:
        matches.append(m)

    if not matches:
        return None

    return min(matches, key=lambda x: x.start())


def _strip_author_prefix(text):
    text = str(text or "")
    cue = _first_affiliation_cue(text)

    if not cue:
        return text

    prefix = text[:cue.start()].strip(" ,;:-")
    words = prefix.split()

    if (
        prefix
        and len(words) <= 8
        and not AFF_INST_RE.search(prefix)
        and not AFF_ROLE_RE.search(prefix)
    ):
        return text[cue.start():]

    return text


def _author_marker_counts(text):
    counts = {}

    for raw in AFF_LATEX_MARKER_RE.findall(str(text or "")):
        marker = re.sub(r"[^0-9]", "", raw)

        if marker:
            counts[marker] = counts.get(marker, 0) + 1

    for char, marker in AFF_UNICODE_MARKERS.items():
        n = str(text or "").count(char)

        if n:
            counts[marker] = counts.get(marker, 0) + n

    return counts


def _rough_author_count(text):
    text = str(text or "")

    text = AFF_LATEX_MARKER_RE.sub("", text)

    for char in AFF_UNICODE_MARKERS:
        text = text.replace(char, "")

    text = text.replace("*", " ")

    parts = [
        p.strip()
        for p in text.split(",")
        if p.strip()
    ]

    return len(parts)


def _split_role_segments(text):
    text = str(text or "")

    pieces = []

    for semicolon_piece in text.split(";"):
        semicolon_piece = semicolon_piece.strip()

        if not semicolon_piece:
            continue

        starts = [
            m.start()
            for m in AFF_ROLE_RE.finditer(semicolon_piece)
        ]

        if len(starts) <= 1:
            pieces.append(semicolon_piece)
            continue

        boundaries = [0]

        for pos in starts[1:]:
            boundaries.append(pos)

        boundaries.append(len(semicolon_piece))

        for a, b in zip(boundaries, boundaries[1:]):
            part = semicolon_piece[a:b].strip()

            if part:
                pieces.append(part)

    return pieces


def _split_numbered_affiliations(text):
    text = str(text or "")

    matches = list(AFF_LATEX_MARKER_RE.finditer(text))

    if not matches:
        return []

    result = []

    for i, m in enumerate(matches):
        marker_raw = m.group(1)
        marker = re.sub(r"[^0-9]", "", marker_raw)

        if not marker:
            continue

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        value = text[start:end].strip()

        if value:
            result.append((marker, value))

    return result


def _split_repeated_university(text):
    pattern = re.compile(r"(?i)\b(?:université|university|universitat|universität)\b")
    matches = list(pattern.finditer(text))

    if len(matches) < 2:
        return [text]

    parts = []
    start = 0

    for m in matches[1:]:
        part = text[start:m.start()].strip(" ,;")

        if part:
            parts.append(part)

        start = m.start()

    last = text[start:].strip(" ,;")

    if last:
        parts.append(last)

    return parts


def _is_affiliation_candidate(text):
    text = str(text or "")

    field_candidate = (
        len(text.split()) <= 18
        and AFF_FIELD_RE.search(text)
    )

    return bool(
        AFF_INST_RE.search(text)
        or AFF_ROLE_RE.search(text)
        or AFF_DEGREE_RE.search(text)
        or field_candidate
    )



def _front_matter_blocks(page):
    result = []

    for block in _ordered_blocks(page):
        label = block.get("block_label")
        content = str(block.get("block_content", "") or "").strip()

        if not content:
            continue

        if label == "abstract":
            break

        if (
            label == "paragraph_title"
            and re.search(
                r"(?i)\b(résumé|resume|abstract)\b",
                content,
            )
        ):
            break

        if (
            label == "text"
            and len(content) >= 260
            and len(result) >= 2
            and not _is_affiliation_candidate(content)
        ):
            break

        if (
            label == "text"
            and len(content) >= 420
            and len(result) >= 1
            and len(content.split()) > 60
        ):
            break

        result.append(block)

    return result


AFF_METADATA_ITEM_START_RE = re.compile(
    r"(?i)"
    r"(?:"
    r"ph\.?\s*d\.?|"
    r"ll\.\s*b\.|ll\.\s*m\.|"
    r"m\.\s*st\.|d\.\s*phil\.|"
    r"professeur(?:e)?(?:\s+\w+){0,2}|"
    r"titulaire\b|"
    r"chercheur(?:e)?\b|"
    r"candidate?\s+au\s+doctorat|"
    r"boursi(?:er|ère|ere)\b|"
    r"baccalauréat\b|baccalaureat\b|"
    r"maîtrise\b|maitrise\b|"
    r"auxiliaire\s+de\s+recherche|"
    r"coordonnat(?:eur|rice)\b|"
    r"spécialiste\b|specialiste\b|"
    r"animat(?:eur|rice)\b|"
    r"avocat-conseil\b|"
    r"forensic\s+auditor\b"
    r")"
)


def _clean_footnote_prefix(text):
    text = str(text or "").strip()

    text = re.sub(
        r"^\s*\$\s*\^\s*\{[^{}]*\}\s*\$\s*",
        "",
        text,
    )

    text = re.sub(r"^\s*\*+\s*", "", text)
    text = re.sub(r"^\s*\d+\s*[.)]\s*", "", text)

    return text.strip()


def _split_metadata_affiliation_items(text):
    text = str(text or "").strip()

    if not text:
        return []

    matches = list(AFF_METADATA_ITEM_START_RE.finditer(text))

    if not matches:
        return [text] if _is_affiliation_candidate(text) else []

    starts = []

    for m in matches:
        pos = m.start()

        matched = m.group(0).lower()

        if (
            re.match(r"ma[iî]trise\b", matched)
            and re.search(r"\bet\s*$", text[:pos], re.IGNORECASE)
        ):
            continue

        if not starts or pos != starts[-1]:
            starts.append(pos)

    pieces = []

    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        piece = text[start:end].strip(" ,;")

        if piece:
            pieces.append(piece)

    return pieces


def _looks_like_author_plus_affiliation_block(text):
    lines = [
        line.strip()
        for line in str(text or "").splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        return False

    first_line = lines[0]
    remaining = "\n".join(lines[1:])

    return (
        not _is_affiliation_candidate(first_line)
        and _is_affiliation_candidate(remaining)
    )


def _extract_page0_affiliations(pdf, page):
    results = []
    blocks = _front_matter_blocks(page)

    previous_text_block = None

    for block in blocks:
        label = block.get("block_label")
        content = str(block.get("block_content", "") or "").strip()

        if not content:
            continue

        if label != "text":
            continue

        if not _is_affiliation_candidate(content):
            previous_text_block = content
            continue

        if previous_text_block:
            prev_value = _clean_affiliation(
                _strip_author_prefix(previous_text_block)
            )
            curr_value = _clean_affiliation(
                _strip_author_prefix(content)
            )

            merge_value = None

            unit_then_university = (
                prev_value
                and curr_value
                and re.match(
                    r"(?i)^(?:"
                    r"département|departement|department|"
                    r"laboratoire|laboratory|"
                    r"faculté|faculte|faculty|"
                    r"centre|center"
                    r")\b",
                    prev_value,
                )
                and re.match(
                    r"(?i)^universit(?:é|e|y|ät)\b",
                    curr_value,
                )
            )

            role_then_institution = (
                prev_value
                and curr_value
                and len(prev_value.split()) <= 8
                and AFF_ROLE_RE.search(prev_value)
                and not AFF_INST_RE.search(prev_value)
                and AFF_INST_RE.search(curr_value)
            )

            general_adjacent_affiliation = (
                prev_value
                and curr_value
                and len(prev_value.split()) <= 12
                and len(curr_value.split()) <= 18
                and _is_affiliation_candidate(prev_value)
                and _is_affiliation_candidate(curr_value)
                and (
                    AFF_INST_RE.search(prev_value)
                    or AFF_INST_RE.search(curr_value)
                )
                and not EMAIL_RE.search(prev_value)
                and not EMAIL_RE.search(curr_value)
                and _rough_author_count(prev_value) < 2
                and not _looks_like_author_plus_affiliation_block(
                    previous_text_block
                )
                and not _looks_like_author_plus_affiliation_block(
                    content
                )
            )

            if (
                unit_then_university
                or role_then_institution
                or general_adjacent_affiliation
            ):
                merge_value = _clean_affiliation(
                    f"{prev_value}, {curr_value}"
                )

            if merge_value:
                if results and results[-1][3] == prev_value:
                    results.pop()

                results.append(
                    (
                        pdf.pdf_name,
                        page.get("page_index", 0),
                        "affiliation",
                        merge_value,
                    )
                )

                previous_text_block = content
                continue

        numbered = _split_numbered_affiliations(content)

        if numbered:
            counts = _author_marker_counts(previous_text_block or "")

            author_count = _rough_author_count(
                previous_text_block or ""
            )

            for marker, raw_value in numbered:
                value = _clean_affiliation(raw_value)

                if not value:
                    continue

                repeat = counts.get(marker, 0)

                if (
                    repeat == 0
                    and len(numbered) == 1
                    and 2 <= author_count <= 8
                    and not _is_affiliation_candidate(
                        previous_text_block or ""
                    )
                ):
                    repeat = author_count

                if repeat <= 0:
                    repeat = 1

                for _ in range(repeat):
                    results.append(
                        (
                            pdf.pdf_name,
                            page.get("page_index", 0),
                            "affiliation",
                            value,
                        )
                    )

            previous_text_block = content
            continue

        marker_counts = _author_marker_counts(
            previous_text_block or ""
        )
        unique_markers = sorted(marker_counts)

        raw_parts = [content]

        if len(unique_markers) > 1:
            raw_parts = _split_repeated_university(content)

        final_parts = []

        for raw_part in raw_parts:
            final_parts.extend(
                _split_role_segments(raw_part)
            )

        cleaned_parts = []

        for raw_part in final_parts:
            raw_part = _strip_author_prefix(raw_part)
            value = _clean_affiliation(raw_part)

            if value and _is_affiliation_candidate(value):
                cleaned_parts.append(value)

        if (
            len(cleaned_parts) == 1
            and previous_text_block
            and not marker_counts
        ):
            author_count = _rough_author_count(
                previous_text_block
            )

            if (
                2 <= author_count <= 8
                and not _is_affiliation_candidate(
                    previous_text_block
                )
            ):
                for _ in range(author_count):
                    results.append(
                        (
                            pdf.pdf_name,
                            page.get("page_index", 0),
                            "affiliation",
                            cleaned_parts[0],
                        )
                    )

                previous_text_block = content
                continue

        if (
            cleaned_parts
            and marker_counts
            and len(cleaned_parts) == len(unique_markers)
        ):
            for value, marker in zip(
                cleaned_parts,
                unique_markers,
            ):
                for _ in range(
                    marker_counts.get(marker, 1)
                ):
                    results.append(
                        (
                            pdf.pdf_name,
                            page.get("page_index", 0),
                            "affiliation",
                            value,
                        )
                    )

            previous_text_block = content
            continue

        for value in cleaned_parts:
            results.append(
                (
                    pdf.pdf_name,
                    page.get("page_index", 0),
                    "affiliation",
                    value,
                )
            )

        previous_text_block = content

    for block in _ordered_blocks(page):
        if block.get("block_label") != "footnote":
            continue

        content = str(
            block.get("block_content", "") or ""
        ).strip()

        if not content:
            continue

        content = _clean_footnote_prefix(content)

        content = AFF_STOP_RE.split(content, maxsplit=1)[0].strip()

        if not content:
            continue

        beginning = content[:140]

        if not (
            AFF_DEGREE_RE.match(beginning)
            or AFF_ROLE_RE.match(beginning)
            or AFF_INST_RE.match(beginning)
        ):
            continue

        pieces = [
            p.strip()
            for p in content.split(";")
            if p.strip()
        ]

        for piece in pieces:
            piece = _clean_footnote_prefix(piece)
            piece = _strip_author_prefix(piece)
            value = _clean_affiliation(piece)

            if not value:
                continue

            if not _is_affiliation_candidate(value):
                continue

            results.append(
                (
                    pdf.pdf_name,
                    page.get("page_index", 0),
                    "affiliation",
                    value,
                )
            )

    return results


def _extract_late_metadata(pdf, page):
    results = []
    page_index = page.get("page_index", 0)

    if page_index == 0:
        return results

    blocks = _ordered_blocks(page)

    for block in blocks:
        if block.get("block_label") != "text":
            continue

        content = str(
            block.get("block_content", "") or ""
        ).strip()

        if not content or len(content) > 220:
            continue

        if not AFF_ROLE_RE.search(content):
            continue

        if not AFF_INST_RE.search(content):
            continue

        if re.search(r"(?i)https?://|www\.", content):
            continue

        if (
            len(re.findall(r"[.!?]", content)) >= 2
            and len(content.split()) > 22
        ):
            continue

        value = _clean_affiliation(
            _strip_author_prefix(content)
        )

        if value:
            results.append(
                (
                    pdf.pdf_name,
                    page_index,
                    "affiliation",
                    value,
                )
            )

    for block in blocks:
        if block.get("block_label") != "text":
            continue

        content = str(
            block.get("block_content", "") or ""
        ).strip()

        if not content or len(content) > 320:
            continue

        if not EMAIL_RE.search(content):
            continue

        if not AFF_INST_RE.search(content):
            continue

        value = _clean_affiliation(
            _strip_author_prefix(content)
        )

        if value and _is_affiliation_candidate(value):
            results.append(
                (
                    pdf.pdf_name,
                    page_index,
                    "affiliation",
                    value,
                )
            )

    for email_i, block in enumerate(blocks):
        email_content = str(
            block.get("block_content", "") or ""
        )

        if not EMAIL_RE.search(email_content):
            continue

        cluster = []
        j = email_i - 1

        while j >= 0 and len(cluster) < 9:
            prev = blocks[j]

            if prev.get("block_label") != "text":
                break

            content = str(
                prev.get("block_content", "") or ""
            ).strip()

            if not content:
                break

            if len(content) > 220:
                break

            cluster.append(content)
            j -= 1

        cluster.reverse()

        if len(cluster) < 2:
            continue

        cue_indices = [
            i
            for i, value in enumerate(cluster)
            if AFF_INST_RE.search(value)
            or AFF_ROLE_RE.search(value)
        ]

        if not cue_indices:
            continue

        metadata = cluster[cue_indices[0]:]

        if len(metadata) < 2:
            continue

        if not any(
            AFF_INST_RE.search(value)
            for value in metadata
        ):
            continue

        value = _clean_affiliation(
            ", ".join(metadata)
        )

        if value:
            results.append(
                (
                    pdf.pdf_name,
                    page_index,
                    "affiliation",
                    value,
                )
            )

    reference_heading_i = None

    for i, block in enumerate(blocks):
        if block.get("block_label") != "paragraph_title":
            continue

        heading = str(
            block.get("block_content", "") or ""
        ).strip()

        if re.fullmatch(
            r"(?i)\s*(?:"
            r"références|references|"
            r"bibliographie|bibliography"
            r")\s*",
            heading,
        ):
            reference_heading_i = i
            break

    if reference_heading_i is not None:
        start = max(0, reference_heading_i - 8)
        region = blocks[start:reference_heading_i]

        metadata_blocks = []

        for block in region:
            if block.get("block_label") not in {
                "text",
                "reference_content",
            }:
                continue

            content = str(
                block.get("block_content", "") or ""
            ).strip()

            if not content or len(content) > 700:
                continue

            if (
                AFF_METADATA_ITEM_START_RE.search(content)
                or AFF_ROLE_RE.search(content)
                or AFF_DEGREE_RE.search(content)
            ):
                metadata_blocks.append(content)

        if len(metadata_blocks) >= 2:
            for content in metadata_blocks:
                for piece in _split_metadata_affiliation_items(
                    content
                ):
                    value = _clean_affiliation(piece)

                    if not value:
                        continue

                    if not (
                        _is_affiliation_candidate(value)
                        or AFF_METADATA_ITEM_START_RE.search(value)
                    ):
                        continue

                    results.append(
                        (
                            pdf.pdf_name,
                            page_index,
                            "affiliation",
                            value,
                        )
                    )

    return results

def extract_affiliation(pdf):
    pages = load_all_pages(pdf)

    if not pages:
        return False, []

    results = []
    page0_values = set()

    for page in pages:
        page_index = page.get("page_index", 0)

        if page_index == 0:
            page0_results = _extract_page0_affiliations(
                pdf,
                page,
            )

            results.extend(page0_results)

            page0_values.update(
                row[3]
                for row in page0_results
            )

        else:
            late_results = _extract_late_metadata(
                pdf,
                page,
            )

            for row in late_results:
                if row[3] in page0_values:
                    continue

                results.append(row)

    if not results:
        return False, []

    return True, results




def _clean_author_superscripts(text):
    text = str(text or "")

    text = re.sub(
        r"\$\s*\^\{[^}]*\}\s*\$",
        "",
        text,
    )

    text = re.sub(
        r"[¹²³⁴⁵⁶⁷⁸⁹⁰*]+",
        "",
        text,
    )

    return text


def _author_name_like(text):
    text = str(text or "").strip()

    if not text:
        return False

    if len(text) > 170:
        return False

    if re.search(r"[@!?;:]", text):
        return False

    if re.search(r"\d", text):
        return False

    words = re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿŒœ'’.-]+",
        text,
    )

    if len(words) < 2 or len(words) > 8:
        return False

    letters = re.sub(
        r"[^A-Za-zÀ-ÖØ-öø-ÿŒœ]",
        "",
        text,
    )

    return len(letters) >= 4


def _clean_single_author(value):
    value = str(value or "").strip()

    value = _clean_author_superscripts(value)

    value = re.sub(
        r"(?i)^\s*(?:par|by)\s+",
        "",
        value,
    )

    value = re.sub(
        r"(?i)\s*,?\s*"
        r"(?:ph\.?\s*d\.?|t\.?\s*s\.?)"
        r"(?:\s+[A-Z.]{1,8})*\s*$",
        "",
        value,
    )

    value = re.sub(r"\s+", " ", value).strip(" ,;")

    m = re.fullmatch(
        r"\s*([^,]+),\s*([^,]+)\s*",
        value,
    )

    if m:
        family = m.group(1).strip()
        given = m.group(2).strip()

        if (
            _author_name_like(family + " " + given)
            and family.upper() == family
        ):
            value = given + " " + family

    value = re.sub(r"\s+", " ", value).strip()

    return value.lower()


def _split_author_line(line):
    line = str(line or "").strip()

    if not line:
        return []

    surname_probe = _clean_author_superscripts(line)

    surname_probe = re.sub(
        r"(?i)\s*,?\s*"
        r"(?:ph\.?\s*d\.?|t\.?\s*s\.?)"
        r"(?:\s+[A-Z.]{1,8})*\s*$",
        "",
        surname_probe,
    )

    surname_probe = re.sub(
        r"\s+",
        " ",
        surname_probe,
    ).strip(" ,;")

    if surname_probe.count(",") == 1:
        family, given = [
            x.strip()
            for x in surname_probe.split(",", 1)
        ]

        if (
            family
            and given
            and family.upper() == family
            and _author_name_like(family + " " + given)
        ):
            value = _clean_single_author(surname_probe)

            if value and _author_name_like(value):
                return [value]

    line = re.sub(r"<[^>]+>", " ", line)

    line = re.sub(
        r"\$\s*\{?\s*\}\?\s*\^\{[^}]*\}\s*\$",
        ", ",
        line,
    )
    line = re.sub(
        r"\$\s*\^\{[^}]*\}\s*\$",
        ", ",
        line,
    )
    line = re.sub(
        r"\^\{[^}]*\}",
        ", ",
        line,
    )

    line = line.replace("$", " ")

    line = re.sub(
        r"\s*[*†‡§¶]+\s*",
        ", ",
        line,
    )

    line = re.sub(
        r"(?i)^\s*(?:par|by)\s+",
        "",
        line,
    ).strip()

    line = re.sub(r"\s+", " ", line).strip(" ,;")

    conjunction_parts = [
        part.strip(" ,;")
        for part in re.split(
            r"\s+(?:and|et|&)\s+",
            line,
            flags=re.IGNORECASE,
        )
        if part.strip(" ,;")
    ]

    if len(conjunction_parts) > 1:
        authors = []

        for part in conjunction_parts:
            for value in _split_author_line(part):
                if value and value not in authors:
                    authors.append(value)

        return authors

    raw_parts = [
        p.strip(" ,;")
        for p in line.split(",")
        if p.strip(" ,;")
    ]

    if len(raw_parts) == 2:
        family, given = raw_parts

        combined = family + ", " + given
        value = _clean_single_author(combined)

        if value and _author_name_like(value):
            return [value]

    authors = []

    for part in raw_parts if raw_parts else [line]:
        part = part.strip()

        if not part:
            continue

        words = part.split()

        initials = sum(
            1
            for w in words
            if re.fullmatch(r"[A-Z]\.", w)
        )

        particles = {
            "van", "von", "de", "del",
            "der", "di", "da",
        }

        compact_parts = [part]

        if (
            len(words) >= 4
            and len(words) % 2 == 0
            and initials < 2
            and part != part.upper()
            and not any(
                w.lower() in particles
                for w in words
            )
            and all(
                len(w) > 1
                and not (
                    len(w) == 2
                    and w.endswith(".")
                )
                for w in words
            )
        ):
            compact_parts = [
                words[i] + " " + words[i + 1]
                for i in range(0, len(words), 2)
            ]

        for candidate in compact_parts:
            value = _clean_single_author(candidate)

            if not value:
                continue

            if not _author_name_like(value):
                continue

            if value not in authors:
                authors.append(value)

    return authors



def _author_first_line(content):
    lines = [
        line.strip()
        for line in str(content or "").splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    return lines[0]


def extract_author(pdf):
    pages = load_all_pages(pdf)

    if not pages:
        return False, []

    page0 = None

    for page in pages:
        if page.get("page_index", 0) == 0:
            page0 = page
            break

    if page0 is None:
        return False, []

    blocks = _front_matter_blocks(page0)

    results = []
    author_started = False

    for i, block in enumerate(blocks):
        label = block.get("block_label")

        if label not in {"text", "paragraph_title"}:
            continue

        content = str(
            block.get("block_content", "") or ""
        ).strip()

        if not content:
            continue

        first_line = _author_first_line(content)

        if not first_line:
            continue

        has_metadata_after_name = (
            "\n" in content
            and (
                AFF_INST_RE.search(content)
                or AFF_ROLE_RE.search(content)
                or EMAIL_RE.search(content)
            )
        )

        candidate_line = (
            first_line
            if has_metadata_after_name
            else content
        )

        candidate_line = candidate_line.strip()

        if (
            AFF_INST_RE.search(candidate_line)
            or AFF_ROLE_RE.search(candidate_line)
            or EMAIL_RE.search(candidate_line)
        ):
            continue

        prefixed_author = bool(
            re.match(
                r"(?i)^\s*(?:par|by)\s+",
                candidate_line,
            )
        )

        has_author_marker = bool(
            re.search(
                r"\$\s*\^\{[^}]*\}\s*\$|"
                r"[¹²³⁴⁵⁶⁷⁸⁹⁰]",
                candidate_line,
            )
        )

        simple_name = _author_name_like(
            _clean_author_superscripts(
                candidate_line
            )
        )

        candidate = (
            prefixed_author
            or has_metadata_after_name
            or (
                has_author_marker
                and len(candidate_line) <= 500
            )
            or simple_name
        )

        if not candidate:
            if author_started:
                if (
                    label == "text"
                    and not AFF_INST_RE.search(content)
                    and not AFF_ROLE_RE.search(content)
                    and not EMAIL_RE.search(content)
                ):
                    break

            continue

        authors = _split_author_line(
            candidate_line
        )

        if not authors:
            continue

        if not author_started:
            if not (
                has_metadata_after_name
                or prefixed_author
                or has_author_marker
                or (
                    simple_name
                    and len(
                        candidate_line.split()
                    ) <= 8
                )
            ):
                continue

        for value in authors:
            if not value:
                continue

            results.append(
                (
                    pdf.pdf_name,
                    0,
                    "author",
                    value,
                )
            )

        author_started = True

    if not results:
        return False, []

    return True, results


SECTION_NON_SECTION_RE = re.compile(
    r"(?ix)^\s*(?:"
    r"(?:résumé|resume|abstract|resumen)"
    r"(?:\s*(?:\||[-–—])\s*(?:résumé|resume|abstract|resumen))*"
    r"|références?"
    r"|references?"
    r"|références?\s+bibliographiques?"
    r"|references?\s+bibliographiques?"
    r"|bibliographie"
    r"|bibliography"
    r"|notes?"
    r"|remerciements?"
    r"|acknowledgements?"
    r"|acknowledgments?"
    r"|sources?\s+(?:consultées|électroniques)"
    r")\s*:?\s*$"
)

SECTION_RUNNING_HEADER_RE = re.compile(
    r"(?i)\b(?:printemps|été|ete|automne|hiver)\s+\d{4}\b"
)


SECTION_BYLINE_RE = re.compile(
    r"(?i)^\s*par\s+.+$"
)


def extract_reference(pdf):

    pages = load_all_pages(pdf)

    blocks = collect_reference_blocks(pages)

    first_ref_page = None

    for page in pages:
        if any(
            b.get("block_label") == "reference_content"
            for b in page.get("parsing_res_list", [])
        ):

            first_ref_page = page["page_index"]
            break

    if first_ref_page is not None:
        for page in pages:
            page_index = page["page_index"]

            if page_index >= first_ref_page or first_ref_page - page_index > 8:
                continue

            for block in page.get("parsing_res_list", []):
                if block.get("block_label") != "text":
                    continue

                text = normalize_text(block.get("block_content", ""))

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
    pages = load_all_pages(pdf)

    if not pages:
        return False, []

    results = []

    for page in pages:
        page_index = page.get("page_index", 0)
        blocks = page.get("parsing_res_list", [])

        for i, block in enumerate(blocks):
            if block.get("block_label") != "paragraph_title":
                continue

            text = normalize_text(
                block.get("block_content", "")
            )

            if not text:
                continue

            if (
                page_index == 0
                and SECTION_BYLINE_RE.fullmatch(text)
            ):
                continue

            if page_index == 0:
                prev_label = None

                for j in range(i - 1, -1, -1):
                    candidate_label = blocks[j].get("block_label")

                    if candidate_label in {
                        "header",
                        "image",
                        "footer_image",
                        "number",
                    }:
                        continue

                    prev_label = candidate_label
                    break

                words = text.split()

                if (
                    prev_label == "doc_title"
                    and 2 <= len(words) <= 6
                    and not re.search(r"[?!:;]", text)
                ):
                    continue

            if SECTION_NON_SECTION_RE.fullmatch(text):
                continue

            if SECTION_RUNNING_HEADER_RE.search(text):
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



def _table_normalize_special(text):
    if not text:
        return ""

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

    text = text.replace("$", "")

    for latex, uni in replacements.items():
        text = text.replace(latex, uni)

    text = text.replace(r"\{", "{")
    text = text.replace(r"\}", "}")
    text = text.replace(r"\_", "_")

    text = re.sub(
        r"\^\{\{([^{}]+)\}\}",
        r"\1",
        text,
    )

    text = re.sub(
        r"\^\{([^{}]+)\}",
        r"\1",
        text,
    )

    text = re.sub(r"\s+", " ", text)

    return normalize_string(text.strip())




def _table_clean_cell(cell):
    text = cell.get_text(" ", strip=True)

    if not text:
        return ""

    text = text.replace("\\n", " ")
    text = re.sub(r"^\s*[•●▪◦]\s*", "", text)
    text = re.sub(r"\s*[•●▪◦]\s*", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _table_serialize(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    rows = []

    for row in soup.find_all("tr"):
        cells = [
            _table_clean_cell(cell)
            for cell in row.find_all(["td", "th"])
        ]

        if cells:
            rows.append(cells)

    if not rows:
        return normalize_string(
            soup.get_text(" ", strip=True)
        )

    if all(len(row) == 2 for row in rows):
        paired_heading_rows = []

        for i, row in enumerate(rows):
            left = row[0].strip()
            right = row[1].strip()

            if (
                left
                and right
                and left.endswith(":")
                and right.endswith(":")
            ):
                paired_heading_rows.append(i)

        if len(paired_heading_rows) >= 2:
            parts = []

            starts = paired_heading_rows

            for pos, start in enumerate(starts):
                end = (
                    starts[pos + 1]
                    if pos + 1 < len(starts)
                    else len(rows)
                )

                left_parts = []
                right_parts = []

                for row in rows[start:end]:
                    if row[0]:
                        left_parts.append(row[0])

                    if row[1]:
                        right_parts.append(row[1])

                if left_parts:
                    parts.extend(left_parts)

                if right_parts:
                    parts.extend(right_parts)

            text = " ".join(parts)
            return _table_normalize_special(text)

    parts = []

    for row in rows:
        for cell in row:
            if cell:
                parts.append(cell)

    return _table_normalize_special(" ".join(parts))


def extract_table(pdf):
    pages = load_all_pages(pdf)

    if not pages:
        return False, []

    results = []

    for page in pages:
        page_index = page.get("page_index", 0)

        for block in page.get("parsing_res_list", []):
            if block.get("block_label") != "table":
                continue

            html_content = block.get("block_content", "")

            if not html_content:
                continue

            text = _table_serialize(html_content)

            if not text:
                continue

            results.append(
                (
                    pdf.pdf_name,
                    page_index,
                    "table",
                    text,
                )
            )

    if not results:
        return False, []

    return True, results




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

    text = re.sub(
        r'\\(?:begin|end)\{(?:aligned|align|array|cases|matrix|bmatrix|pmatrix|vmatrix|Vmatrix|smallmatrix|split|gathered?)\}',
        '',
        text,
    )

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

    text = re.sub(
        r'frac([0-9]+)([0-9]+)',
        r'\1/\2',
        text,
    )

    text = re.sub(
        r'frac([A-Za-zα-ωΑ-Ω])([A-Za-zα-ωΑ-Ω])',
        r'\1/\2',
        text,
    )

    text = re.sub(
        r'frac(∂)([A-Za-zα-ωΑ-Ω])',
        r'\1/\2',
        text,
    )

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

    text = text.replace("mathrmd", "d")
    text = text.replace("\\mathrm{d}", "d")
    text = text.replace("\\mathrm", "")

    text = normalize_frac(text)
    
    text = apply_replacements(text, REPLACEMENT_RULES)

    text = apply_replacements(text, GREEK_RULES)

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

    
    text = re.sub(r'xlongequal\s*\([^)]*\)', '', text)
    text = re.sub(r'\bxlongequal\b', '=', text) 
    
    text = re.sub(
        r'\\(?:big|Big|bigl|bigr|Bigl|Bigr|bigg|Bigg|biggl|biggr|Biggl|Biggr)\b',
        '',
        text,
    )

    text = re.sub(
        r'\\(?:quad|qquad|,|;|!|:|enspace|thinspace|medspace|thickspace)\b',
        '',
        text,
    )

    text = re.sub(
        r'\\(?:phantom|hphantom|vphantom)\{[^{}]*\}',
        '',
        text,
    )


    text = re.sub(r'\\displaystyle\b', '', text)

    text = re.sub(r'\\ldots\b', '...', text)

    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'_\{([^{}]+)\}', r'_\1', text)
    text = re.sub(r'\^\{([^{}]+)\}', r'^\1', text)

    text = re.sub(r'\{([^{}]+)\}', r'\1', text)

    text = text.replace("\\", "")

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
    
    pages = load_all_pages(pdf)
       
    if not pages:
        return False, []

    equations = []

      
    for page in pages:

        page_index = page["page_index"]

        for block in page.get("parsing_res_list", []):

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

    m = re.search(
        r"^\s*\(?\s*dated\s*:\s*"
        r"(\d{1,2})\s+"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
        r"[a-z]*\.?\s+"
        r"(\d{4})",
        text,
        re.I,
    )

    if m:
        d, mon, y = m.groups()
        return f"{int(d)} {_MONTHS[mon.lower()]} {y}"

    text = re.sub(
        r"^(received|accepted|published|available online|online published|dated)\s*[:\-]?\s*",
        "",
        text,
        flags=re.I,
    )


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

        

def _pub_date_from_bibliographic_moddate(pdf, page):
    try:
        info = subprocess.run(
            ["pdfinfo", pdf.filepath],
            capture_output=True,
            text=True,
            check=False,
        ).stdout

        m = re.search(r"^ModDate:\s*(.+)$", info, re.M)
        if not m:
            return None

        raw = re.sub(r"\s+", " ", m.group(1).strip())
        raw = re.sub(r"\s+(?:EST|EDT)$", "", raw, flags=re.I)

        dt = datetime.strptime(
            raw,
            "%a %b %d %H:%M:%S %Y",
        )
        year = str(dt.year)

        bibliographic_hint = re.compile(
            r"""(?ix)
            (?:
                \bvol(?:ume)?\.?\s*\d+
                |
                \b(?:no|n[oº°]|num[eé]ro|issue)\.?\s*\d+
                |
                \bpp?\.?\s*\d+
                |
                \bpages?\s*\d+
                |
                \b\d+\s*[-–—]\s*\d+\b
            )
            """
        )

        for block in page.get("parsing_res_list", []):
            label = str(block.get("block_label", "")).lower()
            if label not in {"footer", "footnote"}:
                continue

            txt = re.sub(
                r"\s+",
                " ",
                str(block.get("block_content", "")),
            )

            if year in txt and bibliographic_hint.search(txt):
                return dt.strftime("%Y-%m-%d")

        return None

    except (OSError, ValueError):
        return None


def extract_pub_date(pdf):

    pages = load_all_pages(pdf)

    if not pages:
        return False, []

    page = pages[0]
    blocks = page.get("parsing_res_list", [])
    
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

        txt = normalize_string(block.get("block_content", ""))
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
        metadata_date = _pub_date_from_bibliographic_moddate(pdf, page)
        if metadata_date:
            return True, [
                (
                    pdf.pdf_name,
                    page["page_index"],
                    "pub_date",
                    metadata_date,
                )
            ]
        return False, []

    candidates.sort(
        key=lambda x: (
            -x["score"],
            len(x["text"]),
        )
    )

    best = candidates[0]

    if best["score"] < PRIORITY["available"]:
        metadata_date = _pub_date_from_bibliographic_moddate(pdf, page)
        if metadata_date:
            return True, [
                (
                    pdf.pdf_name,
                    page["page_index"],
                    "pub_date",
                    metadata_date,
                )
            ]

    return True, [
        (
            pdf.pdf_name,
            best["page"],
            "pub_date",
            best["date"],
        )
    ]




ABSTRACT_HEADING_RE = re.compile(
    r"(?ix)^\s*(?:"
    r"résumé|resume|abstract|resumen"
    r")\s*:?\s*$"
)

ABSTRACT_MULTI_HEADING_RE = re.compile(
    r"(?ix)^\s*"
    r"(?:résumé|resume|abstract|resumen)"
    r"(?:\s*[-–—|/]\s*(?:résumé|resume|abstract|resumen))+"
    r"\s*:?\s*$"
)

ABSTRACT_INLINE_RE = re.compile(
    r"(?ix)^\s*"
    r"(?:résumé|resume|abstract|resumen)"
    r"\s*(?:[:.]|\s*[-–—•]\s*)+\s*"
    r"(.+?)\s*$"
)

ABSTRACT_KEYWORD_RE = re.compile(
    r"(?ix)^\s*(?:"
    r"mots?\s*[- ]?\s*cl[ée]s?"
    r"|keywords?"
    r"|keys?\s+words?"
    r"|palabras?\s+clave"
    r")\b"
)

def extract_abstract(pdf):
    pages = load_all_pages(pdf)

    if not pages:
        return False, []

    results = []

    region_active = False
    parts = []
    start_page = None

    multi_region = False
    multi_expected = 0
    multi_emitted = 0

    def emit_parts():
        nonlocal parts, start_page, multi_emitted
        nonlocal region_active, multi_region
        nonlocal multi_expected

        if parts:
            merged = normalize_text(" ".join(parts))

            if len(merged) >= 120:
                results.append(
                    (
                        pdf.pdf_name,
                        start_page if start_page is not None else 0,
                        "abstract",
                        merged,
                    )
                )

                if multi_region:
                    multi_emitted += 1

        parts = []
        start_page = None

        if (
            multi_region
            and multi_expected > 0
            and multi_emitted >= multi_expected
        ):
            region_active = False
            multi_region = False
            multi_expected = 0
            multi_emitted = 0

    def close_region():
        nonlocal region_active, multi_region
        nonlocal multi_expected, multi_emitted

        emit_parts()

        region_active = False
        multi_region = False
        multi_expected = 0
        multi_emitted = 0

    def count_summary_markers(raw):
        return len(
            re.findall(
                r"(?i)\b(?:résumé|resume|abstract|resumen)\b",
                raw,
            )
        )

    for page in pages:
        page_index = page.get("page_index", 0)
        blocks = _ordered_blocks(page)

        for block in blocks:
            label = block.get("block_label")
            raw = str(block.get("block_content", "") or "").strip()

            if not raw:
                continue

            clean = normalize_text(raw)

            if region_active and label in {
                "header",
                "footer",
                "number",
                "aside_text",
            }:
                continue

            is_multi_heading = bool(
                ABSTRACT_MULTI_HEADING_RE.fullmatch(raw)
            )

            is_single_heading = bool(
                ABSTRACT_HEADING_RE.fullmatch(raw)
            )

            if is_multi_heading or is_single_heading:
                close_region()

                region_active = True
                start_page = page_index

                if is_multi_heading:
                    multi_region = True
                    multi_expected = count_summary_markers(raw)
                    multi_emitted = 0
                else:
                    multi_region = False
                    multi_expected = 0
                    multi_emitted = 0

                continue

            m = ABSTRACT_INLINE_RE.match(raw)

            if m:
                close_region()

                body = normalize_text(m.group(1))

                if len(body) >= 120:
                    results.append(
                        (
                            pdf.pdf_name,
                            page_index,
                            "abstract",
                            body,
                        )
                    )

                continue

            if not region_active:
                continue

            if ABSTRACT_KEYWORD_RE.match(raw):
                if parts:
                    emit_parts()

                    if not multi_region:
                        region_active = False

                else:
                    pass

                continue

            if label in {"paragraph_title", "doc_title"}:
                if multi_region:
                    if parts:
                        emit_parts()

                    if not region_active:
                        continue

                    continue

                if parts:
                    close_region()

                continue

            if label in {"text", "abstract"}:
                if len(clean) < 120:
                    continue

                if start_page is None:
                    start_page = page_index

                parts.append(clean)

                if multi_region and label == "abstract":
                    emit_parts()

                continue

    close_region()

    if not results and pages:
        first_page = pages[0]
        blocks = _ordered_blocks(first_page)[:15]

        has_explicit_marker = False
        has_doc_title = False
        paragraph_title_count = 0
        long_text_blocks = []

        for block in blocks:
            label = block.get("block_label")
            raw = str(block.get("block_content", "") or "").strip()

            if not raw:
                continue

            clean = normalize_text(raw)

            if label == "doc_title":
                has_doc_title = True

            if label == "paragraph_title":
                paragraph_title_count += 1

            if (
                ABSTRACT_HEADING_RE.fullmatch(raw)
                or ABSTRACT_MULTI_HEADING_RE.fullmatch(raw)
                or ABSTRACT_INLINE_RE.match(raw)
            ):
                has_explicit_marker = True

            if label == "text" and len(clean) >= 300:
                long_text_blocks.append(clean)

        if (
            has_doc_title
            and not has_explicit_marker
            and paragraph_title_count >= 1
            and len(long_text_blocks) == 3
        ):
            for body in long_text_blocks:
                results.append(
                    (
                        pdf.pdf_name,
                        first_page.get("page_index", 0),
                        "abstract",
                        body,
                    )
                )

    if not results and len(pages) >= 2:
        p0 = _ordered_blocks(pages[0])
        p1 = _ordered_blocks(pages[1])

        p0_abs = []
        p0_long_text = []

        for block in p0:
            label = block.get("block_label")
            raw = str(block.get("block_content", "") or "").strip()
            if not raw:
                continue

            clean = normalize_text(raw)

            if label == "abstract" and len(clean) >= 120:
                p0_abs.append(clean)

            if label == "text" and len(clean) >= 300:
                p0_long_text.append(clean)

        p1_meaningful = []
        for block in p1:
            label = block.get("block_label")

            if label in {"header", "footer", "number", "aside_text"}:
                continue

            raw = str(block.get("block_content", "") or "").strip()
            if not raw:
                continue

            p1_meaningful.append(
                (label, normalize_text(raw))
            )

        p1_texts = [
            text
            for label, text in p1_meaningful
            if label == "text" and len(text) >= 120
        ]

        page1_content_boundary = (
            len(p1_meaningful) >= 3
            and p1_meaningful[0][0] == "text"
            and p1_meaningful[1][0] == "text"
            and p1_meaningful[2][0] == "content"
        )

        page2_content_boundary = False

        if len(pages) >= 3:
            p2 = _ordered_blocks(pages[2])

            for block in p2:
                label = block.get("block_label")

                if label in {"header", "footer", "number", "aside_text"}:
                    continue

                raw = str(block.get("block_content", "") or "").strip()
                if not raw:
                    continue

                page2_content_boundary = (label == "content")
                break

        if (
            len(p0_abs) == 1
            and len(p1_texts) == 2
            and (page1_content_boundary or page2_content_boundary)
        ):
            first_page_index = pages[0].get("page_index", 0)
            second_page_index = pages[1].get("page_index", 1)

            if p0_abs[0].rstrip().endswith((".", "!", "?", "»", "”")):
                results.append(
                    (
                        pdf.pdf_name,
                        first_page_index,
                        "abstract",
                        p0_abs[0],
                    )
                )

                for body in p1_texts:
                    results.append(
                        (
                            pdf.pdf_name,
                            second_page_index,
                            "abstract",
                            body,
                        )
                    )

            elif len(p0_long_text) == 1:
                results.append(
                    (
                        pdf.pdf_name,
                        first_page_index,
                        "abstract",
                        p0_long_text[0],
                    )
                )

                results.append(
                    (
                        pdf.pdf_name,
                        first_page_index,
                        "abstract",
                        normalize_text(
                            p0_abs[0] + " " + p1_texts[0]
                        ),
                    )
                )

                results.append(
                    (
                        pdf.pdf_name,
                        second_page_index,
                        "abstract",
                        p1_texts[1],
                    )
                )

    if not results and pages:
        page = pages[0]
        blocks = _ordered_blocks(page)

        abstract_indices = []
        raw_abstracts = []

        for i, block in enumerate(blocks):
            if block.get("block_label") != "abstract":
                continue

            raw = normalize_text(
                block.get("block_content", "")
            )

            if not raw:
                continue

            abstract_indices.append(i)
            raw_abstracts.append(raw)

        if raw_abstracts:
            allow_raw_merge = True

            last_index = abstract_indices[-1]

            next_label = None
            if last_index + 1 < len(blocks):
                next_label = blocks[last_index + 1].get(
                    "block_label"
                )

            merged = normalize_text(
                " ".join(raw_abstracts)
            )

            if next_label == "footnote":
                allow_raw_merge = bool(
                    re.match(
                        r"(?i)^\s*"
                        r"(?:abstract|résumé|resume|resumen)"
                        r"\b",
                        merged,
                    )
                )

            if allow_raw_merge and merged:
                results.append(
                    (
                        pdf.pdf_name,
                        page.get("page_index", 0),
                        "abstract",
                        merged,
                    )
                )

    if not results:
        return False, []

    deduped = []
    seen = set()

    for item in results:
        key = (item[0], item[2], item[3])

        if key in seen:
            continue

        seen.add(key)
        deduped.append(item)

    return True, deduped

def extract_list(pdf):

    pages = load_all_pages(pdf)

    if not pages:
        return False, []

    bullet_re = re.compile(
        r"^\s*[•●▪◦*\-–—]\s+"
    )

    numbered_re = re.compile(
        r"^\s*(\d{1,2})\s*[\.\)]\s+"
    )

    results = []

    for page in pages:
        page_index = page.get("page_index", 0)
        blocks = page.get("parsing_res_list", [])

        texts = []

        for block_index, block in enumerate(blocks):
            if block.get("block_label") != "text":
                continue

            text = normalize_text(
                block.get("block_content", "")
            )

            if text:
                texts.append(
                    (block_index, text)
                )

        i = 0

        while i < len(texts):
            block_index, text = texts[i]

            if not bullet_re.match(text):
                i += 1
                continue

            items = [[block_index, text]]
            marker_count = 1
            last_physical_index = block_index

            j = i + 1

            while j < len(texts):
                next_index, next_text = texts[j]

                if next_index != last_physical_index + 1:
                    break

                if bullet_re.match(next_text):
                    items.append(
                        [next_index, next_text]
                    )
                    marker_count += 1
                    last_physical_index = next_index
                    j += 1
                    continue

                if numbered_re.match(next_text):
                    break

                continuation = []
                k = j
                previous_index = last_physical_index

                while k < len(texts):
                    candidate_index, candidate_text = texts[k]

                    if candidate_index != previous_index + 1:
                        break

                    if (
                        bullet_re.match(candidate_text)
                        or numbered_re.match(candidate_text)
                    ):
                        break

                    continuation.append(
                        (candidate_index, candidate_text)
                    )
                    previous_index = candidate_index
                    k += 1

                if (
                    k < len(texts)
                    and texts[k][0] == previous_index + 1
                    and bullet_re.match(texts[k][1])
                ):
                    if continuation:
                        items[-1][1] = normalize_text(
                            " ".join(
                                [items[-1][1]]
                                + [
                                    continuation_text
                                    for _, continuation_text
                                    in continuation
                                ]
                            )
                        )

                        last_physical_index = continuation[-1][0]

                    next_marker_index, next_marker_text = texts[k]

                    items.append(
                        [
                            next_marker_index,
                            next_marker_text,
                        ]
                    )

                    marker_count += 1
                    last_physical_index = next_marker_index
                    j = k + 1
                    continue

                break

            if marker_count >= 2:
                merged = normalize_text(
                    " ".join(
                        item_text
                        for _, item_text in items
                    )
                )

                if merged:
                    results.append(
                        (
                            pdf.pdf_name,
                            page_index,
                            "list",
                            merged,
                        )
                    )

            i = max(j, i + 1)

        i = 0

        while i < len(texts):
            block_index, text = texts[i]
            match = numbered_re.match(text)

            if (
                not match
                or int(match.group(1)) != 1
            ):
                i += 1
                continue

            items = [[block_index, text]]
            marker_count = 1
            expected = 2
            last_physical_index = block_index
            j = i + 1

            while j < len(texts):
                next_index, next_text = texts[j]

                if next_index != last_physical_index + 1:
                    break

                next_match = numbered_re.match(
                    next_text
                )

                if (
                    next_match
                    and int(next_match.group(1)) == expected
                ):
                    items.append(
                        [next_index, next_text]
                    )

                    marker_count += 1
                    expected += 1
                    last_physical_index = next_index
                    j += 1
                    continue

                if (
                    bullet_re.match(next_text)
                    or next_match
                ):
                    break

                continuation = []
                k = j
                previous_index = last_physical_index

                while k < len(texts):
                    candidate_index, candidate_text = texts[k]

                    if candidate_index != previous_index + 1:
                        break

                    candidate_match = numbered_re.match(
                        candidate_text
                    )

                    if (
                        bullet_re.match(candidate_text)
                        or candidate_match
                    ):
                        break

                    continuation.append(
                        (
                            candidate_index,
                            candidate_text,
                        )
                    )

                    previous_index = candidate_index
                    k += 1

                if k >= len(texts):
                    break

                candidate_index, candidate_text = texts[k]

                if candidate_index != previous_index + 1:
                    break

                candidate_match = numbered_re.match(
                    candidate_text
                )

                if (
                    not candidate_match
                    or int(candidate_match.group(1))
                    != expected
                ):
                    break

                if continuation:
                    items[-1][1] = normalize_text(
                        " ".join(
                            [items[-1][1]]
                            + [
                                continuation_text
                                for _, continuation_text
                                in continuation
                            ]
                        )
                    )

                    last_physical_index = continuation[-1][0]

                items.append(
                    [
                        candidate_index,
                        candidate_text,
                    ]
                )

                marker_count += 1
                expected += 1
                last_physical_index = candidate_index
                j = k + 1

            if marker_count >= 2:
                merged = normalize_text(
                    " ".join(
                        item_text
                        for _, item_text in items
                    )
                )

                if merged:
                    results.append(
                        (
                            pdf.pdf_name,
                            page_index,
                            "list",
                            merged,
                        )
                    )

            i = max(j, i + 1)


    for page in pages:
        page_index = page.get("page_index", 0)
        blocks = page.get("parsing_res_list", [])

        for intro_i, intro_block in enumerate(blocks):
            if intro_block.get("block_label") not in {
                "text",
                "doc_title",
            }:
                continue

            intro = normalize_text(
                intro_block.get("block_content", "")
            )

            if not intro or not intro.endswith(":"):
                continue

            run = []
            j = intro_i + 1

            while j < len(blocks):
                block = blocks[j]

                if block.get("block_label") != "text":
                    break

                text = normalize_text(
                    block.get("block_content", "")
                )

                bbox = block.get("block_bbox")

                if (
                    not text
                    or not isinstance(
                        bbox,
                        (list, tuple),
                    )
                    or len(bbox) != 4
                ):
                    break

                run.append(
                    (
                        j,
                        text,
                        bbox,
                    )
                )

                j += 1

            if len(run) < 4:
                continue

            for cut in range(3, len(run)):
                region = run[:cut]
                after = run[cut]

                xs = [
                    float(item[2][0])
                    for item in region
                ]

                x_range = max(xs) - min(xs)

                if x_range > 5.0:
                    continue

                region_lengths = [
                    len(item[1])
                    for item in region
                ]

                item_sized_count = sum(
                    length >= 70
                    for length in region_lengths
                )

                if item_sized_count < 3:
                    continue

                region_x = sum(xs) / len(xs)
                after_x = float(after[2][0])
                after_len = len(after[1])

                layout_break = (
                    abs(after_x - region_x) >= 20.0
                    and
                    after_len
                    >= 2 * max(region_lengths)
                )

                if not layout_break:
                    continue

                merged = normalize_text(
                    " ".join(
                        item[1]
                        for item in region
                    )
                )

                if merged:
                    results.append(
                        (
                            pdf.pdf_name,
                            page_index,
                            "list",
                            merged,
                        )
                    )

                break

    if not results:
        return False, []

    deduped = []
    seen = set()

    for item in results:
        key = (
            item[0],
            item[2],
            item[3],
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(item)

    return True, deduped




def extract_footer(pdf):

    pages = load_all_pages(pdf)

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

        for block in page.get("parsing_res_list", []):

            if block.get("block_label") not in footer_labels:
                continue

            text = normalize_string(block.get("block_content", ""))

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

    pages = load_all_pages(pdf)

    if not pages:
        return False, []

    headers = []

    for page in pages:
        page_index = page.get("page_index", "?")

        blocks = page.get("parsing_res_list", [])

        page_headers = []

        for block in blocks:

            label = block.get("block_label")

            text = normalize_string(block.get("block_content", ""))

            if label != "header":
                continue

            if not text:
                continue

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

def extract_raw(base_dir, label, pdf):
    handlers = {
        "title": extract_title,
        "caption": extract_caption,
        "email": extract_email,
        "affiliation": extract_affiliation,
        "author": extract_author,
        "keyword": extract_keyword,
        "reference": extract_reference,
        "section": extract_section,
        "table": extract_table,
        "equation": extract_equation,
        "pub_date": extract_pub_date,
        "abstract": extract_abstract,
        "list": extract_list,
        "footer": extract_footer,
        "header": extract_header,
    }

    if label not in handlers:
        return False, []

    return handlers[label](pdf)
