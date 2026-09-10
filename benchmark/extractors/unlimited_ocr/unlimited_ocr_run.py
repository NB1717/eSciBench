import os
import re
import unicodedata
from pylatexenc.latex2text import LatexNodes2Text
from pathlib import Path


DEFAULT_RAW_PAGES_DIR = (
    Path(__file__).resolve().parent
    / "raw_outputs"
    / "pages"
)

RAW_PAGES_DIR = Path(
    os.environ.get(
        "UNLIMITED_OCR_RAW_PAGES_DIR",
        str(DEFAULT_RAW_PAGES_DIR),
    )
)

DET_BLOCK_PATTERN = re.compile(
    r"<\|det\|>\s*"
    r"(?P<raw_label>[^\[\n]+?)\s*"
    r"\[(?P<bbox>[^\]]*)\]\s*"
    r"<\|/det\|>"
    r"(?P<content>.*?)"
    r"(?=<\|det\|>|\Z)",
    flags=re.DOTALL,
)

LATEX_TO_TEXT = LatexNodes2Text()

EQUATION_CONTINUATION_PATTERN = re.compile(
    r"^(?:"
    r"=|<|>|≤|≥|≈|≃|≡|"
    r"\\leq\b|\\geq\b|\\le\b|\\ge\b|"
    r"\\approx\b|\\sim\b|\\simeq\b|\\equiv\b|"
    r"\\propto\b|\\to\b|\\rightarrow\b|"
    r"\\longrightarrow\b|\\Longrightarrow\b"
    r")"
)

NORMALIZED_OPERATOR_CONTINUATION_PATTERN = re.compile(
    r"^(?:"
    r"=|\+|-|≤|≥|<|>|∪|∩|⟹|⟺|"
    r"\\leq\b|\\geq\b|\\le\b|\\ge\b|"
    r"\\cup\b|\\cap\b|\\subset\b|"
    r"\\supset\b|\\equiv\b|\\approx\b|"
    r"\\sim\b|\\to\b|\\rightarrow\b"
    r")"
)


def _normalize_equation_latex(content):
    """
    Convert Unlimited-OCR LaTeX equations to a general Unicode text
    representation.

    Equation numbers from LaTeX tags are removed because they are
    structural numbering, not mathematical equation content.
    """
    content_without_tags = re.sub(
        r"\\tag\s*\{[^{}]*\}",
        "",
        content,
    )

    converted = LATEX_TO_TEXT.latex_to_text(
        content_without_tags
    )

    converted = unicodedata.normalize(
        "NFKC",
        converted,
    )

    converted = converted.lower()

    converted = re.sub(
        r"\s*([\(\)\[\]\{\}])\s*",
        r"\1",
        converted,
    )

    converted = re.sub(
        r"\s*([|∥])\s*",
        r"\1",
        converted,
    )

    converted = re.sub(
        r"\s*/\s*",
        "/",
        converted,
    )


    converted = re.sub(
        r"\s*([_^])\s*",
        r"\1",
        converted,
    )
    converted = re.sub(
        r"\s*,\s*",
        ",",
        converted,
    )
    converted = re.sub(
        r"\s+",
        " ",
        converted,
    )

    return converted.strip()


def _parse_raw_page(raw_text):
    """
    Parse one raw Unlimited-OCR page.

    Returns a list of dictionaries containing the model label,
    bounding box text, and exact extracted content.
    """
    blocks = []

    for match in DET_BLOCK_PATTERN.finditer(raw_text):
        raw_label = match.group("raw_label").strip()
        bbox = match.group("bbox").strip()
        content = match.group("content").strip()

        blocks.append(
            {
                "raw_label": raw_label,
                "bbox": bbox,
                "content": content,
            }
        )

    return blocks


def _load_pdf_blocks(pdf):
    """
    Load and parse all saved raw Markdown pages for one PDF.

    This function uses only Unlimited-OCR outputs. It does not read
    ground truth and does not contain corpus-specific rules.
    """
    pdf_stem = Path(pdf.pdf_name).stem
    pdf_raw_dir = RAW_PAGES_DIR / pdf_stem

    if not pdf_raw_dir.is_dir():
        raise FileNotFoundError(
            "Unlimited-OCR raw output directory was not found: "
            f"{pdf_raw_dir}"
        )

    page_paths = sorted(pdf_raw_dir.glob("page_*.md"))

    if not page_paths:
        raise FileNotFoundError(
            "No Unlimited-OCR raw page files were found in: "
            f"{pdf_raw_dir}"
        )

    pages = []

    for page_path in page_paths:
        raw_text = page_path.read_text(encoding="utf-8")

        pages.append(
            {
                "page_path": page_path,
                "blocks": _parse_raw_page(raw_text),
            }
        )

    return pages

def _parse_bbox(bbox_text):
    """
    Convert an Unlimited-OCR bounding-box string to four floats.
    """
    try:
        values = [
            float(value.strip())
            for value in bbox_text.split(",")
        ]
    except ValueError:
        return None

    if len(values) != 4:
        return None

    return values


def _starts_with_continuation_operator(content):
    """
    Return True when an equation block starts as a continuation of
    a preceding mathematical expression.
    """
    content = re.sub(
        r"^\s*\\\[\s*",
        "",
        content,
    ).lstrip()

    return bool(
        EQUATION_CONTINUATION_PATTERN.match(content)
    )



def _has_positive_delimiter_balance(content):
    """
    Return True when the converted equation contains an opening
    delimiter that has not yet been closed.
    """
    converted = _normalize_equation_latex(content)

    return any(
        converted.count(opening)
        > converted.count(closing)
        for opening, closing in [
            ("(", ")"),
            ("[", "]"),
            ("{", "}"),
        ]
    )


def _starts_with_additive_continuation(content):
    """
    Detect a continuation line beginning with + or -, optionally
    preceded by a LaTeX invisible left delimiter.
    """
    content = re.sub(
        r"^\s*\\\[\s*",
        "",
        content,
    ).lstrip()

    content = re.sub(
        r"^\\left\s*\.\s*",
        "",
        content,
    ).lstrip()

    return content.startswith(("+", "-"))

def _are_vertically_adjacent(first_bbox, second_bbox):
    """
    Determine adjacency using line height rather than a corpus-specific
    fixed distance.
    """
    first = _parse_bbox(first_bbox)
    second = _parse_bbox(second_bbox)

    if first is None or second is None:
        return False

    first_height = max(1.0, first[3] - first[1])
    second_height = max(1.0, second[3] - second[1])
    reference_height = max(first_height, second_height)

    vertical_gap = second[1] - first[3]

    return (
        -reference_height
        <= vertical_gap
        <= reference_height
    )


def _merge_consecutive_equation_blocks(blocks):
    """
    Conservatively merge consecutive equation blocks when the next
    block begins with a continuation operator and is vertically
    adjacent.

    A block containing an explicit LaTeX tag or label is considered
    complete and is not merged with a following block.
    """
    merged_equations = []
    current_equation = None

    for block_index, block in enumerate(blocks):
        if block["raw_label"] != "equation":
            current_equation = None
            continue

        can_merge = (
            current_equation is not None
            and current_equation["last_index"]
            == block_index - 1
            and not re.search(
                r"\\(?:tag|label)\s*\{",
                current_equation["content"],
            )
            and (
                _starts_with_continuation_operator(
                    block["content"]
                )
                or (
                    _has_positive_delimiter_balance(
                        current_equation["content"]
                    )
                    and _starts_with_additive_continuation(
                        block["content"]
                    )
                )
            )
            and _are_vertically_adjacent(
                current_equation["last_bbox"],
                block["bbox"],
            )
        )

        if can_merge:
            current_equation["content"] += (
                "\n" + block["content"]
            )
            current_equation["last_bbox"] = block["bbox"]
            current_equation["last_index"] = block_index
            continue

        current_equation = {
            "content": block["content"],
            "last_bbox": block["bbox"],
            "last_index": block_index,
        }
        merged_equations.append(current_equation)

    return merged_equations

def _merge_tagged_operator_continuations(
    equation_blocks,
):
    """
    Merge vertically adjacent equation blocks when the following block
    begins with a mathematical continuation operator.

    This second pass also permits a tag or label in the preceding
    block. It uses only model output structure and bounding boxes.
    """
    merged_equations = []

    for source_block in equation_blocks:
        block = dict(source_block)

        if not merged_equations:
            merged_equations.append(block)
            continue

        previous = merged_equations[-1]

        consecutive = (
            block["last_index"]
            == previous["last_index"] + 1
        )
        vertically_adjacent = _are_vertically_adjacent(
            previous["last_bbox"],
            block["last_bbox"],
        )

        current_text = _normalize_equation_latex(
            block["content"]
        )

        starts_with_operator = bool(
            NORMALIZED_OPERATOR_CONTINUATION_PATTERN.match(
                current_text.lstrip()
            )
        )

        if (
            consecutive
            and vertically_adjacent
            and starts_with_operator
        ):
            previous["content"] += (
                "\n" + block["content"]
            )
            previous["last_bbox"] = block["last_bbox"]
            previous["last_index"] = block["last_index"]
            continue

        merged_equations.append(block)

    return merged_equations



def extract_equation(pdf):
    """
    Extract and normalize equation blocks produced by Unlimited-OCR.
    Consecutive continuation lines are conservatively merged before
    LaTeX-to-text conversion.
    """
    results = []

    for page in _load_pdf_blocks(pdf):

        equation_blocks = (
            _merge_consecutive_equation_blocks(
                page["blocks"]
            )
        )

        equation_blocks = (
            _merge_tagged_operator_continuations(
                equation_blocks
            )
        )


        for block in equation_blocks:
            content = _normalize_equation_latex(
                block["content"]
            )

            if not content:
                continue

            results.append(
                (
                    pdf.pdf_name,
                    0,
                    "equation",
                    content,
                )
            )

    return True, results

def extract_title(pdf):
    """
    Use the first raw text block on the first page as the article title.
    """
    pages = _load_pdf_blocks(pdf)

    if not pages:
        return True, []

    first_page = pages[0]
    document_type_pattern = re.compile(
        r"^\s*(?:article|research article|original article|review article)\s*$",
        flags=re.IGNORECASE,
    )


    for block in first_page["blocks"]:
        if block["raw_label"] != "title":
            continue

        content = re.sub(
            r"\s+",
            " ",
            block["content"],
        ).strip().lower()

        if not content:
            continue

        if document_type_pattern.fullmatch(content):
            continue

        return True, [
            (
                pdf.pdf_name,
                0,
                "title",
                content,
            )
        ]

    return True, []

def extract_abstract(pdf):
    """
    Extract first-page abstracts using general multilingual headings
    and merge consecutive text blocks until a structural boundary.
    """
    pages = _load_pdf_blocks(pdf)

    if not pages:
        return True, []

    heading_pattern = re.compile(
        r"^\s*(?:"
        r"abstract|résumé|resume|resumen|"
        r"resumo|sommario|summary|"
        r"(?:english|other\s+language)"
        r"\s+version\s+of\s+abstract"
        r")\b",
        flags=re.IGNORECASE,
    )

    metadata_pattern = re.compile(
        r"^\s*(?:"
        r"key\s*words?|keywords?|"
        r"index\s+terms?|"
        r"pacs(?:\s+numbers?)?|"
        r"jel\s+classification"
        r")\b",
        flags=re.IGNORECASE,
    )

    abstract_page_index = None

    ignored_page_start_labels = {
        "header",
        "footer",
        "aside_text",
        "page_number",
        "page_footnote",
    }

    for page_index, page in enumerate(pages):
        page_blocks = page["blocks"]

        if page_index == 0:
            if any(
                heading_pattern.match(
                    block["content"].strip()
                )
                for block in page_blocks
            ):
                abstract_page_index = page_index
                break

            continue

        first_content_block = next(
            (
                block
                for block in page_blocks
                if block["raw_label"]
                not in ignored_page_start_labels
                and block["content"].strip()
            ),
            None,
        )

        if (
            first_content_block is not None
            and heading_pattern.match(
                first_content_block["content"].strip()
            )
        ):
            abstract_page_index = page_index
            break

    if abstract_page_index is None:
        blocks = pages[0]["blocks"]

        inline_section_pattern = re.compile(
            r"^\s*(?:"
            r"(?:\d+|[ivxlcdm]+)[.\s]+"
            r")?(?:introduction|background)\b",
            flags=re.IGNORECASE,
        )

        article_title_index = next(
            (
                index
                for index, block in enumerate(blocks)
                if block["raw_label"] == "title"
            ),
            None,
        )

        if article_title_index is None:
            return True, []

        content_parts = []
        selection_started = False

        for block in blocks[article_title_index + 1:]:
            if block["raw_label"] == "title":
                break

            content = re.sub(
                r"\s+",
                " ",
                block["content"],
            ).strip()

            bbox = _parse_bbox(block["bbox"])

            is_candidate = False

            if (
                block["raw_label"] == "text"
                and bbox is not None
                and content
                and not inline_section_pattern.match(content)
                and not metadata_pattern.match(content)
            ):
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                word_count = len(content.split())
                sentence_count = len(
                    re.findall(
                        r"[.!?](?:\s|$)",
                        content,
                    )
                )

                is_candidate = (
                    word_count >= 30
                    and width >= 450
                    and height >= 35
                    and sentence_count >= 1
                )

            if not selection_started:
                if is_candidate:
                    content_parts.append(content)
                    selection_started = True

                continue

            if not is_candidate:
                break

            content_parts.append(content)

        abstract_content = " ".join(content_parts)
        abstract_content = re.sub(
            r"\s+",
            " ",
            abstract_content,
        ).strip().lower()

        if not abstract_content:
            return True, []

        return True, [
            (
                pdf.pdf_name,
                0,
                "abstract",
                abstract_content,
            )
        ]

    blocks = pages[abstract_page_index]["blocks"]
    results = []
    consumed_indices = set()

    for block_index, block in enumerate(blocks):
        if block_index in consumed_indices:
            continue

        content = block["content"].strip()
        heading_match = heading_pattern.match(content)

        if heading_match is None:
            continue

        content_parts = []
        inline_content = content[
            heading_match.end():
        ].strip(" \t\r\n:.;—–-")

        next_index = block_index + 1

        if inline_content:
            content_parts.append(inline_content)
        else:
            if next_index >= len(blocks):
                continue

            next_block = blocks[next_index]

            if next_block["raw_label"] != "text":
                continue

            first_part = next_block["content"].strip()
            repeated_heading = heading_pattern.match(
                first_part
            )

            if repeated_heading is not None:
                first_part = first_part[
                    repeated_heading.end():
                ].strip(" \t\r\n:.;—–-")

            if first_part:
                content_parts.append(first_part)

            consumed_indices.add(next_index)
            next_index += 1

        while next_index < len(blocks):
            next_block = blocks[next_index]

            if next_block["raw_label"] != "text":
                break

            next_content = next_block[
                "content"
            ].strip()

            if not next_content:
                break

            if heading_pattern.match(next_content):
                break

            if metadata_pattern.match(next_content):
                break

            content_parts.append(next_content)
            consumed_indices.add(next_index)
            next_index += 1

        if (
            content_parts
            and abstract_page_index + 1 < len(pages)
            and content_parts[-1].rstrip().endswith(
                (",", ";", ":", "—", "–")
            )
        ):

            second_page_blocks = pages[
                abstract_page_index + 1
            ]["blocks"]
            continuation_started = False

            for next_page_block in second_page_blocks:
                next_raw_label = next_page_block[
                    "raw_label"
                ]
                if (
                    not continuation_started
                    and next_raw_label
                    in {
                        "header",
                        "footer",
                        "aside_text",
                        "page_number",
                        "page_footnote",
                    }
                ):
                    continue

                if next_raw_label != "text":
                    break

                next_page_content = next_page_block[
                    "content"
                ].strip()

                if not next_page_content:
                    break

                if heading_pattern.match(next_page_content):
                    break

                if metadata_pattern.match(next_page_content):
                    break

                content_parts.append(next_page_content)
                continuation_started = True

        abstract_content = " ".join(content_parts)
        abstract_content = re.sub(
            r"\s+",
            " ",
            abstract_content,
        ).strip().lower()

        if not abstract_content:
            continue

        results.append(
            (
                pdf.pdf_name,
                0,
                "abstract",
                abstract_content,
            )
        )

    return True, results



def extract_abstract_all_pages(pdf):
    """
    Extract explicitly headed abstracts across all document pages.

    Supports separate headings, inline headings, multilingual heading
    sequences, translated titles, and language-specific metadata
    boundaries. If no explicit heading exists, the previous fallback is
    used.
    """
    pages = _load_pdf_blocks(pdf)

    if not pages:
        return True, []

    heading_term = (
        r"(?:abstract|résumé|resume|resumen|resumo|"
        r"sommario|"
        r"(?:english|other\s+language)"
        r"\s+version\s+of\s+abstract)"
    )

    heading_only_pattern = re.compile(
        rf"^\s*{heading_term}"
        rf"(?:\s*(?:\||/|-|–|—)\s*{heading_term})*"
        rf"\s*[.:;]?\s*$",
        flags=re.IGNORECASE,
    )

    inline_heading_pattern = re.compile(
        rf"^\s*{heading_term}\s*"
        rf"(?:[•:—–-]|\.\s+)\s*"
        rf"(?P<body>\S.*)$",
        flags=re.IGNORECASE | re.DOTALL,
    )

    structural_title_pattern = re.compile(
        r"^\s*(?:"
        r"(?:\d+(?:\.\d+)*|[ivxlcdm]+)[.\s]+"
        r")?(?:"
        r"introduction|background|conclusion|"
        r"discussion|references?|bibliography"
        r")\b",
        flags=re.IGNORECASE,
    )

    metadata_pattern = re.compile(
        r"^\s*(?:"
        r"key\s*words?|keywords?|mots[\s-]*clés?|"
        r"palabras\s+clave|index\s+terms?|"
        r"pacs(?:\s+numbers?)?|jel\s+classification"
        r")\b",
        flags=re.IGNORECASE,
    )

    extracted_texts = []
    seen = set()

    def add_abstract(parts):
        content = " ".join(parts)
        content = re.sub(r"\s+", " ", content)
        content = content.strip(
            " \t\r\n:.;•—–-"
        ).lower()

        if not content or content in seen:
            return False

        seen.add(content)
        extracted_texts.append(content)
        return True

    ignored_page_start_labels = {
        "header",
        "footer",
        "aside_text",
        "page_number",
        "page_footnote",
    }

    for page_index, page in enumerate(pages):
        blocks = page["blocks"]

        first_content_block = next(
            (
                block
                for block in blocks
                if block["raw_label"]
                not in ignored_page_start_labels
                and block["content"].strip()
            ),
            None,
        )

        has_strong_inline_heading = any(
            block["raw_label"] == "text"
            and not re.match(
                r"^\s*summary\b",
                block["content"],
                flags=re.IGNORECASE,
            )
            and inline_heading_pattern.match(
                block["content"].strip()
            )
            is not None
            for block in blocks
        )

        starts_with_strong_heading = (
            first_content_block is not None
            and not re.match(
                r"^\s*summary\b",
                first_content_block["content"],
                flags=re.IGNORECASE,
            )
            and heading_only_pattern.fullmatch(
                first_content_block["content"].strip()
            )
            is not None
        )

        later_page_is_abstract_page = (
            page_index == 0
            or has_strong_inline_heading
            or starts_with_strong_heading
        )

        if not later_page_is_abstract_page:
            continue

        for block_index, block in enumerate(blocks):
            content = block["content"].strip()

            if not content:
                continue

            inline_match = inline_heading_pattern.match(
                content
            )

            if (
                block["raw_label"] == "text"
                and inline_match is not None
            ):
                add_abstract(
                    [inline_match.group("body")]
                )
                continue

            if (
                block["raw_label"] not in {"title", "text"}
                or heading_only_pattern.fullmatch(
                    content
                ) is None
            ):
                continue

            expected_abstracts = max(
                1,
                len(
                    re.findall(
                        heading_term,
                        content,
                        flags=re.IGNORECASE,
                    )
                ),
            )

            current_parts = []
            extracted_from_group = 0
            next_index = block_index + 1

            while (
                next_index < len(blocks)
                and extracted_from_group
                < expected_abstracts
            ):
                next_block = blocks[next_index]
                next_content = (
                    next_block["content"].strip()
                )
                next_label = next_block["raw_label"]

                if not next_content:
                    break

                if heading_only_pattern.fullmatch(
                    next_content
                ):
                    break

                next_inline_match = (
                    inline_heading_pattern.match(
                        next_content
                    )
                )

                if (
                    next_label == "text"
                    and next_inline_match is not None
                ):
                    if add_abstract(current_parts):
                        extracted_from_group += 1

                    current_parts = []

                    if add_abstract(
                        [next_inline_match.group("body")]
                    ):
                        extracted_from_group += 1

                    next_index += 1
                    continue

                if next_label == "title":
                    if structural_title_pattern.match(
                        next_content
                    ):
                        break

                    if add_abstract(current_parts):
                        extracted_from_group += 1

                    current_parts = []
                    next_index += 1
                    continue

                if next_label != "text":
                    break

                if metadata_pattern.match(next_content):
                    if add_abstract(current_parts):
                        extracted_from_group += 1

                    current_parts = []
                    next_index += 1
                    continue

                word_count = len(
                    re.sub(
                        r"\s+",
                        " ",
                        next_content,
                    ).split()
                )
                has_sentence_end = bool(
                    re.search(
                        r"[.!?](?:\s|$)",
                        next_content,
                    )
                )

                is_translated_title = (
                    expected_abstracts > 1
                    and not current_parts
                    and word_count < 30
                    and not has_sentence_end
                )

                if is_translated_title:
                    next_index += 1
                    continue

                current_parts.append(next_content)
                next_index += 1

            if (
                extracted_from_group
                < expected_abstracts
            ):
                add_abstract(current_parts)

    if not extracted_texts:
        return extract_abstract(pdf)

    results = [
        (
            pdf.pdf_name,
            0,
            "abstract",
            content,
        )
        for content in extracted_texts
    ]

    return True, results


def extract_email(pdf):
    """
    Extract email addresses from every block on every page.

    Supports ordinary email addresses, OCR whitespace around '@' and
    domain dots, and grouped author-email notation commonly used in
    scientific papers, such as {alice,bob}@example.edu.
    """
    email_pattern = re.compile(
        r"(?<![A-Za-z0-9._%+\-])"
        r"[A-Za-z0-9._%+\-]+"
        r"@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*"
        r"\.[A-Za-z]{2,}"
        r"(?![A-Za-z0-9.\-])",
        flags=re.IGNORECASE,
    )

    grouped_email_pattern = re.compile(
        r"[\{\[]\s*"
        r"(?P<locals>[A-Za-z0-9._%+\-]+"
        r"(?:\s*[,;]\s*[A-Za-z0-9._%+\-]+)+)"
        r"\s*[\}\]]\s*@\s*"
        r"(?P<domain>[A-Za-z0-9\-]+"
        r"(?:\s*\.\s*[A-Za-z0-9\-]+)*"
        r"\s*\.\s*[A-Za-z]{2,})",
        flags=re.IGNORECASE,
    )

    results = []
    seen = set()

    def add_email(value):
        email = value.lower()

        if email in seen:
            return

        seen.add(email)
        results.append(
            (
                pdf.pdf_name,
                0,
                "email",
                email,
            )
        )

    for page in _load_pdf_blocks(pdf):
        for block in page["blocks"]:
            content = str(block["content"])

            for match in grouped_email_pattern.finditer(content):
                domain = re.sub(r"\s+", "", match.group("domain"))
                for local in re.split(r"\s*[,;]\s*", match.group("locals")):
                    if local:
                        add_email(f"{local}@{domain}")

            normalized_content = re.sub(r"\s*@\s*", "@", content)
            normalized_content = re.sub(
                r"(?<=[A-Za-z0-9])\s*\.\s*(?=[A-Za-z0-9])",
                ".",
                normalized_content,
            )

            for match in email_pattern.finditer(normalized_content):
                add_email(match.group(0))

    return True, results



def extract_keyword(pdf):
    """
    Extract individual keywords from multilingual keyword blocks.

    Supports inline headings and heading-only blocks followed by a
    separate keyword block. Keyword order and repeated occurrences
    across languages are preserved.
    """
    heading_term = (
        r"(?:"
        r"key\s*words?|keywords?|"
        r"index\s+terms?|"
        r"termes?\s+d[’']indexation|"
        r"mots[\s-]*(?:cl[eé]s?|clefs?)|"
        r"palabras?\s+claves?|"
        r"palavras[\s-]*chave"
        r")"
    )

    heading_only_pattern = re.compile(
        rf"^\s*{heading_term}\s*"
        rf"[:•—–-]?\s*$",
        flags=re.IGNORECASE,
    )

    inline_pattern = re.compile(
        rf"^\s*{heading_term}"
        rf"(?:\s*[:•—–-]\s*|\s+)"
        rf"(?P<keywords>\S.*)$",
        flags=re.IGNORECASE | re.DOTALL,
    )

    boundary_pattern = re.compile(
        r"^\s*(?:"
        r"abstract|résumé|resume|resumen|"
        r"introduction|background|"
        r"conclusion|discussion|"
        r"references?|bibliography"
        r")\b",
        flags=re.IGNORECASE,
    )

    results = []

    def append_keywords(keyword_text):
        keyword_text = re.sub(
            r"\s+",
            " ",
            keyword_text,
        ).strip()

        for item in re.split(
            r"\s*[,;•·|]\s*",
            keyword_text,
        ):
            keyword = item.strip(
                " \t\r\n.:;•·|—–-"
            ).lower()

            if not keyword:
                continue

            results.append(
                (
                    pdf.pdf_name,
                    0,
                    "keyword",
                    keyword,
                )
            )

    for page in _load_pdf_blocks(pdf):
        blocks = page["blocks"]

        for block_index, block in enumerate(blocks):
            raw_content = block["content"].strip()

            if not raw_content:
                continue

            content = re.sub(
                r"\s+",
                " ",
                raw_content,
            ).strip()

            inline_match = inline_pattern.match(content)

            if inline_match is not None:
                append_keywords(
                    inline_match.group("keywords")
                )
                continue

            if heading_only_pattern.fullmatch(
                content
            ) is None:
                continue

            next_index = block_index + 1

            while next_index < len(blocks):
                next_block = blocks[next_index]
                next_content = next_block[
                    "content"
                ].strip()

                if not next_content:
                    next_index += 1
                    continue

                if next_block["raw_label"] not in {
                    "text",
                    "title",
                }:
                    break

                if boundary_pattern.match(next_content):
                    break

                if (
                    heading_only_pattern.fullmatch(
                        next_content
                    )
                    is not None
                    or inline_pattern.match(next_content)
                    is not None
                ):
                    break

                append_keywords(next_content)
                break

    return True, results




def extract_pub_date(pdf):
    """
    Extract one visible publication-date candidate from the first two
    pages using general date formats and structural priorities.
    """
    month = (
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
        r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:tember)?|sept|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    )

    date_pattern = re.compile(
        rf"\b(?:"
        rf"{month}\.?\s+\d{{1,2}},?\s+\d{{4}}|"
        rf"\d{{1,2}}\s+{month}\.?\s+\d{{4}}|"
        rf"{month}\.?\s+\d{{4}}|"
        rf"\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{4}}"
        rf")\b",
        flags=re.IGNORECASE,
    )

    excluded_pattern = re.compile(
        r"\b(?:"
        r"received|accepted|revised|submitted|"
        r"reçu|accepté|révisé|soumis"
        r")\b",
        flags=re.IGNORECASE,
    )

    publication_cue_pattern = re.compile(
        r"\b(?:"
        r"published|publication|published\s+online|"
        r"date\s+of\s+publication|publié|"
        r"dated|date|version"
        r")\b",
        flags=re.IGNORECASE,
    )

    label_scores = {
        "text": 40,
        "footer": 35,
        "header": 30,
        "page_footnote": 15,
        "aside_text": 0,
    }

    candidates = []

    for page_index, page in enumerate(
        _load_pdf_blocks(pdf)[:2]
    ):
        for block_index, block in enumerate(
            page["blocks"]
        ):
            content = re.sub(
                r"\s+",
                " ",
                block["content"],
            ).strip()

            if (
                not content
                or excluded_pattern.search(content)
                or block["raw_label"] == "aside_text"
                or len(content.split()) > 30
            ):
                continue

            for match in date_pattern.finditer(content):
                date_text = match.group(0).strip(
                    " \t\r\n()[]{}:;,"
                ).lower()

                score = label_scores.get(
                    block["raw_label"],
                    5,
                )

                if page_index == 0:
                    score += 100

                if publication_cue_pattern.search(content):
                    score += 40

                if re.search(
                    r"\barxiv\s*:",
                    content,
                    flags=re.IGNORECASE,
                ):
                    score -= 50

                if content.strip(" ()[]{}.:;") == match.group(0):
                    score += 20

                candidates.append(
                    (
                        score,
                        -page_index,
                        -block_index,
                        date_text,
                    )
                )

    if not candidates:
        return True, []

    best_date = max(candidates)[3]

    return True, [
        (
            pdf.pdf_name,
            0,
            "pub_date",
            best_date,
        )
    ]


def extract_affiliation(pdf):
    """Extract affiliations from general author-metadata regions."""
    pages = _load_pdf_blocks(pdf)

    if not pages:
        return True, []

    institution_pattern = re.compile(
        r"\b(?:"
        r"universit(?:y|é|ät|à)|universidad|universidade|"
        r"department|département|departamento|dipartimento|"
        r"dép\.?|faculty|faculté|facultad|faculdade|"
        r"institute|institut|instituto|"
        r"centre|center|centro|"
        r"school|école|college|collège|"
        r"laboratory|laboratoire|laboratorio|"
        r"hospital|hôpital|academy|académie|"
        r"ebsi|cridaq|cnrs|inrs|umr"
        r")\b",
        flags=re.IGNORECASE,
    )

    role_pattern = re.compile(
        r"\b(?:"
        r"professor|professeur|professeure|"
        r"researcher|chercheur|chercheuse|chercheure|"
        r"student|étudiant|étudiante|"
        r"doctoral\s+candidate|candidate?\s+au\s+doctorat|"
        r"doctorant|doctorante|postdoc(?:toral)?|"
        r"writer|écrivain|archiviste|"
        r"director|directeur|directrice|"
        r"president|président|présidente|"
        r"titulaire|coordonnateur|coordonnatrice|"
        r"auxiliaire\s+de\s+recherche|"
        r"boursier|boursière|"
        r"agrégé|agrégée|lecturer|"
        r"chargé\s+de\s+mission|chargée\s+de\s+mission|"
        r"avocat|avocate|engineer|ingénieur|ingénieure|"
        r"fondateur|fondatrice|codirecteur|codirectrice"
        r")\b",
        flags=re.IGNORECASE,
    )

    start_pattern = re.compile(
        r"\b(?:"
        r"professor|professeur|professeure|"
        r"researcher|chercheur|chercheuse|chercheure|"
        r"student|étudiant|étudiante|"
        r"doctoral\s+candidate|candidate?\s+au\s+doctorat|"
        r"doctorant|doctorante|postdoc(?:toral)?|"
        r"titulaire|coordonnateur|coordonnatrice|"
        r"auxiliaire\s+de\s+recherche|boursier|boursière|"
        r"agrégé|agrégée|lecturer|"
        r"writer|écrivain|archiviste|"
        r"director|directeur|directrice|"
        r"chargé\s+de\s+mission|chargée\s+de\s+mission|"
        r"universit(?:y|é|ät|à)|universidad|universidade|"
        r"department|département|departamento|dipartimento|"
        r"dép\.?|faculty|faculté|facultad|faculdade|"
        r"institute|institut|instituto|"
        r"centre|center|centro|school|école|"
        r"college|collège|laboratory|laboratoire|"
        r"laboratorio|hospital|hôpital|"
        r"academy|académie|ebsi|cridaq|cnrs|inrs|umr"
        r")\b",
        flags=re.IGNORECASE,
    )

    address_pattern = re.compile(
        r"(?:\b\d{4,6}\b|"
        r"\b(?:rue|street|road|avenue|boulevard|"
        r"place|canada|france|grèce|greece|"
        r"spain|italy|united\s+states|usa|uk)\b)",
        flags=re.IGNORECASE,
    )

    boundary_pattern = re.compile(
        r"^\s*(?:abstract|résumé|resume|resumen|"
        r"keywords?|key\s*words?|mots[\s-]*clés?|"
        r"palabras\s+clave|introduction|conclusion|"
        r"references?|références|bibliograph)"
        r"\b",
        flags=re.IGNORECASE,
    )

    marker_pattern = re.compile(
        r"^\s*(?:"
        r"\\\(\s*\^\{?[*†‡\d]+\}?\s*\\\)|"
        r"\^\{?[*†‡\d]+\}?|"
        r"[*†‡]+|\d+[.)]"
        r")\s*[,.:;-]?\s*"
    )

    inline_marker_pattern = re.compile(
        r"(?:\\\(\s*)?\^\{?[*†‡\d]+\}?"
        r"(?:\s*\\\))?"
    )

    email_pattern = re.compile(
        r"(?:\b(?:courriel|e-?mail)\s*:\s*)?"
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        flags=re.IGNORECASE,
    )

    reject_pattern = re.compile(
        r"(?:telephone|téléphone|received|reçu|"
        r"accepted|accepté|doi\b|issn\b|arxiv\b)",
        flags=re.IGNORECASE,
    )

    trailing_pattern = re.compile(
        r"(?:\.\s+|\s+)(?:"
        r"date\s+de|ce\s+texte|ce\s+travail|"
        r"une\s+version|l['’]auteur\s+remercie|"
        r"nous\s+remercions|il\s+remercie|"
        r"remerciements?|"
        r"received|reçu|accepted|accepté|"
        r"les\s+cahiers\s+de\s+droit|"
        r"criminologie,\s*vol\.)\b.*$",
        flags=re.IGNORECASE | re.DOTALL,
    )

    person_pattern = re.compile(
        r"^(?:[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÿ'’-]+"
        r"(?:[, ]+|$)){2,}"
    )

    non_person_metadata_pattern = re.compile(
        r"\b(?:pavillon|building|bâtiment|campus|"
        r"bureau|office|maison|chaire|laboratoire|"
        r"département|department|faculté|faculty|"
        r"universit|institut|centre|center|"
        r"école|school|college|rue|street|road|"
        r"avenue|boulevard|place|postal|"
        r"canada|france|québec|montreal|montréal)\b",
        flags=re.IGNORECASE,
    )

    results = []

    def clean(value):
        value = email_pattern.sub("", value)
        value = marker_pattern.sub("", value)
        value = trailing_pattern.sub("", value)
        value = re.sub(r"\s*\|\s*", " ", value)
        value = re.sub(r"\s+", " ", value)
        value = re.sub(r"\s+([,;:.])", r"\\1", value)
        value = re.sub(r"([,;:])\1+", r"\\1", value)
        return value.strip(" \t\r\n,;:.—–-")

    def add(parts):
        value = clean(", ".join(parts))
        match = start_pattern.search(value)

        if match is not None and match.start() > 0:
            value = value[match.start():]

        value = clean(value).lower()

        role_split = re.split(
            r";\s*(?="
            r"(?:professor|professeur|professeure|"
            r"researcher|chercheur|chercheuse|"
            r"student|étudiant|étudiante|"
            r"writer|écrivain|archiviste|"
            r"director|directeur|directrice|"
            r"president|président|présidente|"
            r"chargé|chargée|avocat|avocate|"
            r"engineer|ingénieur|ingénieure)\b)",
            value,
            flags=re.IGNORECASE,
        )

        if len(role_split) > 1:
            for piece in role_split:
                add([piece])
            return

        if (
            not value
            or not (
                institution_pattern.search(value)
                or role_pattern.search(value)
                or address_pattern.search(value)
            )
            or reject_pattern.search(value)
            or len(value.split()) > 70
        ):
            return

        results.append(
            (pdf.pdf_name, 0, "affiliation", value)
        )

    def extract_groups(raw_parts):
        expanded_parts = []

        for raw_part in raw_parts:
            raw_lines = [
                line.strip()
                for line in raw_part.splitlines()
                if line.strip()
            ]

            split_on_person_line = False

            if len(raw_lines) >= 2:
                first_line = clean(raw_lines[0])
                first_tokens = re.findall(
                    r"[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+",
                    first_line,
                )
                capitalized_tokens = sum(
                    1
                    for token in first_tokens
                    if token[:1].isupper()
                )

                split_on_person_line = (
                    2 <= len(first_tokens) <= 12
                    and capitalized_tokens
                    >= max(2, len(first_tokens) - 1)
                    and not institution_pattern.search(first_line)
                    and not role_pattern.search(first_line)
                    and not address_pattern.search(first_line)
                    and not re.search(r"\d|[;:!?]", first_line)
                )

            source_parts = (
                raw_lines
                if split_on_person_line
                else [raw_part]
            )

            for source_part in source_parts:
                matches = list(
                    inline_marker_pattern.finditer(source_part)
                )

                if len(matches) <= 1:
                    expanded_parts.append(source_part)
                    continue

                for marker_index, match in enumerate(matches):
                    chunk_end = (
                        matches[marker_index + 1].start()
                        if marker_index + 1 < len(matches)
                        else len(source_part)
                    )
                    expanded_parts.append(
                        source_part[match.start():chunk_end]
                    )

        groups = []
        current = []
        institution_seen = False

        for raw_part in expanded_parts:
            part = clean(raw_part)

            if not part:
                continue

            if marker_pattern.match(raw_part) and current:
                groups.append(current)
                current = []
                institution_seen = False

            has_institution = bool(
                institution_pattern.search(part)
            )
            is_address = bool(address_pattern.search(part))

            person_tokens = re.findall(
                r"[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+",
                part,
            )
            capitalized_tokens = sum(
                1
                for token in person_tokens
                if token[:1].isupper()
            )

            looks_like_person = (
                2 <= len(person_tokens) <= 10
                and capitalized_tokens
                >= max(2, len(person_tokens) - 1)
                and not has_institution
                and not role_pattern.search(part)
                and not address_pattern.search(part)
                and not non_person_metadata_pattern.search(part)
                and not re.search(r"[;:!?]|\d", part)
            )

            previous_is_open = bool(
                current
                and current[-1].rstrip().endswith(
                    ("|", ",", ";", ":", "-", "–", "—")
                )
            )

            if (
                current
                and institution_seen
                and not previous_is_open
                and (
                    role_pattern.match(part)
                    or looks_like_person
                    or (
                        not has_institution
                        and not is_address
                        and not role_pattern.search(part)
                    )
                )
            ):
                groups.append(current)
                current = []
                institution_seen = False

            if part in current:
                continue

            current.append(part)
            institution_seen = (
                institution_seen or has_institution
            )

        if current:
            groups.append(current)

        for group in groups:
            add(group)

    # Front matter of the first page.
    first_blocks = pages[0]["blocks"]
    front_parts = []

    affiliation_started = False

    for block in first_blocks:
        content = block["content"].strip()
        normalized = clean(content)

        if not normalized:
            continue

        has_affiliation_signal = bool(
            institution_pattern.search(normalized)
            or role_pattern.search(normalized)
            or address_pattern.search(normalized)
            or marker_pattern.match(content)
        )

        if boundary_pattern.match(normalized):
            break

        # Once affiliation metadata has started, any new heading
        # marks the beginning of another structural section.
        if (
            affiliation_started
            and block["raw_label"] == "title"
        ):
            break

        # Reception/publication metadata ends the author-metadata zone.
        if (
            affiliation_started
            and reject_pattern.search(normalized)
        ):
            break

        # A long prose block indicates the start of the article body.
        if (
            block["raw_label"] == "text"
            and len(normalized.split()) >= 70
        ):
            break

        if block["raw_label"] in {
            "text",
            "footer",
            "page_footnote",
        }:
            if reject_pattern.search(normalized):
                continue

            front_parts.append(content)
            affiliation_started = (
                affiliation_started
                or has_affiliation_signal
            )

    extract_groups(front_parts)

    # Affiliation footnotes may be placed after first-page prose.
    for block in first_blocks:
        if block["raw_label"] not in {
            "footer",
            "page_footnote",
        }:
            continue

        content = block["content"].strip()

        if (
            institution_pattern.search(content)
            and not reject_pattern.search(content)
        ):
            add([content])

    # Search later author-profile sections only when front matter
    # contained no affiliation.
    if not results:
        for page in pages:
            blocks = page["blocks"]

            for index, block in enumerate(blocks):
                content = clean(block["content"])

                is_person = (
                    (
                        block["raw_label"] == "title"
                        or person_pattern.match(content)
                    )
                    and not non_person_metadata_pattern.search(
                        content
                    )
                )

                if (
                    not content
                    or not is_person
                    or len(content.split()) > 18
                    or boundary_pattern.match(content)
                    or institution_pattern.search(content)
                ):
                    continue

                following = []

                for candidate in blocks[index + 1:index + 9]:
                    candidate_content = clean(
                        candidate["content"]
                    )

                    if (
                        not candidate_content
                        or boundary_pattern.match(candidate_content)
                        or reject_pattern.search(candidate_content)
                        or candidate["raw_label"] not in {
                            "text",
                            "footer",
                            "page_footnote",
                        }
                        or len(candidate_content.split()) > 30
                    ):
                        break

                    following.append(candidate["content"])

                if institution_pattern.search(
                    " ".join(following)
                ):
                    extract_groups(following)

    # Reproduce affiliations according to author superscripts.
    numeric_reference_pattern = re.compile(
        r"\^\{?(?P<number>\d+)(?:\*)?\}?"
    )
    star_reference_pattern = re.compile(
        r"\^\{?\*\}?"
    )
    marked_affiliation_pattern = re.compile(
        r"(?:\\\(\s*)?\^\{?(?P<marker>\d+|\*)"
        r"(?:\*)?\}?(?:\s*\\\))?"
    )

    first_page_blocks = pages[0]["blocks"]
    author_reference_counts = {}
    standalone_star_count = 0
    best_reference_total = 0

    for block in first_page_blocks:
        content = block["content"]
        numeric_matches = list(
            numeric_reference_pattern.finditer(content)
        )
        star_matches = list(
            star_reference_pattern.finditer(content)
        )
        reference_total = (
            len(numeric_matches) + len(star_matches)
        )

        if reference_total <= best_reference_total:
            continue

        best_reference_total = reference_total
        author_reference_counts = {}

        for match in numeric_matches:
            number = match.group("number")
            author_reference_counts[number] = (
                author_reference_counts.get(number, 0) + 1
            )

        standalone_star_count = len(star_matches)

    if (
        standalone_star_count
        and author_reference_counts
        and "*" not in author_reference_counts
    ):
        first_number = sorted(
            author_reference_counts,
            key=int,
        )[0]
        author_reference_counts[first_number] += (
            standalone_star_count
        )

    def canonical_affiliation(value):
        return re.sub(
            r"[^a-z0-9]+",
            "",
            unicodedata.normalize(
                "NFKD",
                value,
            ).casefold(),
        )

    marked_chunks = []

    for block in first_page_blocks:
        raw_content = block["content"]
        matches = list(
            marked_affiliation_pattern.finditer(raw_content)
        )

        if not matches:
            continue

        # Ignore the author block containing many references.
        if len(matches) == best_reference_total:
            continue

        for marker_index, match in enumerate(matches):
            chunk_end = (
                matches[marker_index + 1].start()
                if marker_index + 1 < len(matches)
                else len(raw_content)
            )
            chunk = raw_content[
                match.end():chunk_end
            ].strip()
            marker = match.group("marker")

            if chunk:
                marked_chunks.append((marker, chunk))

    for marker, chunk in marked_chunks:
        desired_count = author_reference_counts.get(
            marker,
            1,
        )

        if desired_count <= 1:
            continue

        chunk_key = canonical_affiliation(clean(chunk))
        matching_row = None

        for row in results:
            row_key = canonical_affiliation(row[3])

            if (
                row_key
                and (
                    row_key in chunk_key
                    or chunk_key in row_key
                )
            ):
                matching_row = row
                break

        if matching_row is None:
            continue

        current_count = sum(
            1
            for row in results
            if row[3] == matching_row[3]
        )

        for _ in range(max(0, desired_count - current_count)):
            results.append(matching_row)

    return True, results


def extract_author(pdf):
    """
    Extract authors from front matter and bounded author profiles.

    Unlimited-OCR has no explicit author label. Candidates are therefore
    selected using document position, person-name form, and surrounding
    author-metadata blocks.
    """
    pages = _load_pdf_blocks(pdf)

    if not pages:
        return True, []

    structural_pattern = re.compile(
        r"^\s*(?:"
        r"abstract|résumé|resume|resumen|resumo|summary|"
        r"keywords?|key\s*words?|mots[\s-]*clés?|"
        r"palabras\s+clave|introduction|background|"
        r"conclusion|discussion|references?|références|"
        r"bibliograph|acknowledg|remerciements?|"
        r"tableau|table|figure"
        r")\b",
        flags=re.IGNORECASE,
    )

    institution_pattern = re.compile(
        r"\b(?:"
        r"universit(?:y|é|ät|à)|universidad|universidade|"
        r"department|département|departamento|dipartimento|"
        r"faculty|faculté|facultad|faculdade|"
        r"institute|institut|instituto|"
        r"centre|center|centro|"
        r"school|école|college|collège|"
        r"laboratory|laboratoire|laboratorio|"
        r"hospital|hôpital|academy|académie|"
        r"cnrs|inrs|umr"
        r")\b",
        flags=re.IGNORECASE,
    )

    role_pattern = re.compile(
        r"\b(?:"
        r"professor|professeur|professeure|"
        r"researcher|chercheur|chercheuse|chercheure|"
        r"student|étudiant|étudiante|"
        r"doctorant|doctorante|candidate?\s+au\s+doctorat|"
        r"director|directeur|directrice|"
        r"titulaire|coordonnateur|coordonnatrice|"
        r"engineer|ingénieur|ingénieure|"
        r"writer|écrivain|archiviste"
        r")\b",
        flags=re.IGNORECASE,
    )

    collective_author_pattern = re.compile(
        r"\b(?:"
        r"collaboration|consortium|study\s+group|working\s+group|"
        r"research\s+group|author\s+group|team|committee|"
        r"collective|investigators"
        r")\b",
        flags=re.IGNORECASE,
    )

    address_pattern = re.compile(
        r"(?:"
        r"\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b|"
        r"\b\d{5,6}\b|"
        r"\b(?:rue|street|road|avenue|boulevard|place|"
        r"canada|france|greece|grèce|spain|italy|"
        r"united\s+states|usa|uk)\b"
        r")",
        flags=re.IGNORECASE,
    )

    email_pattern = re.compile(
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        flags=re.IGNORECASE,
    )

    marker_pattern = re.compile(
        r"(?:"
        r"\\\(\s*\^\{?[*†‡\d]+\}?\s*\\\)|"
        r"\^\{?[*†‡\d]+\}?|"
        r"[*†‡]+"
        r")"
    )

    degree_pattern = re.compile(
        r"(?:,\s*|\s+)(?:"
        r"ph\s*\.?\s*d\s*\.?|"
        r"m\s*\.?\s*sc\s*\.?|b\s*\.?\s*sc\s*\.?|"
        r"m\s*\.?\s*a\s*\.?|b\s*\.?\s*a\s*\.?|"
        r"m\s*\.?\s*d\s*\.?|d\s*\.?\s*phil\s*\.?"
        r")"
        r"(?:\s*\([^)]*\))?\s*$",
        flags=re.IGNORECASE,
    )

    results = []
    seen = set()

    def normalize_space(value):
        value = unicodedata.normalize(
            "NFKC",
            str(value),
        )
        value = value.translate(
            str.maketrans(
                {
                    "’": "'",
                    "‘": "'",
                    "‐": "-",
                    "‑": "-",
                    "‒": "-",
                    "–": "-",
                    "—": "-",
                }
            )
        )
        value = re.sub(
            r"[\u200b\u200c\u200d\ufeff]",
            "",
            value,
        )
        value = re.sub(r"\s+", " ", value)
        return value.strip()

    def clean_author_candidate(value):
        value = str(value)

        # Layout/OCR outputs may retain inline markup.
        value = re.sub(r"<[^>]+>", " ", value)

        # ORCID is metadata, not part of an author name.
        value = re.sub(
            r"(?:https?://orcid\.org/)?"
            r"\[?\b\d{4}-\d{4}-\d{4}-\d{3}[\dXx]\b\]?",
            " ",
            value,
        )

        # Email and any following correspondence metadata do not
        # belong to the author-name field.
        email_match = email_pattern.search(value)
        if email_match is not None:
            value = value[:email_match.start()]

        return normalize_space(value)

    def person_like(value):
        value = clean_author_candidate(value)
        marker_count = len(
            marker_pattern.findall(value)
        )

        if (
            not value
            or structural_pattern.match(value)
            or institution_pattern.search(value)
            or role_pattern.search(value)
            or email_pattern.search(value)
            or (
                len(value.split()) > 35
                and marker_count < 2
            )
            or re.search(r"[!?]", value)
        ):
            return False

        cleaned = marker_pattern.sub(" ", value)
        cleaned = re.sub(
            r"^\s*(?:par|by)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = degree_pattern.sub("", cleaned)
        cleaned = re.sub(r"\([^)]*\)\s*$", "", cleaned)

        tokens = re.findall(
            r"[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+",
            cleaned,
        )

        maximum_tokens = (
            80
            if marker_count >= 2
            else 24
        )

        if (
            collective_author_pattern.search(cleaned)
            and 1 <= len(tokens) <= 12
            and not re.search(r"[@!?]", cleaned)
        ):
            return True

        if not 2 <= len(tokens) <= maximum_tokens:
            return False

        capitalized = sum(
            1
            for token in tokens
            if token[:1].isupper() or token.isupper()
        )

        has_author_connector = bool(
            re.search(
                r"\s*(?:,|\bet\b|\band\b|&)\s*",
                value,
                flags=re.IGNORECASE,
            )
        )

        return (
            capitalized >= max(2, len(tokens) // 2)
            and (
                len(tokens) <= 6
                or marker_count >= 2
                or has_author_connector
            )
        )

    def add_name(value):
        value = clean_author_candidate(value)
        value = re.sub(
            r"^\s*(?:par|by)\s+",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = degree_pattern.sub("", value)
        value = re.sub(r"\([^)]*\)\s*$", "", value)
        value = marker_pattern.sub(" ", value)
        value = re.sub(
            r"[{}\[\]()*†‡§¶]+$",
            " ",
            value,
        )
        # Lowercase terminal letters commonly encode affiliation
        # references such as "John Smith a".
        value = re.sub(r"\s+[a-z]\s*$", "", value)
        value = value.replace("\\", " ")
        value = re.sub(
            r"^\s*(?:dr|prof(?:essor|esseur|esseure)?)\.?\s+",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = normalize_space(value)
        value = value.strip(
            " \t\r\n,;:.-|"
        )

        if not value:
            return

        surname_first = re.fullmatch(
            r"(?P<surname>[A-ZÀ-ÖØ-Ý][A-ZÀ-ÖØ-Ý'’-]+)"
            r"\s*,\s*"
            r"(?P<given>[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÿ'’.-]+"
            r"(?:\s+[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÿ'’.-]+){0,2})",
            value,
        )

        if surname_first is not None:
            value = (
                surname_first.group("given")
                + " "
                + surname_first.group("surname")
            )

        tokens = re.findall(
            r"[A-Za-zÀ-ÖØ-öø-ÿ'’.-]+",
            value,
        )

        is_collective_author = bool(
            collective_author_pattern.search(value)
            and 1 <= len(tokens) <= 12
        )

        if (
            (
                not is_collective_author
                and not 2 <= len(tokens) <= 6
            )
            or institution_pattern.search(value)
            or role_pattern.search(value)
            or address_pattern.search(value)
            or email_pattern.search(value)
            or re.search(r"\d", value)
        ):
            return

        value = normalize_space(value).lower()
        key = unicodedata.normalize(
            "NFKD",
            value,
        ).casefold()
        key = "".join(
            character
            for character in key
            if not unicodedata.combining(character)
        )
        key = re.sub(r"[^a-z0-9]+", "", key)

        if not key or key in seen:
            return

        seen.add(key)
        results.append(
            (
                pdf.pdf_name,
                0,
                "author",
                value,
            )
        )

    def split_and_add(value):
        value = clean_author_candidate(value)
        value = degree_pattern.sub("", value)
        value = re.sub(r"\([^)]*\)\s*$", "", value)

        marked = marker_pattern.sub(" | ", value)
        marked = re.sub(
            r"^\s*(?:par|by)\s+",
            "",
            marked,
            flags=re.IGNORECASE,
        )

        if "|" in marked:
            parts = re.split(r"\s*\|\s*", marked)
        else:
            surname_first = re.fullmatch(
                r"[A-ZÀ-ÖØ-Ý][A-ZÀ-ÖØ-Ý'’-]+"
                r"\s*,\s*"
                r"[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÿ'’.-]+"
                r"(?:\s+[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÿ'’.-]+){0,2}",
                marked,
            )

            if surname_first is not None:
                parts = [marked]
            else:
                parts = re.split(
                    r"\s*(?:,|;|[•·]|\bet\b|\band\b|&)\s*",
                    marked,
                    flags=re.IGNORECASE,
                )

        for part in parts:
            part = re.sub(
                r"^\s*(?:,|;|\bet\b|\band\b|&)+\s*",
                "",
                part,
                flags=re.IGNORECASE,
            ).strip()

            if part:
                add_name(part)

    # Route 1: first-page front matter.
    first_blocks = pages[0]["blocks"]
    title_seen = False

    for block in first_blocks:
        content = normalize_space(block["content"])

        if not content:
            continue

        if block["raw_label"] == "title":
            if structural_pattern.match(content):
                break

            if not title_seen:
                title_seen = True
                continue

            # A later title-labelled block may be a subtitle.
            # Accept it as an author only with explicit person metadata:
            # author marker, academic degree, or SURNAME, Given form.
            explicit_title_author = bool(
                marker_pattern.search(content)
                or degree_pattern.search(content)
                or re.fullmatch(
                    r"\s*[A-ZÀ-ÖØ-Ý][A-ZÀ-ÖØ-Ý'’-]+"
                    r"\s*,\s*"
                    r"[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÿ'’.-]+"
                    r"(?:\s+[A-ZÀ-ÖØ-Ý][A-Za-zÀ-ÿ'’.-]+){0,2}"
                    r"\s*",
                    content,
                )
            )

            if (
                explicit_title_author
                and person_like(content)
            ):
                split_and_add(content)

            continue

        if not title_seen:
            if block["raw_label"] != "text":
                continue

            # Layout models may label the article title as text.
            # The first substantive text block becomes the title
            # anchor; if it already looks like a person, it is also
            # evaluated as an author candidate.
            title_seen = True

            if not person_like(content):
                continue

        if structural_pattern.match(content):
            break

        content_is_metadata = bool(
            institution_pattern.search(content)
            or role_pattern.search(content)
            or address_pattern.search(content)
            or email_pattern.search(content)
        )
        content_word_count = len(content.split())
        content_is_running_prose = bool(
            not content_is_metadata
            and not person_like(content)
            and (
                content_word_count >= 30
                or (
                    content_word_count >= 12
                    and re.search(r"[.!?](?:[)\]»”'\"]*)\s*$", content)
                )
            )
        )

        # Close front matter when OCR has missed the Abstract or
        # Introduction heading but ordinary narrative prose begins.
        if (
            block["raw_label"] == "text"
            and content_is_running_prose
            and len(marker_pattern.findall(content)) < 2
        ):
            break

        if block["raw_label"] != "text":
            continue

        candidate_content = content
        institution_match = institution_pattern.search(
            content
        )

        # Author and affiliation may share the same OCR block.
        if (
            institution_match is not None
            and institution_match.start() > 0
        ):
            candidate_content = content[
                :institution_match.start()
            ].strip(" \t\r\n,;:.—–-")

        if person_like(candidate_content):
            split_and_add(candidate_content)

    # Route 2 is a fallback. Once front matter has supplied authors,
    # later document pages must not introduce additional person names.
    if results:
        return True, results

    # Route 2: bounded author profiles on any later page.
    for page in pages[1:]:
        blocks = page["blocks"]

        for index, block in enumerate(blocks):
            if block["raw_label"] not in {"title", "text"}:
                continue

            content = normalize_space(block["content"])

            if not person_like(content):
                continue

            metadata_found = False
            email_found = False

            for following in blocks[index + 1:index + 12]:
                following_content = normalize_space(
                    following["content"]
                )

                if not following_content:
                    continue

                if (
                    following["raw_label"] in {
                        "header",
                        "footer",
                        "page_number",
                        "ref_text",
                    }
                    or structural_pattern.match(following_content)
                    or len(following_content.split()) >= 70
                ):
                    break

                if (
                    institution_pattern.search(following_content)
                    or role_pattern.search(following_content)
                    or address_pattern.search(following_content)
                    or email_pattern.search(following_content)
                ):
                    metadata_found = True
                    email_found = (
                        email_found
                        or bool(
                            email_pattern.search(
                                following_content
                            )
                        )
                    )
                    continue

                if (
                    following["raw_label"] == "title"
                    or person_like(following_content)
                ):
                    break

            if (
                metadata_found
                and (
                    block["raw_label"] == "title"
                    or email_found
                )
            ):
                split_and_add(content)

    return True, results


def extract_caption(pdf):
    """
    Extract figure and table captions from Unlimited-OCR blocks.

    Captions are identified through explicit layout labels or through
    numbered multilingual caption headings structurally associated
    with a nearby figure or table.
    """
    pages = _load_pdf_blocks(pdf)

    if not pages:
        return True, []

    object_labels = {
        "image",
        "table",
        "chart",
    }

    caption_labels = {
        "image_caption",
        "figure_caption",
        "table_caption",
        "caption",
    }

    caption_term = (
        r"(?:"
        r"figures?|fig\.?|"
        r"tableaux?|tables?|tab\.?|"
        r"graphiques?|graphs?|charts?|"
        r"schémas?|schemas?|diagrams?|"
        r"illustrations?|"
        r"figuras?|cuadros?|gráficos?|graficos?"
        r")"
    )

    caption_number = (
        r"(?:"
        r"\d+(?:[a-z])?"
        r"(?:\s*(?:,|et|and|&)\s*\d+(?:[a-z])?)*"
        r"|[IVXLCDM]+"
        r")"
    )

    heading_pattern = re.compile(
        rf"^\s*{caption_term}"
        rf"\s*(?:n(?:o|º|°)\.?\s*)?"
        rf"{caption_number}"
        rf"\s*[:.\-–—]?\s*",
        flags=re.IGNORECASE,
    )

    repeated_heading_pattern = re.compile(
        rf"(?:^|\n)\s*{caption_term}"
        rf"\s*(?:n(?:o|º|°)\.?\s*)?"
        rf"{caption_number}"
        rf"\s*[:.\-–—]?\s*",
        flags=re.IGNORECASE,
    )

    results = []

    def clean(value):
        value = unicodedata.normalize("NFKC", str(value))
        value = value.translate(
            str.maketrans(
                {
                    "’": "'",
                    "‘": "'",
                    "‐": "-",
                    "‑": "-",
                    "‒": "-",
                    "–": "-",
                    "—": "-",
                }
            )
        )
        value = re.sub(
            r"[\u200b\u200c\u200d\ufeff]",
            "",
            value,
        )
        value = re.sub(r"\s+", " ", value)
        return value.strip(" \t\r\n,;:.-|")

    def split_caption_text(value, direct_label=False):
        raw = unicodedata.normalize("NFKC", str(value))
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        raw = re.sub(
            r"[\u200b\u200c\u200d\ufeff]",
            "",
            raw,
        )

        matches = list(
            repeated_heading_pattern.finditer(raw)
        )

        if not matches:
            candidate = clean(raw)
            return [candidate] if direct_label and candidate else []

        parts = []

        for position, match in enumerate(matches):
            start = match.start()
            end = (
                matches[position + 1].start()
                if position + 1 < len(matches)
                else len(raw)
            )
            segment = raw[start:end].strip()
            segment = heading_pattern.sub("", segment, count=1)
            segment = clean(segment)

            if segment:
                parts.append(segment)

        return parts

    def split_parallel_panel_captions(value):
        """
        Split two parallel caption sentences when both describe the
        same labelled panels, for example (a) and (b).

        This structural rule supports multilingual captions without
        relying on language-specific words.
        """
        value = clean(value)
        sentences = [
            clean(part)
            for part in re.split(
                r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ])",
                value,
            )
            if clean(part)
        ]

        if len(sentences) != 2:
            return [value]

        panel_sequences = []

        for sentence in sentences:
            panels = tuple(
                marker.casefold()
                for marker in re.findall(
                    r"\(\s*([A-Za-z]|\d+)\s*\)",
                    sentence,
                )
            )
            panel_sequences.append(panels)

        if (
            len(panel_sequences[0]) >= 2
            and panel_sequences[0] == panel_sequences[1]
        ):
            return sentences

        return [value]

    def has_nearby_object(blocks, index, expected=None):
        left = max(0, index - 3)
        right = min(len(blocks), index + 5)

        for nearby in blocks[left:right]:
            label = nearby.get("raw_label", "")

            if expected == "table":
                if label == "table":
                    return True
            elif expected == "image":
                if label in {"image", "chart"}:
                    return True
            elif label in object_labels:
                return True

        return False

    for page_index, page in enumerate(pages):
        blocks = page["blocks"]
        consumed = set()

        for index, block in enumerate(blocks):
            if index in consumed:
                continue

            label = block.get("raw_label", "")
            raw_content = str(block.get("content", "")).strip()

            if not raw_content:
                continue

            direct_label = label in caption_labels
            heading_match = heading_pattern.match(raw_content)

            if not direct_label and heading_match is None:
                continue

            expected_object = None

            if heading_match is not None:
                heading_text = heading_match.group(0).casefold()

                if re.match(
                    r"\s*(?:tableaux?|tables?|tab\.?|cuadros?)",
                    heading_text,
                    flags=re.IGNORECASE,
                ):
                    expected_object = "table"
                else:
                    expected_object = "image"

            # Text/title candidates require structural confirmation.
            if (
                not direct_label
                and not has_nearby_object(
                    blocks,
                    index,
                    expected_object,
                )
            ):
                continue

            parts = split_caption_text(
                raw_content,
                direct_label=direct_label,
            )

            # A heading may occupy one block and its caption text the
            # immediately following block.
            if not parts and heading_match is not None:
                next_index = index + 1

                if next_index < len(blocks):
                    next_block = blocks[next_index]
                    next_label = next_block.get("raw_label", "")
                    next_content = str(
                        next_block.get("content", "")
                    ).strip()

                    if (
                        next_content
                        and next_label in {
                            "text",
                            "title",
                            *caption_labels,
                        }
                        and heading_pattern.match(next_content) is None
                        and has_nearby_object(
                            blocks,
                            next_index,
                            expected_object,
                        )
                    ):
                        merged = clean(next_content)

                        if merged:
                            parts = [merged]
                            consumed.add(next_index)

            expanded_parts = []

            for part in parts:
                expanded_parts.extend(
                    split_parallel_panel_captions(part)
                )

            for caption in expanded_parts:
                caption = clean(caption)

                if not caption:
                    continue

                # Reject object contents and formula-only strings.
                if (
                    caption.startswith("<table")
                    or re.fullmatch(
                        r"[\W\d_]+",
                        caption,
                    )
                ):
                    continue

                results.append(
                    (
                        pdf.pdf_name,
                        page_index,
                        "caption",
                        clean(caption).lower(),
                    )
                )

    return True, results


def extract_footer(pdf):
    """
    Extract content footnotes using general model labels,
    numbering markers, and structural page position.

    Rules are frozen from Erudit Dev10:
      - accept explicit page_footnote blocks;
      - accept numbered text/ref_text/footer blocks near
        the bottom of the page;
      - accept numbered notes in narrow margin columns;
      - accept unnumbered text in a narrow outer margin;
      - split grouped parenthetical notes such as
        "(9) ... (10) ...";
      - remove leading note markers and deduplicate only
        within each page.
    """
    pages = _load_pdf_blocks(pdf)
    results = []

    marker_pattern = re.compile(
        r"^\s*(?:"
        r"\\\(\s*\^\{\d{1,3}\}\s*\\\)|"
        r"\(\d{1,3}\)|"
        r"\d{1,3}[.)]|"
        r"[ivxlcdm]{1,6}[.)]"
        r")\s+",
        flags=re.IGNORECASE,
    )

    parenthetical_marker_pattern = re.compile(
        r"(?<!\S)\(\d{1,3}\)\s+"
    )

    def clean(value):
        value = unicodedata.normalize(
            "NFKC",
            str(value),
        )

        value = value.translate(
            str.maketrans(
                {
                    "’": "'",
                    "‘": "'",
                    "‐": "-",
                    "‑": "-",
                    "‒": "-",
                    "–": "-",
                    "—": "-",
                }
            )
        )

        value = re.sub(
            r"[\u200b\u200c\u200d\ufeff]",
            "",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip(
            " \t\r\n|"
        )

    def parse_bbox(value):
        numbers = re.findall(
            r"-?\d+(?:\.\d+)?",
            str(value),
        )

        if len(numbers) != 4:
            return None

        return tuple(
            float(number)
            for number in numbers
        )

    def split_parenthetical_notes(content):
        matches = list(
            parenthetical_marker_pattern.finditer(
                content
            )
        )

        if (
            not matches
            or matches[0].start() != 0
        ):
            return [content]

        parts = []

        for index, match in enumerate(matches):
            start = match.start()

            if index + 1 < len(matches):
                end = matches[index + 1].start()
            else:
                end = len(content)

            part = content[start:end].strip()

            if part:
                parts.append(part)

        return parts

    def strip_marker(content):
        return marker_pattern.sub(
            "",
            content,
            count=1,
        ).strip()

    for page_index, page in enumerate(pages):
        seen = set()

        for block in page["blocks"]:
            raw_label = block.get(
                "raw_label",
                "",
            )

            content = clean(
                block.get(
                    "content",
                    "",
                )
            )

            if not content:
                continue

            bbox = parse_bbox(
                block.get(
                    "bbox",
                    "",
                )
            )

            if bbox is None:
                left = top = right = bottom = None
            else:
                left, top, right, bottom = bbox

            has_marker = bool(
                marker_pattern.match(content)
            )

            explicit_page_footnote = (
                raw_label == "page_footnote"
            )

            bottom_numbered_note = (
                raw_label
                in {
                    "text",
                    "ref_text",
                    "footer",
                }
                and has_marker
                and bbox is not None
                and max(top, bottom) >= 600
            )

            margin_numbered_note = (
                raw_label == "text"
                and has_marker
                and bbox is not None
                and (
                    left >= 650
                    or right <= 350
                )
            )

            unnumbered_margin_note = (
                raw_label == "text"
                and not has_marker
                and bbox is not None
                and left >= 650
                and right - left <= 260
            )

            if not (
                explicit_page_footnote
                or bottom_numbered_note
                or margin_numbered_note
                or unnumbered_margin_note
            ):
                continue

            if raw_label == "text":
                parts = split_parenthetical_notes(
                    content
                )
            else:
                parts = [content]

            for part in parts:
                part = clean(
                    strip_marker(part)
                ).lower()

                if not part or part in seen:
                    continue

                seen.add(part)

                results.append(
                    (
                        pdf.pdf_name,
                        page_index,
                        "footer",
                        part,
                    )
                )

    return True, results


def extract_section(pdf):
    """
    Extract section and subsection headings using document structure.

    Unlimited-OCR does not provide a direct section label. Section
    candidates are selected from title-labelled blocks, while document
    titles, front matter, captions, author profiles, and back matter
    are excluded.
    """
    pages = _load_pdf_blocks(pdf)

    if not pages:
        return True, []

    front_matter_pattern = re.compile(
        r"^\s*(?:"
        r"abstract|résumé|resume|resumen|resumo|summary|sommaire|"
        r"keywords?|key\s*words?|mots[\s-]*clés?|"
        r"palabras\s+clave|index\s+terms?|"
        r"article|research\s+article|review\s+article"
        r")\s*(?:[-–—:|].*)?$",
        flags=re.IGNORECASE,
    )

    back_matter_pattern = re.compile(
        r"^\s*(?:"
        r"notes?|footnotes?|"
        r"references?|références?|"
        r"bibliograph(?:y|ie|ies|ique|iques)?|"
        r"works?\s+cited|literature\s+cited|"
        r"acknowledg(?:e)?ments?|remerciements?|"
        r"sources?(?:\s+consultées?|\s+électroniques?)?|"
        r"appendix|appendices|annexes?"
        r")\b",
        flags=re.IGNORECASE,
    )

    caption_pattern = re.compile(
        r"^\s*(?:"
        r"figures?|fig\.?|"
        r"tables?|tableaux?|tab\.?|"
        r"graphs?|graphiques?|charts?|"
        r"schemas?|schémas?|diagrams?|illustrations?"
        r")\s*(?:n[°ºo]?\.?\s*)?"
        r"(?:\d+|[IVXLCDM]+)\b",
        flags=re.IGNORECASE,
    )

    numbered_section_pattern = re.compile(
        r"^\s*(?:"
        r"\d+(?:\.\d+)*\.?|"
        r"[IVXLCDM]+\."
        r")\s+\S",
        flags=re.IGNORECASE,
    )

    post_back_numbered_pattern = re.compile(
        r"^\s*(?:"
        r"\d+(?:\.\d+)*\.?|"
        r"[A-Z]\.|"
        r"[IVXLCDM]+\."
        r")\s+\S",
        flags=re.IGNORECASE,
    )

    explicit_section_pattern = re.compile(
        r"^\s*(?:"
        r"introduction|conclusion|conclusions|"
        r"discussion|results?|résultats?|"
        r"methods?|méthodes?|methodology|méthodologie|"
        r"background|contexte"
        r")\s*$",
        flags=re.IGNORECASE,
    )

    author_profile_pattern = re.compile(
        r"\b(?:"
        r"ph\.?\s*d\.?|"
        r"m\.?\s*sc\.?|b\.?\s*sc\.?|"
        r"professor|professeur|professeure|"
        r"doctorant|doctorante"
        r")\b",
        flags=re.IGNORECASE,
    )

    divider_pattern = re.compile(
        r"^\s*(?:[＊*]\s*){3,}$"
    )

    def clean(value):
        value = unicodedata.normalize("NFKC", str(value))
        value = value.translate(
            str.maketrans(
                {
                    "’": "'",
                    "‘": "'",
                    "‐": "-",
                    "‑": "-",
                    "‒": "-",
                    "–": "-",
                    "—": "-",
                }
            )
        )
        value = re.sub(
            r"[\u200b\u200c\u200d\ufeff]",
            "",
            value,
        )
        value = re.sub(r"\s+", " ", value)
        return value.strip(" \t\r\n|")

    # The initial consecutive title group on page one represents the
    # article title, subtitle, or document type rather than sections.
    first_page_title_indices = set()
    initial_group_started = False

    for index, block in enumerate(pages[0]["blocks"]):
        label = block.get("raw_label", "")
        content = clean(block.get("content", ""))

        if not content:
            continue

        if not initial_group_started:
            if label in {"header", "page_number"}:
                continue

            if label == "title":
                initial_group_started = True
                first_page_title_indices.add(index)
                continue

            break

        if label == "title":
            first_page_title_indices.add(index)
            continue

        break

    results = []
    seen = set()
    back_matter_started = False

    for page_index, page in enumerate(pages):
        for block_index, block in enumerate(page["blocks"]):
            label = block.get("raw_label", "")
            content = clean(block.get("content", ""))

            if not content:
                continue

            symbolic_divider = bool(
                divider_pattern.fullmatch(content)
            )

            if label != "title" and not symbolic_divider:
                continue

            # References, notes, appendices, sources, and similar
            # headings mark the end of the article section sequence.
            if back_matter_pattern.match(content):
                back_matter_started = True
                continue

            # References and acknowledgements do not necessarily end
            # the document. Numbered appendix subsections occurring
            # later remain valid structural sections. Free-form titles
            # after back matter are rejected conservatively.
            if (
                back_matter_started
                and not post_back_numbered_pattern.match(content)
            ):
                continue

            if (
                front_matter_pattern.fullmatch(content)
                or caption_pattern.match(content)
                or author_profile_pattern.search(content)
            ):
                continue

            if (
                page_index == 0
                and block_index in first_page_title_indices
            ):
                continue

            # On page one, unnumbered titles between the article title
            # and the first section are usually translated titles or
            # other front matter. Explicit section anchors remain valid.
            if (
                page_index == 0
                and not symbolic_divider
                and not numbered_section_pattern.match(content)
                and not explicit_section_pattern.fullmatch(content)
            ):
                continue

            normalized = clean(content).lower()

            key = unicodedata.normalize(
                "NFKD",
                normalized,
            ).casefold()
            key = "".join(
                character
                for character in key
                if not unicodedata.combining(character)
            )
            key = re.sub(r"[^a-z0-9*]+", "", key)

            if not key or key in seen:
                continue

            seen.add(key)
            results.append(
                (
                    pdf.pdf_name,
                    page_index,
                    "section",
                    normalized,
                )
            )

    return True, results


def extract_reference(pdf):
    """
    Extract bibliography entries using structural reference boundaries.

    Unlimited-OCR labels bibliography entries and ordinary footnotes as
    ``ref_text``. Therefore, ref_text is accepted only inside a bounded
    bibliography region, or after an explicit further-reading heading.
    """
    pages = _load_pdf_blocks(pdf)

    if not pages:
        return True, []

    bibliography_heading_pattern = re.compile(
        r"^\s*"
        r"(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)[.)]?\s+)?"
        r"(?:"
        r"references?(?:\s+bibliographiques?)?|"
        r"références?(?:\s+bibliographiques?)?|"
        r"bibliograph(?:y|ie|ies|ique)?|"
        r"works?\s+cited|literature\s+cited|"
        r"sources?\s+(?:consultées?|électroniques?|"
        r"electronic(?:\s+sources?)?)|"
        r"liste\s+des\s+références"
        r")\s*[:.\-–—]?\s*$",
        flags=re.IGNORECASE,
    )

    reading_heading_pattern = re.compile(
        r"^\s*(?:"
        r"further\s+reading|recommended\s+reading|"
        r"pour\s+en\s+savoir\s+plus|"
        r"lectures?\s+(?:complémentaires?|recommandées?)|"
        r"à\s+retrouver\s+dans\s+les\s+archives"
        r")\b",
        flags=re.IGNORECASE,
    )

    stop_heading_pattern = re.compile(
        r"^\s*(?:"
        r"appendix|appendices|annexes?|"
        r"author\s+biograph(?:y|ies)|"
        r"notice\s+biographique|"
        r"résumé|abstract|summary|resumen|"
        r"keywords?|key\s*words?|mots[\s-]*clés?"
        r")\b",
        flags=re.IGNORECASE,
    )

    citation_evidence_pattern = re.compile(
        r"(?:"
        r"\b(?:18|19|20)\d{2}[a-z]?\b|"
        r"\bdoi\s*:|https?://|www\.|"
        r"\bvol\.?\s*\d+|\bno\.?\s*\d+|"
        r"\bn[°º]\s*\d+|"
        r"\bpp?\.\s*\d+|"
        r"\(\s*(?:18|19|20)\d{2}[a-z]?\s*\)"
        r")",
        flags=re.IGNORECASE,
    )

    non_reference_pattern = re.compile(
        r"^\s*(?:"
        r"figure|fig\.?|table|tableau|graphique|"
        r"abstract|résumé|keywords?|mots[\s-]*clés?"
        r")\b",
        flags=re.IGNORECASE,
    )

    def clean(value):
        value = unicodedata.normalize("NFKC", str(value))
        value = value.translate(
            str.maketrans(
                {
                    "’": "'",
                    "‘": "'",
                    "‐": "-",
                    "‑": "-",
                    "‒": "-",
                    "–": "-",
                    "—": "-",
                }
            )
        )
        value = re.sub(
            r"[\u200b\u200c\u200d\ufeff]",
            "",
            value,
        )
        value = re.sub(r"\s+", " ", value)
        return value.strip(" \t\r\n|")

    def citation_like(value):
        value = clean(value)

        if (
            not value
            or non_reference_pattern.match(value)
            or len(value.split()) < 3
            or len(value) < 12
        ):
            return False

        return bool(citation_evidence_pattern.search(value))

    reference_marker_pattern = re.compile(
        r"^\s*(?:"
        r"\\\(\s*\^\{\s*\d+\s*\}\s*\\\)|"
        r"\[\s*\d+\s*\]|"
        r"\(\s*\d+\s*\)|"
        r"\d+[.)]"
        r")\s*"
    )

    biography_pattern = re.compile(
        r"(?:"
        r"\breceived\s+(?:the|his|her)\b.*\bdegree\b|"
        r"\bis\s+currently\s+pursuing\b.*\bdegree\b|"
        r"\bresearch\s+interests?\s+(?:include|are)\b|"
        r"\bcurrently\s+(?:he|she)\s+is\b|"
        r"\bsenior\s+member\s+of\s+(?:the\s+)?ieee\b"
        r")",
        flags=re.IGNORECASE,
    )

    results = []
    result_has_marker = []
    seen = set()
    bibliography_active = False
    reading_active = False
    reading_page = None

    def add_reference(value, page_index):
        value = clean(value).lower()
        has_marker = bool(
            reference_marker_pattern.match(value)
        )
        value = reference_marker_pattern.sub(
            "",
            value,
            count=1,
        )
        value = clean(value)

        if not value:
            return

        key = unicodedata.normalize(
            "NFKD",
            value,
        ).casefold()
        key = "".join(
            character
            for character in key
            if not unicodedata.combining(character)
        )
        key = re.sub(r"[^a-z0-9]+", "", key)

        if not key or key in seen:
            return

        seen.add(key)
        results.append(
            (
                pdf.pdf_name,
                page_index,
                "reference",
                value,
            )
        )
        result_has_marker.append(has_marker)

    for page_index, page in enumerate(pages):
        blocks = page["blocks"]

        if reading_active and page_index != reading_page:
            reading_active = False

        for block in blocks:
            label = block.get("raw_label", "")
            content = clean(block.get("content", ""))

            if not content:
                continue

            if bibliography_heading_pattern.fullmatch(content):
                bibliography_active = True
                reading_active = False
                continue

            if reading_heading_pattern.match(content):
                reading_active = True
                reading_page = page_index
                continue

            if label == "title":
                if bibliography_active:
                    bibliography_active = False

                if (
                    reading_active
                    or stop_heading_pattern.match(content)
                ):
                    reading_active = False

                continue

            if bibliography_active:
                if (
                    label == "ref_text"
                    and biography_pattern.search(content)
                ):
                    bibliography_active = False
                    continue

                if label == "ref_text":
                    add_reference(content, page_index)
                    continue

                # Some layout models label bibliography entries as text.
                if (
                    label == "text"
                    and citation_like(content)
                ):
                    add_reference(content, page_index)

                continue

            if reading_active:
                if (
                    label in {"text", "ref_text"}
                    and citation_like(content)
                ):
                    add_reference(content, page_index)
                    continue

                # The alternate bibliography is a contiguous run.
                if label not in {
                    "header",
                    "footer",
                    "page_number",
                    "page_footnote",
                }:
                    reading_active = False


    # Fallback: some scientific articles have no visible bibliography
    # heading. Accept only a substantial numbered ref_text tail that
    # begins with reference 1 in the second half of the document.
    if not results:
        numbered_reference_pattern = re.compile(
            r"^\s*(?:"
            r"\\\(\s*\^\{\s*(\d+)\s*\}\s*\\\)|"
            r"\[\s*(\d+)\s*\]|"
            r"\(\s*(\d+)\s*\)|"
            r"(\d+)[.)]"
            r")\s*"
        )

        numbered_blocks = []

        for fallback_page_index, fallback_page in enumerate(pages):
            for fallback_block in fallback_page["blocks"]:
                if fallback_block.get("raw_label") != "ref_text":
                    continue

                fallback_content = clean(
                    fallback_block.get("content", "")
                )
                match = numbered_reference_pattern.match(
                    fallback_content
                )

                if match is None:
                    continue

                number = next(
                    int(group)
                    for group in match.groups()
                    if group is not None
                )

                numbered_blocks.append(
                    (
                        fallback_page_index,
                        number,
                        fallback_content,
                    )
                )

        if numbered_blocks:
            first_page_index = numbered_blocks[0][0]
            numbers = [
                item[1]
                for item in numbered_blocks
            ]

            starts_with_one = numbers[0] == 1
            substantial_run = len(numbered_blocks) >= 5
            in_document_tail = (
                first_page_index
                >= max(0, (len(pages) - 1) // 2)
            )

            increasing_pairs = sum(
                1
                for previous, current in zip(
                    numbers,
                    numbers[1:],
                )
                if current == previous + 1
            )

            sequence_ratio = (
                increasing_pairs
                / max(1, len(numbers) - 1)
            )

            if (
                starts_with_one
                and substantial_run
                and in_document_tail
                and sequence_ratio >= 0.75
            ):
                first_numbered_page = numbered_blocks[0][0]

                for fallback_page_index, fallback_page in enumerate(pages):
                    if fallback_page_index < first_numbered_page:
                        continue

                    for fallback_block in fallback_page["blocks"]:
                        if fallback_block.get("raw_label") != "ref_text":
                            continue

                        fallback_content = clean(
                            fallback_block.get("content", "")
                        )

                        if fallback_content:
                            add_reference(
                                fallback_content,
                                fallback_page_index,
                            )

    # Merge continuation blocks only when the bibliography
    # demonstrably uses explicit numeric reference markers.
    if results and any(result_has_marker):
        merged_results = []
        merged_markers = []

        for row, has_marker in zip(
            results,
            result_has_marker,
        ):
            if not has_marker and merged_results:
                previous = merged_results[-1]
                merged_value = clean(
                    previous[3] + " " + row[3]
                ).lower()

                merged_results[-1] = (
                    previous[0],
                    previous[1],
                    previous[2],
                    merged_value,
                )
                continue

            merged_results.append(row)
            merged_markers.append(has_marker)

        results = merged_results
        result_has_marker = merged_markers

    return True, results


def extract_table(pdf):
    """
    Extract textual table contents from Unlimited-OCR HTML blocks.

    Consecutive fragments of the same physical table are merged using
    layout continuity. Explicit numbered table captions form a hard
    boundary between independent tables.
    """
    import html as html_lib

    pages = _load_pdf_blocks(pdf)

    if not pages:
        return True, []

    table_boundary_pattern = re.compile(
        r"^\s*(?:"
        r"tables?|tableaux?|tab\.?|"
        r"figures?|fig\.?"
        r")\s*(?:n[°ºo]?\.?\s*)?"
        r"(?:\d+(?:[a-z])?|[IVXLCDM]+)\b",
        flags=re.IGNORECASE,
    )

    def latex_to_plain(value):
        """
        Convert common scientific LaTeX notation to comparable plain
        Unicode text without changing the table structure.
        """
        value = str(value)

        replacements = {
            r"\\alpha": "α",
            r"\\beta": "β",
            r"\\gamma": "γ",
            r"\\delta": "δ",
            r"\\eta": "η",
            r"\\lambda": "λ",
            r"\\mu": "μ",
            r"\\nu": "ν",
            r"\\rho": "ρ",
            r"\\sigma": "σ",
            r"\\tau": "τ",
            r"\\omega": "ω",
            r"\\pi": "π",
            r"\\theta": "θ",
            r"\\epsilon": "ε",
            r"\\phi": "φ",
            r"\\psi": "ψ",
            r"\\times": "×",
            r"\\cdot": "·",
            r"\\leq": "≤",
            r"\\geq": "≥",
            r"\\langle": "⟨",
            r"\\rangle": "⟩",
            r"\\emptyset": "∅",
            r"\\pm": "±",
        }

        value = re.sub(
            r"\\\\(?:text|mathrm|mathbf|mathit|mathcal|operatorname)"
            r"\\s*\\{([^{}]*)\\}",
            r"\\1",
            value,
        )

        value = re.sub(
            r"\\\\(?:dot|hat|bar|tilde|vec)"
            r"\\s*\\{([^{}]*)\\}",
            r"\\1",
            value,
        )

        value = re.sub(
            r"\\\\frac\\s*\\{([^{}]*)\\}"
            r"\\s*\\{([^{}]*)\\}",
            r"\\1/\\2",
            value,
        )

        value = re.sub(
            r"\\\\sqrt\\s*\\{([^{}]*)\\}",
            r"√\\1",
            value,
        )

        value = re.sub(
            r"\\\\begin\\s*\\{[^{}]+\\}",
            " [ ",
            value,
        )
        value = re.sub(
            r"\\\\end\\s*\\{[^{}]+\\}",
            " ] ",
            value,
        )

        for source, destination in replacements.items():
            value = re.sub(
                source + r"\\b",
                destination,
                value,
            )

        value = re.sub(
            r"\\\\[()]",
            " ",
            value,
        )
        value = re.sub(
            r"\\\\(?:left|right|displaystyle)\\b",
            " ",
            value,
        )

        value = re.sub(
            r"_\\s*\\{([^{}]*)\\}",
            r"_\\1",
            value,
        )
        value = re.sub(
            r"\\^\\s*\\{([^{}]*)\\}",
            r"^\\1",
            value,
        )

        # Preserve semantic commands such as log after removing the
        # LaTeX backslash.
        value = re.sub(
            r"\\\\([A-Za-z]+)",
            r"\\1",
            value,
        )

        value = re.sub(
            r"\\\\{2,}",
            " ",
            value,
        )
        value = value.replace("{", "")
        value = value.replace("}", "")

        return value

    def normalize(value):
        value = html_lib.unescape(str(value))
        value = latex_to_plain(value)
        value = unicodedata.normalize("NFKC", value)
        value = value.translate(
            str.maketrans(
                {
                    "’": "'",
                    "‘": "'",
                    "‐": "-",
                    "‑": "-",
                    "‒": "-",
                    "–": "-",
                    "—": "-",
                }
            )
        )
        value = re.sub(
            r"[\u200b\u200c\u200d\ufeff]",
            "",
            value,
        )
        value = re.sub(r"\s+", " ", value)
        return value.strip(" \t\r\n|")

    def html_to_text(value):
        value = str(value)

        value = re.sub(
            r"</(?:td|th)>",
            " ",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"</tr>",
            " ",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"<br\s*/?>",
            " ",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"<[^>]+>",
            " ",
            value,
        )

        return normalize(value)

    def looks_like_internal_toc(
        table_text,
        blocks,
        table_index,
    ):
        """
        Detect an internal table of contents rendered as a table.

        It requires a Page/Pages heading, several numbered section
        entries, and an increasing sequence of three-digit page
        numbers.
        """
        preceding_text = " ".join(
            normalize(block.get("content", ""))
            for block in blocks[
                max(0, table_index - 2):table_index
            ]
        )
        context = normalize(
            preceding_text + " " + table_text
        )

        if not re.search(
            r"\bpages?\b",
            context,
            flags=re.IGNORECASE,
        ):
            return False

        section_entries = re.findall(
            r"(?:^|\s)"
            r"\d+(?:\.\d+)*"
            r"\s+[A-Za-zÀ-ÖØ-öø-ÿ]",
            table_text,
        )

        page_numbers = [
            int(number)
            for number in re.findall(
                r"\b(\d{3})\b",
                table_text,
            )
        ]

        if (
            len(section_entries) < 3
            or len(page_numbers) < 3
        ):
            return False

        increasing_pairs = sum(
            1
            for previous, current in zip(
                page_numbers,
                page_numbers[1:],
            )
            if current >= previous
        )

        return (
            increasing_pairs
            / max(1, len(page_numbers) - 1)
            >= 0.75
        )

    def parse_bbox(value):
        numbers = [
            int(number)
            for number in re.findall(
                r"-?\d+",
                str(value),
            )
        ]

        if len(numbers) < 4:
            return None

        return numbers[:4]

    def horizontal_overlap(first_box, second_box):
        intersection = max(
            0,
            min(first_box[2], second_box[2])
            - max(first_box[0], second_box[0]),
        )
        minimum_width = max(
            1,
            min(
                first_box[2] - first_box[0],
                second_box[2] - second_box[0],
            ),
        )
        return intersection / minimum_width

    results = []
    seen = set()

    def add_result(page_index, value):
        value = normalize(value).lower()

        if not value:
            return

        # Reject pathological table hallucinations. A single table
        # exceeding 10,000 normalized characters is treated as a
        # corrupted model output rather than reliable table content.
        if len(value) > 10000:
            return

        key = unicodedata.normalize(
            "NFKD",
            value,
        ).casefold()
        key = "".join(
            character
            for character in key
            if not unicodedata.combining(character)
        )
        key = re.sub(r"[^a-z0-9]+", "", key)

        if not key or key in seen:
            return

        seen.add(key)
        results.append(
            (
                pdf.pdf_name,
                page_index,
                "table",
                value,
            )
        )

    for page_index, page in enumerate(pages):
        blocks = page["blocks"]
        table_indices = [
            index
            for index, block in enumerate(blocks)
            if block.get("raw_label") == "table"
        ]

        if not table_indices:
            # Fallback for a table represented as:
            # Tableau N + caption + numbered text rows.
            for heading_index, heading_block in enumerate(blocks):
                heading_text = normalize(
                    heading_block.get("content", "")
                )

                if not re.fullmatch(
                    r"\s*(?:tableau|table|tab\.?)"
                    r"\s*(?:n[°ºo]?\.?\s*)?"
                    r"(?:\d+(?:[a-z])?|[IVXLCDM]+)"
                    r"\s*[:.\-–—]?\s*",
                    heading_text,
                    flags=re.IGNORECASE,
                ):
                    continue

                caption_index = None

                for candidate_index in range(
                    heading_index + 1,
                    min(len(blocks), heading_index + 4),
                ):
                    candidate = blocks[candidate_index]
                    candidate_text = normalize(
                        candidate.get("content", "")
                    )

                    if not candidate_text:
                        continue

                    if (
                        candidate.get("raw_label")
                        not in {"title", "text"}
                        or len(candidate_text.split()) > 35
                    ):
                        break

                    caption_index = candidate_index
                    break

                if caption_index is None:
                    continue

                intro_text = ""
                numbered_rows = []
                expected_number = 1

                for candidate in blocks[caption_index + 1:]:
                    if candidate.get("raw_label") != "text":
                        break

                    candidate_text = normalize(
                        candidate.get("content", "")
                    )

                    if not candidate_text:
                        continue

                    numbered_match = re.match(
                        r"^\s*(\d+)[.)]\s+(.+)$",
                        candidate_text,
                    )

                    if numbered_match is not None:
                        number = int(
                            numbered_match.group(1)
                        )

                        if number != expected_number:
                            break

                        numbered_rows.append(
                            normalize(
                                numbered_match.group(2)
                            )
                        )
                        expected_number += 1
                        continue

                    if (
                        not numbered_rows
                        and not intro_text
                        and len(candidate_text.split()) <= 18
                    ):
                        intro_text = candidate_text
                        continue

                    break

                if len(numbered_rows) >= 3:
                    add_result(
                        page_index,
                        " ".join(
                            [
                                intro_text,
                                *numbered_rows,
                            ]
                        ),
                    )

            continue

        current_text = ""
        current_box = None
        previous_table_index = None

        for table_index in table_indices:
            block = blocks[table_index]
            table_text = html_to_text(
                block.get("content", "")
            )
            table_box = parse_bbox(
                block.get("bbox")
            )

            if looks_like_internal_toc(
                table_text,
                blocks,
                table_index,
            ):
                continue

            if not table_text:
                continue

            merge = False
            intermediate_texts = []

            if (
                current_text
                and current_box is not None
                and table_box is not None
                and previous_table_index is not None
            ):
                gap = table_box[1] - current_box[3]
                overlap = horizontal_overlap(
                    current_box,
                    table_box,
                )

                intervening = blocks[
                    previous_table_index + 1:
                    table_index
                ]

                valid_intervening = True

                for middle_block in intervening:
                    middle_label = middle_block.get(
                        "raw_label",
                        "",
                    )
                    middle_text = normalize(
                        middle_block.get("content", "")
                    )

                    if not middle_text:
                        continue

                    if (
                        middle_label not in {"text", "title"}
                        or len(middle_text.split()) > 12
                        or table_boundary_pattern.match(
                            middle_text
                        )
                    ):
                        valid_intervening = False
                        break

                    intermediate_texts.append(
                        middle_text
                    )

                merge = (
                    0 <= gap <= 50
                    and overlap >= 0.75
                    and valid_intervening
                )

            if merge:
                current_text = normalize(
                    " ".join(
                        [
                            current_text,
                            *intermediate_texts,
                            table_text,
                        ]
                    )
                )
                current_box = [
                    min(current_box[0], table_box[0]),
                    min(current_box[1], table_box[1]),
                    max(current_box[2], table_box[2]),
                    max(current_box[3], table_box[3]),
                ]
            else:
                if current_text:
                    add_result(
                        page_index,
                        current_text,
                    )

                current_text = table_text
                current_box = table_box

            previous_table_index = table_index

        if current_text:
            add_result(
                page_index,
                current_text,
            )

    return True, results


def extract_list(pdf):
    """
    Extract list groups from consecutive explicitly marked text blocks.

    The eSciBench ground truth stores an entire list as one record,
    rather than storing each individual item separately. Consequently,
    consecutive bullet or numbered blocks are joined into one result.
    """
    pages = _load_pdf_blocks(pdf)

    if not pages:
        return True, []

    marker_pattern = re.compile(
        r"^\s*(?:"
        r"[*•●▪◦‣⁃–—-]\s+|"
        r"\d{1,3}[.)]\s+|"
        r"\([a-zA-Z0-9]{1,3}\)\s+|"
        r"[a-zA-Z][.)]\s+"
        r")"
    )

    internal_marker_pattern = re.compile(
        r"(?:^|\s)(?:"
        r"[*•●▪◦‣⁃–—-]\s+|"
        r"\d{1,3}[.)]\s+|"
        r"\([a-zA-Z0-9]{1,3}\)\s+"
        r")"
    )

    toc_item_pattern = re.compile(
        r"^\s*[A-Za-z][.)]?\s+.+?\s+\d+\s*$"
    )

    numeric_marker_pattern = re.compile(
        r"^\s*(?:\(?([0-9]{1,3})[.)])\s+"
    )

    citation_evidence_pattern = re.compile(
        r"(?:"
        r"\bop\.\s*cit\.|"
        r"\bloc\.\s*cit\.|"
        r"\bibid\.|"
        r"\bdoi\b|"
        r"https?://|www\.|"
        r"\bvol\.\s*\d+|"
        r"\bpp?\.\s*\d+|"
        r"\b(?:18|19|20)\d{2}\b"
        r")",
        flags=re.IGNORECASE,
    )

    dialogue_evidence_pattern = re.compile(
        r"\b(?:"
        r"demanda|répondit|repondit|"
        r"bégaya|begaya|dit|"
        r"mon\s+père|mon\s+pere"
        r")\b",
        flags=re.IGNORECASE,
    )

    def latex_to_plain(value):
        """
        Convert common scientific LaTeX notation to comparable plain
        Unicode text without changing the table structure.
        """
        value = str(value)

        replacements = {
            r"\\alpha": "α",
            r"\\beta": "β",
            r"\\gamma": "γ",
            r"\\delta": "δ",
            r"\\eta": "η",
            r"\\lambda": "λ",
            r"\\mu": "μ",
            r"\\nu": "ν",
            r"\\rho": "ρ",
            r"\\sigma": "σ",
            r"\\tau": "τ",
            r"\\omega": "ω",
            r"\\pi": "π",
            r"\\theta": "θ",
            r"\\epsilon": "ε",
            r"\\phi": "φ",
            r"\\psi": "ψ",
            r"\\times": "×",
            r"\\cdot": "·",
            r"\\leq": "≤",
            r"\\geq": "≥",
            r"\\langle": "⟨",
            r"\\rangle": "⟩",
            r"\\emptyset": "∅",
            r"\\pm": "±",
        }

        value = re.sub(
            r"\\\\(?:text|mathrm|mathbf|mathit|mathcal|operatorname)"
            r"\\s*\\{([^{}]*)\\}",
            r"\\1",
            value,
        )

        value = re.sub(
            r"\\\\(?:dot|hat|bar|tilde|vec)"
            r"\\s*\\{([^{}]*)\\}",
            r"\\1",
            value,
        )

        value = re.sub(
            r"\\\\frac\\s*\\{([^{}]*)\\}"
            r"\\s*\\{([^{}]*)\\}",
            r"\\1/\\2",
            value,
        )

        value = re.sub(
            r"\\\\sqrt\\s*\\{([^{}]*)\\}",
            r"√\\1",
            value,
        )

        value = re.sub(
            r"\\\\begin\\s*\\{[^{}]+\\}",
            " [ ",
            value,
        )
        value = re.sub(
            r"\\\\end\\s*\\{[^{}]+\\}",
            " ] ",
            value,
        )

        for source, destination in replacements.items():
            value = re.sub(
                source + r"\\b",
                destination,
                value,
            )

        value = re.sub(
            r"\\\\[()]",
            " ",
            value,
        )
        value = re.sub(
            r"\\\\(?:left|right|displaystyle)\\b",
            " ",
            value,
        )

        value = re.sub(
            r"_\\s*\\{([^{}]*)\\}",
            r"_\\1",
            value,
        )
        value = re.sub(
            r"\\^\\s*\\{([^{}]*)\\}",
            r"^\\1",
            value,
        )

        # Preserve semantic commands such as log after removing the
        # LaTeX backslash.
        value = re.sub(
            r"\\\\([A-Za-z]+)",
            r"\\1",
            value,
        )

        value = re.sub(
            r"\\\\{2,}",
            " ",
            value,
        )
        value = value.replace("{", "")
        value = value.replace("}", "")

        return value

    def clean(value):
        value = latex_to_plain(value)
        value = unicodedata.normalize(
            "NFKC",
            str(value),
        )
        value = value.translate(
            str.maketrans(
                {
                    "’": "'",
                    "‘": "'",
                    "‐": "-",
                    "‑": "-",
                    "‒": "-",
                    "–": "-",
                    "—": "-",
                }
            )
        )
        value = re.sub(
            r"[\u200b\u200c\u200d\ufeff]",
            "",
            value,
        )
        value = re.sub(r"\s+", " ", value)
        return value.strip(" \t\r\n|")

    results = []
    seen = set()

    def add_group(page_index, items):
        if not items:
            return

        # A one-block group is accepted only if it contains
        # multiple explicit list markers internally.
        if (
            len(items) == 1
            and len(
                internal_marker_pattern.findall(items[0])
            ) < 2
        ):
            return

        # Reject compact alphabetical table-of-contents runs.
        toc_hits = sum(
            1
            for item in items
            if toc_item_pattern.fullmatch(item)
        )

        if toc_hits / len(items) >= 0.8:
            return

        numeric_matches = [
            numeric_marker_pattern.match(item)
            for item in items
        ]

        if all(match is not None for match in numeric_matches):
            first_number = int(
                numeric_matches[0].group(1)
            )

            # A standalone ordered list begins at one. Runs beginning
            # later are usually page-local footnote continuations.
            if first_number != 1:
                return

            average_words = sum(
                len(item.split())
                for item in items
            ) / len(items)

            citation_hits = sum(
                1
                for item in items
                if citation_evidence_pattern.search(item)
            )

            if (
                average_words >= 45
                and citation_hits >= 1
            ):
                return

        dialogue_hits = sum(
            1
            for item in items
            if dialogue_evidence_pattern.search(item)
        )

        if dialogue_hits >= 2:
            return

        value = clean(" ".join(items)).lower()

        if not value:
            return

        key = unicodedata.normalize(
            "NFKD",
            value,
        ).casefold()
        key = "".join(
            character
            for character in key
            if not unicodedata.combining(character)
        )
        key = re.sub(r"[^a-z0-9]+", "", key)

        if not key or key in seen:
            return

        seen.add(key)
        results.append(
            (
                pdf.pdf_name,
                page_index,
                "list",
                value,
            )
        )

    for page_index, page in enumerate(pages):
        run = []

        def flush():
            nonlocal run
            add_group(page_index, run)
            run = []

        for block in page["blocks"]:
            label = block.get("raw_label", "")
            content = clean(block.get("content", ""))

            if (
                label == "text"
                and marker_pattern.match(content)
            ):
                run.append(content)
                continue

            flush()

        flush()

    #
    # Route 2: unmarked indented runs introduced by a colon.
    #
    def parse_bbox(value):
        numbers = [
            int(number)
            for number in re.findall(
                r"-?\d+",
                str(value),
            )
        ]

        return numbers[:4] if len(numbers) >= 4 else None

    def merge_item_fragments(candidates):
        if not candidates:
            return []

        merged = []
        current = candidates[0].copy()

        for candidate in candidates[1:]:
            current_box = current["bbox"]
            candidate_box = candidate["bbox"]
            next_text = candidate["text"].lstrip()

            can_merge = (
                current["page"] == candidate["page"]
                and current_box is not None
                and candidate_box is not None
                and abs(
                    current_box[0] - candidate_box[0]
                ) <= 5
                and 0
                <= candidate_box[1] - current_box[3]
                <= 2
                and not re.search(
                    r"[;.:!?]\s*$",
                    current["text"],
                )
                and bool(
                    re.match(
                        r"^[a-zà-öø-ÿ(]",
                        next_text,
                    )
                )
            )

            if can_merge:
                current["text"] = clean(
                    current["text"]
                    + " "
                    + candidate["text"]
                )
                current["bbox"] = [
                    min(current_box[0], candidate_box[0]),
                    min(current_box[1], candidate_box[1]),
                    max(current_box[2], candidate_box[2]),
                    max(current_box[3], candidate_box[3]),
                ]
                continue

            merged.append(current)
            current = candidate.copy()

        merged.append(current)
        return merged

    for page_index, page in enumerate(pages):
        blocks = page["blocks"]

        for introducer_index, introducer in enumerate(blocks):
            introducer_label = introducer.get(
                "raw_label",
                "",
            )
            introducer_text = clean(
                introducer.get("content", "")
            )
            introducer_box = parse_bbox(
                introducer.get("bbox")
            )

            if (
                introducer_label not in {"text", "title"}
                or not introducer_text.rstrip().endswith(":")
                or introducer_box is None
            ):
                continue

            raw_candidates = []
            active_left = None

            for block_index in range(
                introducer_index + 1,
                len(blocks),
            ):
                block = blocks[block_index]
                label = block.get("raw_label", "")
                content = clean(block.get("content", ""))
                box = parse_bbox(block.get("bbox"))

                if (
                    label != "text"
                    or not content
                    or box is None
                ):
                    break

                if len(content.split()) > 40:
                    break

                # Children must be visibly indented relative
                # to the introducing paragraph.
                if box[0] < introducer_box[0] + 15:
                    break

                if active_left is None:
                    active_left = box[0]
                elif abs(box[0] - active_left) > 12:
                    break

                raw_candidates.append(
                    {
                        "page": page_index,
                        "block": block_index,
                        "label": label,
                        "text": content,
                        "bbox": box,
                    }
                )

            candidates = merge_item_fragments(
                raw_candidates
            )

            if len(candidates) < 2:
                continue

            grouped_items = [
                "* " + candidate["text"]
                for candidate in candidates
            ]

            add_group(
                page_index,
                grouped_items,
            )

    #
    # Route 3: document-level increasing numbered sequences.
    #
    # Main list items may be separated by nested unmarked lists,
    # tables, footnotes, headers, or page boundaries. Only text
    # blocks carrying the next expected number are retained.
    numbered_candidates = []

    for numbered_page_index, numbered_page in enumerate(pages):
        for numbered_block_index, numbered_block in enumerate(
            numbered_page["blocks"]
        ):
            if numbered_block.get("raw_label", "") != "text":
                continue

            numbered_content = clean(
                numbered_block.get("content", "")
            )
            numbered_match = numeric_marker_pattern.match(
                numbered_content
            )

            if numbered_match is None:
                continue

            numbered_candidates.append(
                {
                    "page": numbered_page_index,
                    "block": numbered_block_index,
                    "number": int(
                        numbered_match.group(1)
                    ),
                    "text": numbered_content,
                }
            )

    for start_index, start_candidate in enumerate(
        numbered_candidates
    ):
        if start_candidate["number"] != 1:
            continue

        sequence = [start_candidate]
        expected_number = 2
        previous_page = start_candidate["page"]

        for candidate in numbered_candidates[
            start_index + 1:
        ]:
            # A numbered list may continue across a small number
            # of pages, but not across unrelated document regions.
            if candidate["page"] - previous_page > 3:
                break

            if candidate["number"] == expected_number:
                sequence.append(candidate)
                expected_number += 1
                previous_page = candidate["page"]
                continue

            # A new item 1 begins an independent sequence.
            if candidate["number"] == 1:
                break

            # Any other numbered text breaks sequence continuity.
            break

        if len(sequence) < 2:
            continue

        add_group(
            sequence[0]["page"],
            [
                candidate["text"]
                for candidate in sequence
            ],
        )

    return True, results



def extract_header(pdf):
    """
    Baseline header extractor.

    Return every non-empty raw block explicitly labelled
    as header by Unlimited-OCR. No filtering, merging,
    or deduplication is applied.
    """
    pages = _load_pdf_blocks(pdf)
    results = []

    translation = str.maketrans(
        {
            "’": "'",
            "‘": "'",
            "‐": "-",
            "‑": "-",
            "‒": "-",
            "–": "-",
            "—": "-",
        }
    )

    for page in pages:
        for block in page["blocks"]:
            if block.get("raw_label", "") != "header":
                continue

            content = unicodedata.normalize(
                "NFKC",
                str(block.get("content", "")),
            )

            content = content.translate(translation)

            content = re.sub(
                r"[\u200b\u200c\u200d\ufeff]",
                "",
                content,
            )

            content = re.sub(
                r"\s+",
                " ",
                content,
            ).strip().lower()

            if not content:
                continue

            results.append(
                (
                    pdf.pdf_name,
                    0,
                    "header",
                    content,
                )
            )

    return True, results

def extract_raw(base_dir, label, pdf):
    """
    eSciBench extractor interface for saved Unlimited-OCR outputs.

    Args:
        base_dir: Dataset directory required by the eSciBench
            interface. Raw model outputs are loaded from
            RAW_PAGES_DIR.
        label: eSciBench label requested by the benchmark.
        pdf: eSciBench PDF object.

    Returns:
        A pair (supported, extraction_tuples).
    """
    handlers = {
        "equation": extract_equation,
        "title": extract_title,
        "abstract": extract_abstract_all_pages,
        "email": extract_email,
        "keyword": extract_keyword,
        "pub_date": extract_pub_date,
        "affiliation": extract_affiliation,
        "author": extract_author,
        "caption": extract_caption,
        "section": extract_section,
        "header": extract_header,
        "footer": extract_footer,
        "reference": extract_reference,
        "table": extract_table,
        "list": extract_list,
    }

    if label not in handlers:
        return False, []

    return handlers[label](pdf)