"""
Clean OLMOCR wrapper for eSciBench.

Built from scratch.
No previous OLMOCR wrapper logic is used here.

Final evaluation copy locked to the fixed Test96 dataset.

Expected extractor interface:
    extract_raw(base_dir, label, pdf)
        -> (flag, extraction_tuple)

Each extraction tuple item must be:
    (pdf_name, page, label, extracted_text)
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from benchmark.normalisation import normalize_string


# ---------------------------------------------------------------------------
# Development paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/scratch/nasimb/escibench_project/OlmOcr_orginal")

TEST96_DATA_DIR = (PROJECT_ROOT / "test96_escibench").resolve()

TEST96_RAW_JSONL = (
    PROJECT_ROOT
    / "raw_test96_escibench"
    / "results"
    / "olmocr_test96.jsonl"
).resolve()


# ---------------------------------------------------------------------------
# OLMOCR raw-output loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_olmocr_jsonl(jsonl_path_str):
    """
    Load an OLMOCR JSONL file and index its records by source-PDF stem.

    The canonical raw fields used are:
        record["text"]
        record["metadata"]["Source-File"]
        record["attributes"]["pdf_page_numbers"]
    """
    jsonl_path = Path(jsonl_path_str)

    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f"OLMOCR raw JSONL not found: {jsonl_path}"
        )

    records = {}

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            metadata = record.get("metadata")
            attributes = record.get("attributes")
            text = record.get("text")

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Invalid metadata at JSONL line {line_number}"
                )

            if not isinstance(attributes, dict):
                raise ValueError(
                    f"Invalid attributes at JSONL line {line_number}"
                )

            source_file = metadata.get("Source-File")

            if not isinstance(source_file, str) or not source_file:
                raise ValueError(
                    f"Missing Source-File at JSONL line {line_number}"
                )

            if not isinstance(text, str):
                raise ValueError(
                    f"Invalid text at JSONL line {line_number}"
                )

            page_ranges = attributes.get("pdf_page_numbers")

            if not isinstance(page_ranges, list):
                raise ValueError(
                    f"Missing pdf_page_numbers at JSONL line {line_number}"
                )

            pdf_stem = Path(source_file).stem

            if pdf_stem in records:
                raise ValueError(
                    f"Duplicate OLMOCR record for PDF: {pdf_stem}"
                )

            records[pdf_stem] = record

    return records


def _get_olmocr_record(base_dir, pdf):
    """
    Return the OLMOCR raw record corresponding to one eSciBench PDF.

    Final-evaluation safety guard:
    only the fixed Test96 dataset is accepted here.
    """
    resolved_base_dir = Path(base_dir).resolve()

    if resolved_base_dir != TEST96_DATA_DIR:
        raise RuntimeError(
            "Final evaluation wrapper is locked to the fixed Test96 dataset. "
            f"Received base_dir: {resolved_base_dir}"
        )

    records = _load_olmocr_jsonl(str(TEST96_RAW_JSONL))

    pdf_stem = Path(pdf.pdf_name).stem

    if pdf_stem not in records:
        raise KeyError(
            f"No OLMOCR raw record found for PDF: {pdf.pdf_name}"
        )

    return records[pdf_stem]


def _get_olmocr_pages(record):
    """
    Split one OLMOCR full-document text into page texts using the
    official OLMOCR pdf_page_numbers character offsets stored in the raw record.

    Returned items are:
        (olmocr_page_number, page_text)

    These OLMOCR page numbers are only internal parsing information.
    eSciBench JSON ground truth uses page=0, and official alignment groups by
    tool/pdf_name/label rather than page.
    """
    text = record["text"]
    page_ranges = record["attributes"]["pdf_page_numbers"]

    pages = []
    previous_end = 0

    for item in page_ranges:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(
                f"Invalid pdf_page_numbers entry: {item!r}"
            )

        start_char, end_char, page_number = item

        if not all(
            isinstance(value, int)
            for value in (start_char, end_char, page_number)
        ):
            raise ValueError(
                f"Non-integer pdf_page_numbers entry: {item!r}"
            )

        if start_char != previous_end:
            raise ValueError(
                "Non-contiguous OLMOCR page offsets: "
                f"expected start {previous_end}, got {start_char}"
            )

        if start_char < 0 or end_char < start_char or end_char > len(text):
            raise ValueError(
                f"Invalid OLMOCR page range: {item!r}"
            )

        pages.append(
            (page_number, text[start_char:end_char])
        )

        previous_end = end_char

    if previous_end != len(text):
        raise ValueError(
            "OLMOCR page offsets do not cover the complete raw text: "
            f"{previous_end} != {len(text)}"
        )

    return pages


# ---------------------------------------------------------------------------
# eSciBench extractor interface
# ---------------------------------------------------------------------------

def _extract_title(record):
    """
    Extract the scientific-article title from OLMOCR raw text.

    Generic structural rule:
    - inspect only the first OLMOCR page;
    - take the first non-empty blank-line-delimited text block at the top;
    - preserve every line belonging to that block, allowing multi-line titles;
    - join internal title lines with spaces;
    - apply the official eSciBench normalize_string function.

    No filename-, journal-, corpus-, or document-specific rule is used.
    """
    pages = _get_olmocr_pages(record)

    if not pages:
        return []

    _, first_page_text = pages[0]

    blocks = [
        block.strip()
        for block in first_page_text.split("\n\n")
        if block.strip()
    ]

    if not blocks:
        return []

    title_text = " ".join(
        line.strip()
        for line in blocks[0].splitlines()
        if line.strip()
    )

    if not title_text:
        return []

    title_text = normalize_string(title_text)

    if not title_text:
        return []

    return [title_text]



def _extract_author(record):
    """
    Generic AUTHOR_V4 for scientific front matter.

    - first OLMOCR page only;
    - scan blocks after title and before article body;
    - stop author collection when affiliation/contact metadata begins;
    - remove affiliation markers, ORCID/correspondence metadata and credentials;
    - split common multi-author separators;
    - validate candidates using generic person-name structure rather than
      document/corpus-specific patterns;
    - normalize with official eSciBench normalize_string.
    """
    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    _, first_page_text = pages[0]

    blocks = [
        block.strip()
        for block in first_page_text.split("\n\n")
        if block.strip()
    ]

    if len(blocks) < 2:
        return []

    body_start = re.compile(
        r"^\s*(?:abstract\b|keywords?\b|key\s+words?\b|"
        r"(?:[ivxlcdm]+[.\s]+)?introduction\b)",
        flags=re.IGNORECASE,
    )

    affiliation_contact = re.compile(
        r"\b(?:"
        r"university|université|department|département|faculty|school|"
        r"institute|institution|laboratory|laboratoire|centre|center|"
        r"college|hospital|academy|research\s+(?:center|centre|group)|"
        r"division|unit|faculty|campus|"
        r"address|street|road|avenue|boulevard|postal|postcode|zip|"
        r"e-?mail|telephone|phone|fax|"
        r"corresponding\s+author|correspondence|"
        r"orcid"
        r")\b",
        flags=re.IGNORECASE,
    )

    publication_metadata = re.compile(
        r"\b(?:"
        r"doi|arxiv|preprint|submitted|received|accepted|published|"
        r"copyright|journal|conference|proceedings|online\s+version|"
        r"manuscript|article\s+history|publication"
        r")\b",
        flags=re.IGNORECASE,
    )

    url_re = re.compile(r"https?://|www\.", flags=re.IGNORECASE)
    orcid_re = re.compile(
        r"\b(?:orcid\s*:?\s*)?\d{4}-\d{4}-\d{4}-[\dX]{4}\b",
        flags=re.IGNORECASE,
    )

    author_fragments = []

    for block in blocks[1:]:
        if body_start.search(block):
            break

        lines = [
            line.strip()
            for line in block.replace("\r", "\n").splitlines()
            if line.strip()
        ]

        if not lines:
            continue

        prefix = []

        for line in lines:
            # Numbered/symbolic affiliation line after an author line.
            if prefix and re.match(
                r"^\s*(?:\^?\{?\d+\}?|[*†‡§¶])\s*\S",
                line
            ):
                break

            if (
                "@" in line
                or url_re.search(line)
                or affiliation_contact.search(line)
                or orcid_re.search(line)
            ):
                break

            if publication_metadata.search(line):
                prefix = []
                break

            prefix.append(line)

        if prefix:
            author_fragments.append("\n".join(prefix))

    authors = []

    particles = {
        "al", "bin", "da", "de", "del", "della", "der", "di", "dos",
        "du", "la", "le", "van", "von", "ten", "ter"
    }

    suffixes = {
        "jr", "jr.", "sr", "sr.", "ii", "iii", "iv"
    }

    credentials_re = re.compile(
        r"(?:,\s*|\s+)(?:ph\.?d\.?|m\.?d\.?|m\.?sc\.?|b\.?sc\.?|"
        r"msc|bsc|meng|meng\.?|eng\.?|prof\.?|dr\.?)\s*$",
        flags=re.IGNORECASE,
    )

    membership_re = re.compile(
        r",?\s*\b(?:student\s+member|graduate\s+student\s+member|"
        r"associate\s+member|senior\s+member|fellow|member)"
        r"\s*,?\s*IEEE\b",
        flags=re.IGNORECASE,
    )

    def clean_candidate(text):
        text = orcid_re.sub("", text)
        text = re.sub(
            r"\b(?:corresponding\s+author|correspondence)\b.*$",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = membership_re.sub("", text)

        # Remove trailing academic/professional credentials repeatedly.
        previous = None
        while previous != text:
            previous = text
            text = credentials_re.sub("", text).strip()

        # Leading footnote symbols.
        text = re.sub(r"^[*†‡§¶]+\s*", "", text)

        # Trailing affiliation markers:
        # Name1 / Name1,2 / Name a,b / Name* / Name1,*
        text = re.sub(
            r"(?:"
            r"\s*\^?\{?\d+(?:\s*,\s*\d+)*\}?"
            r"(?:\s*,?\s*[*†‡§¶]+)?"
            r"|"
            r"(?:\s*,\s*|\s+)[a-z](?:\s*,\s*[a-z])*"
            r"(?:\s*,?\s*[*†‡§¶]+)?"
            r"|"
            r"\s*,?\s*[*†‡§¶]+"
            r")$",
            "",
            text,
            flags=re.IGNORECASE,
        )

        return text.strip(" ,;|")

    def plausible_name(candidate):
        if not candidate:
            return False

        if (
            "@" in candidate
            or ":" in candidate
            or url_re.search(candidate)
            or affiliation_contact.search(candidate)
            or publication_metadata.search(candidate)
            or orcid_re.search(candidate)
        ):
            return False

        # Digits remaining after marker cleanup usually indicate metadata.
        if re.search(r"\d", candidate):
            return False

        raw_tokens = candidate.split()

        # Scientific personal names normally contain at least two components.
        if len(raw_tokens) < 2 or len(raw_tokens) > 10:
            return False

        strong_name_tokens = 0
        lexical_name_tokens = 0

        for raw_token in raw_tokens:
            token = raw_token.strip("(),[]{}")

            if not token:
                return False

            low = token.casefold()

            # Lower-case surname particles are legitimate name components.
            if low in particles:
                lexical_name_tokens += 1
                continue

            # Conventional suffixes are legitimate only after a name.
            if low in suffixes:
                lexical_name_tokens += 1
                continue

            # Initials: J. / J.P. / A.-B.
            if re.fullmatch(
                r"(?:[^\W\d_]\.)+(?:-[^\W\d_]\.?)?",
                token,
                flags=re.UNICODE,
            ):
                first_alpha = next(
                    (ch for ch in token if ch.isalpha()),
                    ""
                )
                if not first_alpha or not first_alpha.isupper():
                    return False

                strong_name_tokens += 1
                lexical_name_tokens += 1
                continue

            # Ordinary personal-name component, including:
            # Jean-Pierre / O'Connor / García / D'Arcy.
            if re.fullmatch(
                r"[^\W\d_]+(?:[-'’][^\W\d_]+)*\.?",
                token,
                flags=re.UNICODE,
            ):
                core = token.rstrip(".")
                first_alpha = next(
                    (ch for ch in core if ch.isalpha()),
                    ""
                )

                # Main name-bearing components must start with an uppercase
                # letter (or be entirely uppercase). Lower-case forms are
                # accepted only when they are recognised particles above.
                if not first_alpha or not (
                    first_alpha.isupper() or core.isupper()
                ):
                    return False

                strong_name_tokens += 1
                lexical_name_tokens += 1
                continue

            return False

        if lexical_name_tokens != len(raw_tokens):
            return False

        # Require at least two real name-bearing components.
        if strong_name_tokens < 2:
            return False

        # A final period is valid for an initial or conventional suffix,
        # but ordinary prose punctuation is not.
        if candidate.endswith(("?", "!", ":")):
            return False

        if candidate.endswith("."):
            last = raw_tokens[-1].strip("(),[]{}").casefold()
            if (
                last not in suffixes
                and not re.fullmatch(
                    r"(?:[^\W\d_]\.)+",
                    raw_tokens[-1].strip("(),[]{}"),
                    flags=re.UNICODE,
                )
            ):
                return False

        return True

    for fragment in author_fragments:
        # Generic author separators. Commas are deliberately not used as a
        # universal separator because they can occur inside "Surname, Given".
        parts = re.split(
            r"\n+|;\s*|\s+\band\b\s+|\s*&\s*|[•·]\s*|\s{2,}",
            fragment,
            flags=re.IGNORECASE,
        )

        # Generic comma-separated author-list rescue.
        # Split on commas only when every comma-delimited component is itself
        # independently plausible as a complete person name. This avoids
        # blindly breaking conventional "Surname, Given" name formatting.
        expanded_parts = []

        for part in parts:
            comma_parts = [
                clean_candidate(x)
                for x in part.split(",")
                if clean_candidate(x)
            ]

            if (
                len(comma_parts) >= 2
                and all(plausible_name(x) for x in comma_parts)
            ):
                expanded_parts.extend(comma_parts)
            else:
                expanded_parts.append(part)

        parts = expanded_parts

        for part in parts:
            candidate = clean_candidate(part)

            if not plausible_name(candidate):
                continue

            candidate = normalize_string(candidate)

            if candidate and candidate not in authors:
                authors.append(candidate)

    return authors


def _extract_affiliation(record):
    """
    Generic AFFILIATION_V6 for scholarly front matter.

    General structure:
    - inspect only page 0;
    - use the author region as the anchor for affiliation interpretation;
    - map conventional author markers (1, 2, a, b, ...) to matching
      affiliation markers;
    - merge continuation/address lines into the affiliation they belong to;
    - if several unmarked authors share one unmarked affiliation, emit that
      affiliation once per author;
    - preserve multiple affiliations when marker or structural boundaries
      distinguish them;
    - extract scholarly membership/status metadata attached to author names;
    - never synthesize affiliation text absent from the raw OLMOCR output;
    - normalize all emitted values with official eSciBench normalize_string.

    No filename-, institution-, journal-, country-, corpus-, or document-specific
    rule is used.
    """
    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    _, first_page_text = pages[0]

    blocks = [
        block.strip()
        for block in first_page_text.split("\n\n")
        if block.strip()
    ]
    if len(blocks) < 2:
        return []

    body_start = re.compile(
        r"^\s*(?:abstract\b|keywords?\b|key\s+words?\b|"
        r"(?:[ivxlcdm]+[.\s]+)?introduction\b)",
        flags=re.IGNORECASE,
    )

    affiliation_cue = re.compile(
        r"\b(?:"
        r"university|université|college|school|faculty|"
        r"department|département|institute|institution|"
        r"laboratory|laboratoire|centre|center|hospital|clinic|"
        r"academy|observatory|foundation|"
        r"research\s+(?:group|centre|center)|"
        r"corporation|company"
        r")\b",
        flags=re.IGNORECASE,
    )

    contact_metadata = re.compile(
        r"\b(?:"
        r"e-?mail|telephone|phone|fax|orcid|doi|arxiv|preprint|"
        r"received|accepted|submitted|published|copyright|"
        r"online\s+version"
        r")\b",
        flags=re.IGNORECASE,
    )

    membership_status = re.compile(
        r"\b(?:"
        r"graduate\s+student\s+member|student\s+member|"
        r"associate\s+member|senior\s+member|member|fellow"
        r")\s+[A-Z][A-Z0-9.-]{1,}\b",
        flags=re.IGNORECASE,
    )

    leading_marker = re.compile(
        r"^\s*(?P<marker>\d+|[a-z]|[*†‡§¶])"
        r"(?:[\s.,:;)\]]+|(?=[A-ZÀ-ÖØ-Þ]))"
        r"\s*(?P<text>.+)$"
    )

    def split_author_expressions(line):
        return [
            part.strip(" ,;")
            for part in re.split(
                r";\s*|\s+\band\b\s+|\s{2,}",
                line,
                flags=re.IGNORECASE,
            )
            if part.strip(" ,;")
        ]

    def trailing_author_markers(part):
        """
        Return all conventional numeric/letter affiliation markers attached
        to an author expression.

        Supports common scholarly forms such as:
        Name1
        Name1,2
        Name a
        Name a,b
        Name1,*
        while avoiding interpretation of the final letter of a surname as
        an affiliation marker.
        """
        markers = []

        # Numeric marker sequence may be attached directly to the name.
        m = re.search(
            r"(?P<markers>\d+(?:\s*,\s*\d+)*)"
            r"(?:\s*,?\s*[*†‡§¶]+)?$",
            part,
        )
        if m:
            markers.extend(
                marker.strip().lower()
                for marker in m.group("markers").split(",")
                if marker.strip()
            )
            return markers

        # Letter markers require a separator before the first marker.
        m = re.search(
            r"(?:,\s*|\s+)"
            r"(?P<markers>[a-z](?:\s*,\s*[a-z])*)"
            r"(?:\s*,?\s*[*†‡§¶]+)?$",
            part,
        )
        if m:
            markers.extend(
                marker.strip().lower()
                for marker in m.group("markers").split(",")
                if marker.strip()
            )
            return markers

        # Standalone scholarly symbol markers may be attached directly
        # to an author name: Name*, Name†, Name‡, ...
        m = re.search(
            r"(?P<markers>[*†‡§¶]+)$",
            part,
        )
        if m:
            markers.extend(list(m.group("markers")))

        return markers

    detected = []
    author_marker_counts = {}
    last_unmarked_author_count = 0

    for block in blocks[1:]:
        if body_start.search(block):
            break

        lines = [
            line.strip()
            for line in block.replace("\r", "\n").splitlines()
            if line.strip()
        ]
        if not lines:
            continue

        # Ignore blocks that are entirely contact/publication metadata.
        if all(
            "@" in line
            or contact_metadata.search(line)
            or re.search(r"https?://|www\.", line, flags=re.IGNORECASE)
            for line in lines
        ):
            continue

        first_line = lines[0]

        # Scholarly status may be attached to any author line inside the
        # front-matter author region, not necessarily only the first line.
        # Scan the author-like prefix of the block until affiliation/contact
        # material begins.
        for status_line in lines:
            if (
                affiliation_cue.search(status_line)
                or contact_metadata.search(status_line)
                or "@" in status_line
                or re.search(r"https?://|www\\.", status_line, flags=re.IGNORECASE)
            ):
                break

            for match in membership_status.finditer(status_line):
                detected.append({
                    "marker": None,
                    "text": match.group(0),
                    "repeat": 1,
                    "kind": "status",
                })

        marker_match = leading_marker.match(first_line)
        first_is_affiliation = bool(
            affiliation_cue.search(first_line)
            or (
                marker_match is not None
                and affiliation_cue.search(marker_match.group("text"))
            )
        )

        author_count_here = 0
        local_author_markers = []

        if (
            not first_is_affiliation
            and not contact_metadata.search(first_line)
            and "@" not in first_line
        ):
            for part in split_author_expressions(first_line):
                # Remove scholarly status before deciding whether the
                # remainder is a plausible author name.
                cleaned = membership_status.sub("", part).strip(" ,;")

                words = [
                    word
                    for word in cleaned.split()
                    if re.search(r"[^\W\d_]", word, flags=re.UNICODE)
                ]

                if 2 <= len(words) <= 12:
                    author_count_here += 1

                    markers = trailing_author_markers(cleaned)
                    for marker in markers:
                        local_author_markers.append(marker)
                        author_marker_counts[marker] = (
                            author_marker_counts.get(marker, 0) + 1
                        )

            if author_count_here and not local_author_markers:
                last_unmarked_author_count = author_count_here

        current = None
        start_index = 1 if author_count_here else 0

        for line in lines[start_index:]:
            if (
                "@" in line
                or re.search(r"https?://|www\.", line, flags=re.IGNORECASE)
                or contact_metadata.search(line)
            ):
                if current is not None:
                    detected.append(current)
                    current = None
                continue

            marker_match = leading_marker.match(line)

            # A line beginning with a marker already observed on an author
            # is an affiliation candidate even when the organization name
            # itself does not contain one of our lexical affiliation cues.
            if (
                marker_match is not None
                and marker_match.group("marker").lower() in author_marker_counts
            ):
                if current is not None:
                    detected.append(current)

                current = {
                    "marker": marker_match.group("marker").lower(),
                    "text": marker_match.group("text").strip(" ,;"),
                    "repeat": 1,
                    "kind": "affiliation",
                }
                continue

            # A normal lexical affiliation starts a new affiliation unit.
            if affiliation_cue.search(line):
                if current is not None:
                    detected.append(current)

                current = {
                    "marker": None,
                    "text": line.strip(" ,;"),
                    "repeat": 1,
                    "kind": "affiliation",
                }
                continue

            # Merge address/location/organizational continuation lines into
            # the current affiliation until a structural boundary is reached.
            if current is not None:
                current["text"] += " " + line.strip(" ,;")

        if current is not None:
            detected.append(current)

    affiliation_items = [
        item for item in detected
        if item.get("kind") == "affiliation"
    ]

    # Generic shared-affiliation rule:
    # if an unmarked author group is followed by exactly one unmarked
    # affiliation, that affiliation applies to every author in the group.
    unmarked_affiliations = [
        item for item in affiliation_items
        if item["marker"] is None
    ]

    if (
        last_unmarked_author_count > 1
        and len(unmarked_affiliations) == 1
    ):
        unmarked_affiliations[0]["repeat"] = last_unmarked_author_count

    emitted = []

    for item in detected:
        value = normalize_string(item["text"])
        if not value:
            continue

        marker = item["marker"]

        if marker is not None and marker in author_marker_counts:
            repeat = author_marker_counts[marker]
        else:
            repeat = item.get("repeat", 1)

        for _ in range(max(1, repeat)):
            emitted.append(value)

    return emitted



def _extract_email(record):
    """
    Generic EMAIL_V2 for scholarly front matter.

    General scientific-document structures supported:
    - inspect only page 0;
    - ordinary email syntax: name@domain.tld;
    - OCR whitespace/newline around ``@`` and ``.``;
    - common scholarly obfuscation: [at]/(at) and [dot]/(dot);
    - optional ``mailto:`` prefix;
    - grouped scholarly-address notation such as
      ``{alice,bob}@example.edu``;
    - preserve occurrence multiplicity;
    - never infer an address from author names or affiliations;
    - never use document-, author-, institution-, or domain-specific rules;
    - normalize emitted values with official eSciBench normalize_string.
    """
    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    _, first_page_text = pages[0]

    # Generic OCR / scholarly-email normalization.
    work = first_page_text

    # Common explicit obfuscations used in papers and author contact blocks.
    work = re.sub(r"(?i)\[\s*at\s*\]|\(\s*at\s*\)", "@", work)
    work = re.sub(r"(?i)\[\s*dot\s*\]|\(\s*dot\s*\)", ".", work)

    # OCR may insert spaces or line breaks around email separators.
    work = re.sub(r"\s*@\s*", "@", work)
    work = re.sub(
        r"(?<=[A-Za-z0-9])\s*\.\s*(?=[A-Za-z0-9])",
        ".",
        work,
    )

    emitted = []

    # Scholarly grouped notation, e.g. {alice,bob}@example.edu
    grouped_pattern = re.compile(
        r"(?<![\w.+-])"
        r"(?:mailto:\s*)?"
        r"\{\s*(?P<locals>"
        r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
        r"(?:\s*,\s*[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+)+"
        r")\s*\}"
        r"@"
        r"(?P<domain>[A-Z0-9-]+(?:\.[A-Z0-9-]+)+)",
        flags=re.IGNORECASE,
    )

    consumed = []
    for match in grouped_pattern.finditer(work):
        domain = match.group("domain")
        locals_part = match.group("locals")

        for local in re.split(r"\s*,\s*", locals_part):
            value = normalize_string(f"{local}@{domain}")
            if value:
                emitted.append(value)

        consumed.append(match.span())

    # Standard individual email syntax.
    email_pattern = re.compile(
        r"(?<![\w.+-])"
        r"(?:mailto:\s*)?"
        r"(?P<email>"
        r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
        r"@"
        r"[A-Z0-9-]+(?:\.[A-Z0-9-]+)+"
        r")",
        flags=re.IGNORECASE,
    )

    for match in email_pattern.finditer(work):
        # Do not additionally emit the partial match inside grouped notation.
        if any(a <= match.start() < b for a, b in consumed):
            continue

        value = match.group("email").strip(
            " \t\r\n<>[](){}.,;:"
        )

        value = normalize_string(value)
        if value:
            emitted.append(value)

    return emitted



def _extract_keyword(record):
    """
    Generic KEYWORD_V1 for scholarly front matter.

    General structure:
    - inspect only page 0;
    - detect conventional scholarly headings:
      Keyword, Keywords, Key words, Index Terms;
    - accept content on the same line or immediately following lines;
    - stop when another major scholarly section begins;
    - split explicit keyword lists on comma, semicolon, or bullet separators;
    - never infer keywords from title, abstract, or article body;
    - no document-, topic-, corpus-, or vocabulary-specific rules;
    - normalize every emitted keyword with official eSciBench normalize_string.
    """
    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    _, first_page_text = pages[0]

    heading_re = re.compile(
        r"(?im)^[ \t]*"
        r"(?:keywords?|key[ \t]+words?|index[ \t]+terms?)"
        r"[ \t]*(?:[:.—–-][ \t]*)?"
        r"(?P<rest>.*)$"
    )

    stop_re = re.compile(
        r"(?i)^[ \t]*"
        r"(?:abstract|introduction|background|"
        r"[ivxlcdm]+[.)]?[ \t]+introduction)"
        r"\b"
    )

    lines = first_page_text.splitlines()
    emitted = []

    for i, line in enumerate(lines):
        m = heading_re.match(line)
        if not m:
            continue

        parts = []

        rest = m.group("rest").strip()
        if rest:
            parts.append(rest)

        j = i + 1
        while j < len(lines):
            nxt = lines[j].strip()

            if not nxt:
                break

            if heading_re.match(lines[j]) or stop_re.match(lines[j]):
                break

            parts.append(nxt)
            j += 1

        text = " ".join(parts).strip()
        if not text:
            continue

        # Explicit list separators commonly used in scientific papers.
        items = re.split(
            r"\s*(?:;|,|•|\u2022)\s*|\s+—\s+",
            text,
        )

        for item in items:
            item = item.strip(" \t\r\n.;,:-–—")
            value = normalize_string(item)
            if value:
                emitted.append(value)

    return emitted



def _extract_pub_date(record):
    """
    Generic scholarly publication-date extraction from OLMOCR raw text.

    Evidence hierarchy:
    1. explicit publication cues;
    2. publication metadata lines in front matter;
    3. strongly isolated front-matter date lines.

    Editorial lifecycle/version dates are rejected.
    No corpus-, journal-, filename-, author-, or date-specific rules.
    """
    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    _, first_page_text = pages[0]
    lines = [x.strip() for x in first_page_text.splitlines() if x.strip()]
    if not lines:
        return []

    publication_cue = re.compile(
        r"""(?ix)\b(?:
            published(?:\s+(?:online|electronically|in\s+print|ahead\s+of\s+print))?
            |first\s+published(?:\s+online)?
            |online\s+first
            |first\s+online
            |publication\s+date
            |date\s+of\s+publication
            |date\s+published
            |online\s+publication
            |electronic\s+publication
            |electronically\s+published
            |publication\s+online
            |e[-\s]?published
            |e[-\s]?publication
            |epub
            |available\s+online
            |online\s+ahead\s+of\s+print
            |issue\s+date
            |publication\s+year
            |issued
            |print\s+publication
        )\b"""
    )

    reject_cue = re.compile(
        r"""(?ix)\b(?:
            received
            |revised
            |accepted
            |submitted
            |resubmitted
            |manuscript\s+received
            |revision\s+received
            |preprint
            |pre[-\s]?print
            |manuscript\s+version
            |author\s+version
            |draft
            |revision
        )\b"""
    )

    metadata_cue = re.compile(
        r"""(?ix)\b(?:
            journal
            |volume
            |vol\.?
            |issue
            |number
            |no\.?
            |doi
            |issn
            |copyright
            |©
            |article
            |publisher
        )\b"""
    )

    body_start = re.compile(
        r"""(?ix)^\s*(?:
            abstract
            |keywords?
            |1[\.\s]+introduction
            |introduction
            |background
        )\b"""
    )

    month = (
        r"(?:"
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
        r")"
    )

    date_re = re.compile(
        rf"""(?ix)
        (?:
            \b(?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?
            \s+{month}\s*,?\s+\d{{4}}\b
            |
            \b{month}\s+
            (?:0?[1-9]|[12][0-9]|3[01])(?:st|nd|rd|th)?
            ,?\s+\d{{4}}\b
            |
            \b\d{{4}}[-/.](?:0?[1-9]|1[0-2])[-/.]
            (?:0?[1-9]|[12][0-9]|3[01])\b
            |
            \b(?:0?[1-9]|[12][0-9]|3[01])[-/.]
            (?:0?[1-9]|1[0-2])[-/.]\d{{4}}\b
            |
            \b{month}\s+\d{{4}}\b
            |
            \b(?:19|20)\d{{2}}\b
        )
        """
    )

    candidates = []

    # Keep only front matter.
    front = []
    for line in lines[:60]:
        if body_start.match(line):
            break
        front.append(line)

    # --------------------------------------------------------------
    # Tier 1: explicit publication cue.
    # --------------------------------------------------------------
    for i, line in enumerate(front):
        if reject_cue.search(line):
            continue

        cue = publication_cue.search(line)
        if not cue:
            continue

        tail = line[cue.end():]

        bad_after = reject_cue.search(tail)
        if bad_after:
            tail = tail[:bad_after.start()]

        m = date_re.search(tail)

        if not m and i + 1 < len(front):
            nxt = front[i + 1]
            if not reject_cue.search(nxt):
                m = date_re.search(nxt)

        if m:
            candidates.append((100, i, m.group(0)))

    # --------------------------------------------------------------
    # Tier 2: date embedded in publication metadata.
    # --------------------------------------------------------------
    for i, line in enumerate(front[:30]):
        if reject_cue.search(line):
            continue
        if not metadata_cue.search(line):
            continue

        m = date_re.search(line)
        if m:
            candidates.append((70, i, m.group(0)))

    # --------------------------------------------------------------
    # Tier 3: isolated date near the very top of page 1.
    # Must look like metadata, not prose.
    # --------------------------------------------------------------
    for i, line in enumerate(front[:15]):
        if reject_cue.search(line):
            continue

        m = date_re.search(line)
        if not m:
            continue

        stripped = line.strip()

        # Strong isolation: short line or date occupies most of the line.
        compact = re.sub(r"\s+", " ", stripped)
        date_txt = m.group(0)

        if len(compact) <= 45:
            candidates.append((50, i, date_txt))
            continue

        ratio = len(date_txt) / max(len(compact), 1)
        if ratio >= 0.45:
            candidates.append((45, i, date_txt))

    if not candidates:
        return []

    # Highest-confidence candidate first; then earliest in front matter.
    candidates.sort(key=lambda x: (-x[0], x[1]))

    seen = set()
    emitted = []

    for _, _, raw_date in candidates:
        value = normalize_string(raw_date)
        if not value:
            continue

        key = value.casefold()
        if key in seen:
            continue

        seen.add(key)
        emitted.append(value)

        # PUB_DATE is a singular benchmark field.
        break

    return emitted


def _extract_abstract(record):
    """
    Generic ABSTRACT_V1 for scholarly articles.

    General scientific-document structure only:
    - inspect page 0 only;
    - recognize common multilingual abstract headings;
    - accept heading and abstract on the same line or following lines;
    - merge OCR-broken lines and consecutive abstract paragraphs;
    - tolerate multi-column OCR reading order when abstract text is emitted
      as consecutive fragments;
    - stop at common front-matter/section boundaries;
    - never infer an abstract from arbitrary article body text;
    - no document-, journal-, corpus-, topic-, or language-specific content;
    - normalize with official eSciBench normalize_string.
    """
    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    _, first_page_text = pages[0]
    lines = first_page_text.splitlines()

    # Common abstract labels in scholarly publishing.
    abstract_heading = re.compile(
        r"""(?ix)^\s*
        (?:[*_]{1,3}\s*)?
        (?:
            abstract
            |résumé
            |resume
            |resumen
            |zusammenfassung
            |riassunto
            |resumo
            |samenvatting
        )
        (?:\s*[*_]{1,3})?
        \s*(?:[:.\-–—]\s*)?
        (?P<rest>.*?)
        \s*$
        """
    )

    # Generic scholarly boundaries marking the end of abstract/front matter.
    stop_heading = re.compile(
        r"""(?ix)^\s*
        (?:[*_]{1,3}\s*)?
        (?:
            key[\s-]*words?
            |keywords?
            |index[\s-]+terms?
            |subject[\s-]+terms?
            |introduction
            |background
            |materials?\s+and\s+methods?
            |methods?
            |acknowledg(?:e)?ments?
            |references?
            |bibliography
            |correspondence
            |author\s+information
            |affiliations?
            |doi
            |received
            |revised
            |accepted
            |published
            |[ivxlcdm]+[.)]?\s+introduction
            |\d+(?:\.\d+)*[.)]?\s+introduction
        )
        \b
        """
    )

    emitted = []

    for i, line in enumerate(lines):
        m = abstract_heading.match(line)
        if not m:
            continue

        fragments = []

        rest = m.group("rest").strip()
        if rest:
            fragments.append(rest)

        j = i + 1
        empty_run = 0

        while j < len(lines):
            raw_line = lines[j]
            nxt = raw_line.strip()

            if abstract_heading.match(raw_line):
                # A repeated abstract heading can occur in duplicated OCR or
                # multilingual metadata. Do not merge a new labelled abstract
                # into the current one.
                break

            if stop_heading.match(raw_line):
                break

            if not nxt:
                # Abstracts can legitimately contain paragraph breaks.
                # Keep crossing a single blank line, but two consecutive
                # blank lines indicate a stronger block boundary.
                empty_run += 1
                if empty_run >= 2:
                    break
                j += 1
                continue

            empty_run = 0
            fragments.append(nxt)
            j += 1

        if not fragments:
            continue

        # Merge OCR line wraps / column fragments into one benchmark abstract.
        text = " ".join(fragments)
        text = re.sub(r"\s+", " ", text).strip()

        # Remove harmless Markdown emphasis around the extracted block.
        text = re.sub(r"^\s*[*_]{1,3}\s*", "", text)
        text = re.sub(r"\s*[*_]{1,3}\s*$", "", text)

        value = normalize_string(text)
        if value:
            emitted.append(value)

    if emitted:
        return emitted

    # Generic fallback for scholarly papers where the abstract has no explicit
    # "Abstract" heading. A common layout is:
    # title -> authors/affiliations -> substantial prose abstract -> first section.
    #
    # We only accept a substantial prose block immediately before the first
    # major section heading. This avoids guessing from arbitrary body text.
    section_heading = re.compile(
        r"""(?ix)^\s*
        (?:
            [ivxlcdm]+\s*[.)]?\s+
            |\d+(?:\.\d+)*\s*[.)]?\s+
        )?
        (?:
            abstract
            |introduction
            |background
            |materials?\s+and\s+methods?
            |methods?
            |related\s+work
            |literature\s+review
        )
        \s*$
        """
    )

    # Split page 0 into blank-line-delimited blocks while preserving structure.
    blocks = [
        re.sub(r"\s+", " ", b).strip()
        for b in re.split(r"\n\s*\n+", first_page_text)
        if b.strip()
    ]

    if not blocks:
        return []

    first_section_idx = None
    for idx, block in enumerate(blocks):
        if section_heading.match(block):
            first_section_idx = idx
            break

    if first_section_idx is None or first_section_idx < 2:
        return []

    # Examine blocks between front matter and the first body section.
    candidates = blocks[1:first_section_idx]

    # Work backwards: an unlabelled abstract is normally the substantial
    # prose block closest to the first section heading.
    for block in reversed(candidates):
        # Reject obvious metadata/contact/affiliation-style blocks.
        if re.search(
            r"""(?ix)\b(?:
                university|department|institute|laboratory|centre|center|
                faculty|school|college|hospital|email|e-mail|doi|
                corresponding\s+author|received|accepted|revised|published
            )\b""",
            block,
        ):
            continue

        words = re.findall(r"\b[\w'-]+\b", block, flags=re.UNICODE)
        sentence_marks = len(re.findall(r"[.!?](?:\s|$)", block))

        # Generic scholarly-prose requirements:
        # substantial length + more than one sentence-like unit.
        if len(words) < 50:
            continue
        if sentence_marks < 2:
            continue

        value = normalize_string(block)
        if value:
            return [value]

    return []




def _extract_section(record):
    """
    Generic SECTION_V2 for scholarly articles.

    General scientific-document structure only:
    - scan all pages;
    - support Arabic, Roman and letter-based hierarchical numbering;
    - support Markdown headings;
    - support common multilingual scholarly headings;
    - support generic numbered/custom section and subsection titles;
    - preserve actual OCR page number;
    - reject front matter, captions, references, equations, URLs and prose;
    - no document-, journal-, corpus-, filename- or Test96-specific rules.
    """
    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    known_heading = re.compile(
        r"""(?ix)^
        (?:
            abstract
            |introduction
            |background
            |motivation
            |overview
            |related\s+(?:work|works|research)
            |previous\s+work
            |literature\s+review
            |state\s+of\s+the\s+art
            |theoretical\s+(?:background|framework|model)
            |conceptual\s+framework
            |preliminaries
            |definitions?
            |notation
            |problem\s+(?:statement|definition|formulation)
            |model
            |framework
            |approach
            |proposed\s+(?:method|approach|model|framework)
            |materials?
            |materials?\s+and\s+methods?
            |methods?
            |methodology
            |data
            |dataset
            |datasets
            |data\s+and\s+methods?
            |experimental\s+(?:setup|design|method|methods|procedure|procedures|results?)
            |experimental\s+evaluation
            |experiments?
            |implementation
            |evaluation
            |analysis
            |results?
            |findings
            |discussion
            |results?\s+and\s+discussion
            |discussion\s+and\s+conclusions?
            |conclusions?
            |conclusion\s+and\s+future\s+work
            |concluding\s+remarks
            |summary
            |future\s+(?:work|research|directions?)
            |limitations?
            |implications?
            |acknowledg(?:e)?ments?
            |funding
            |conflict(?:s)?\s+of\s+interest
            |competing\s+interests?
            |author\s+contributions?
            |data\s+availability
            |supplementary\s+(?:material|materials|information)
            |appendix
            |appendices
            |references?
            |bibliography

            |contexte
            |état\s+de\s+l['’]art
            |etat\s+de\s+l['’]art
            |cadre\s+théorique
            |cadre\s+theorique
            |matériel
            |materiel
            |matériels
            |materiels
            |méthode
            |methode
            |méthodes
            |methodes
            |matériels?\s+et\s+méthodes?
            |materiels?\s+et\s+methodes?
            |données
            |donnees
            |résultats?
            |resultats?
            |analyse
            |discussion
            |conclusions?
            |perspectives
            |remerciements?
            |annexe
            |annexes
            |références?
            |references?
            |bibliographie

            |introducción
            |introduccion
            |antecedentes
            |estado\s+del\s+arte
            |marco\s+teórico
            |marco\s+teorico
            |materiales?
            |métodos?
            |metodos?
            |materiales?\s+y\s+métodos?
            |materiales?\s+y\s+metodos?
            |datos
            |resultados?
            |análisis
            |analisis
            |discusión
            |discusion
            |conclusiones?
            |trabajo\s+futuro
            |agradecimientos?
            |referencias?
            |bibliografía
            |bibliografia

            |einleitung
            |hintergrund
            |stand\s+der\s+forschung
            |theoretischer\s+rahmen
            |material
            |methoden
            |material(?:ien)?\s+und\s+methoden
            |daten
            |ergebnisse
            |analyse
            |diskussion
            |schlussfolgerungen?
            |ausblick
            |danksagung
            |anhang
            |literatur

            |introduzione
            |stato\s+dell['’]arte
            |quadro\s+teorico
            |materiali
            |metodi
            |materiali\s+e\s+metodi
            |dati
            |risultati
            |analisi
            |discussione
            |conclusioni
            |lavori\s+futuri
            |ringraziamenti
            |appendice
            |bibliografia

            |introdução
            |introducao
            |fundamentação\s+teórica
            |fundamentacao\s+teorica
            |materiais
            |métodos?
            |metodos?
            |materiais\s+e\s+métodos?
            |materiais\s+e\s+metodos?
            |dados
            |resultados?
            |análise
            |analise
            |discussão
            |discussao
            |conclusões
            |conclusoes
            |trabalhos?\s+futuros?
            |agradecimentos?
            |apêndice
            |apendice
            |referências
            |referencias
        )
        $"""
    )

    front_matter = re.compile(
        r"""(?ix)^
        (?:
            résumé|resume|resumen|riassunto|resumo|samenvatting|
            keywords?|key[\s-]*words?|index\s+terms?|subject\s+terms?|
            authors?|affiliations?|correspondence|corresponding\s+author|
            doi|orcid|received|accepted|revised|submitted|resubmitted|
            published|publication\s+date|available\s+online|preprint
        )
        \b"""
    )

    caption_like = re.compile(
        r"""(?ix)^\s*
        (?:fig(?:ure)?|table|tab\.?|scheme|algorithm|listing|equation|eq\.?)
        \s*[\.:#-]?\s*
        [A-Z]?\d+
        \b"""
    )

    # Numbering commonly used in scientific papers:
    # 1, 1.1, 2.3.1, I, II, IV, A, B, A.1, etc.
    appendix_heading = re.compile(
        r"""(?ix)^\s*
        appendix
        \s+
        (?P<label>[A-Z]|\d+|[IVXLCDM]+)
        \s*[:.\-–—]?\s*
        (?P<title>.+?)
        \s*$
        """
    )

    numbered = re.compile(
        r"""(?x)^\s*
        (?P<num>
            \d+(?:\.\d+){0,5}
            |[IVXLCDM]+
            |[A-Z](?:\.\d+)*
        )
        \s*(?:[.\:\-–—])?\s+
        (?P<title>.+?)
        \s*$"""
    )

    emitted = []
    seen_body_structure = False
    in_contents = False

    for page_no, page_text in pages:
        lines = page_text.splitlines()

        for line_idx, raw_line in enumerate(lines):
            original_line = raw_line.strip()
            line = original_line
            if not line:
                continue

            # Generic table-of-contents region detection.
            if re.fullmatch(
                r"(?i)(?:contents|table\s+of\s+contents)",
                line,
            ):
                in_contents = True
                continue

            if in_contents:
                # Dotted-leader/page-number rows are navigation entries.
                if re.search(r"\.{2,}\s*\d+\s*$", line):
                    continue

                # Blank lines do not terminate a TOC. A genuine body section
                # starts when a conventional numbered heading without dotted
                # leaders is followed by substantial prose.
                body_match = re.match(
                    r"""(?x)^\s*
                    (?:\d+(?:\.\d+){0,5}|[IVXLCDM]+)
                    \s*(?:[.:\-–—])?\s+
                    \S.+$
                    """,
                    line,
                )

                next_idx = line_idx + 1
                while next_idx < len(lines) and not lines[next_idx].strip():
                    next_idx += 1

                next_text = (
                    lines[next_idx].strip()
                    if next_idx < len(lines)
                    else ""
                )
                next_words = re.findall(
                    r"\b[\wÀ-ÖØ-öø-ÿ'-]+\b",
                    next_text,
                    flags=re.UNICODE,
                )

                if body_match and len(next_words) >= 8:
                    in_contents = False
                else:
                    continue

            markdown = bool(re.match(r"^\s*#{1,6}\s+\S", line))

            line = re.sub(r"^\s*#{1,6}\s*", "", line)
            line = re.sub(r"^\s*[*_]{1,3}\s*", "", line)
            line = re.sub(r"\s*[*_]{1,3}\s*$", "", line).strip()

            if not line:
                continue

            if front_matter.match(line):
                continue
            if caption_like.match(line):
                continue

            # Table-of-contents entries commonly use dotted leaders followed
            # by a page number. They are not the actual section occurrence.
            if re.search(r"\.{4,}\s*\d+\s*$", line):
                continue

            # Common bibliography/reference-entry structures.
            if re.match(r"^\s*\[\d+\]\s+", line):
                continue

            # Generic bibliography-style row:
            # author-like text + four-digit publication year + citation punctuation.
            if (
                re.search(r"\b(?:18|19|20)\d{2}\b", line)
                and line.count(",") >= 2
                and re.search(r"(?:&|\bet\s+al\.?\b)", line, re.I)
            ):
                continue

            # Reject strongly mathematical / algorithmic / pseudocode lines.
            # These frequently begin with numbers or letters and can otherwise
            # look like numbered headings after OCR.
            if re.search(
                r"""(?ix)
                (?:\\left|\\right|\\frac|\\sum|\\tilde|\\text|\\mathbb|
                   \\epsilon|\\lambda|\\theta|\\psi|\\chi|
                   \\\\|<-|←|:=|=>|=)
                """,
                line,
            ):
                continue

            if re.match(
                r"""(?ix)^\s*
                (?:\d+\s*:\s*)?
                (?:
                    for\s+each\b|for\b|if\b|else\b|while\b|loop\b|
                    return\b|repeat\b|until\b
                )
                \s*$
                """,
                line,
            ):
                continue

            if re.match(r"^\s*\d+\s*\)\s+\S", line):
                continue

            # URLs, emails and explicit LaTeX display structures are not headings.
            if re.search(r"https?://|www\.|@|\\begin\{|\\end\{|^\s*\$", line, re.I):
                continue

            # Generic TeX matrix/alignment fragments.
            # Braces and ordinary LaTeX commands are allowed because valid
            # scientific section titles may contain mathematical notation.
            if (
                line.count("&") >= 2
                or line.count(r"\\") >= 2
                or re.search(
                    r"""(?ix)
                    \\begin\{
                    (?:matrix|pmatrix|bmatrix|vmatrix|Vmatrix|array|aligned|cases)
                    \}
                    """,
                    line,
                )
            ):
                continue

            num = None
            title = line
            appendix_prefix = None

            am = appendix_heading.match(line)
            if am:
                appendix_prefix = "appendix " + am.group("label")
                num = am.group("label")
                title = am.group("title").strip()
            else:
                m = numbered.match(line)
                if m:
                    num = m.group("num")
                    title = m.group("title").strip()

            title = re.sub(r"\s*[:;.\-–—]+\s*$", "", title).strip()
            if not title:
                continue

            words = re.findall(
                r"\b[\wÀ-ÖØ-öø-ÿ'-]+\b",
                title,
                flags=re.UNICODE,
            )

            if not words:
                continue

            # Scientific headings are generally concise noun phrases.
            if len(words) > 16 or len(title) > 160:
                continue

            # Reject lines dominated by mathematical syntax even when OCR has
            # made them superficially resemble numbered headings.
            math_tokens = len(
                re.findall(
                    r"""(?x)
                    \\[A-Za-z]+
                    |[=<>]
                    |\^
                    |_
                    |\{
                    |\}
                    |\\
                    """,
                    title,
                )
            )
            if not num and math_tokens >= 2:
                continue

            if num and re.search(
                r"(?:=|:=|<-|←|\\frac|\\sum|\\leftarrow)",
                title,
            ):
                continue

            # Strong prose indicators.
            if title.endswith(("?", "!")):
                continue

            if len(words) >= 8 and re.search(
                r"(?i)\b(?:is|are|was|were|has|have|had|we|our|this|these|"
                r"shows?|showed|using|used|can|could|would|should)\b",
                title,
            ):
                continue

            # Avoid obvious reference-like numbered lines.
            if num and re.search(
                r"(?i)\b(?:vol\.?|volume|pp\.?|pages?|doi|et\s+al\.?|"
                r"journal|proceedings|conference|press|publisher)\b",
                title,
            ):
                continue

            is_known = bool(known_heading.fullmatch(title))

            # Generic unnumbered scholarly headings:
            #
            # Many journal/conference styles use a short standalone heading
            # with no numeric marker and no Markdown markup. We accept such
            # headings structurally rather than through topic-specific words:
            #   - short title-like line;
            #   - located at a paragraph boundary;
            #   - followed by a substantial prose paragraph;
            #   - not sentence-like or mathematical.
            structural_unnumbered = False

            if not (is_known or num or markdown):
                prev_blank = (
                    line_idx == 0
                    or not lines[line_idx - 1].strip()
                )

                next_idx = line_idx + 1
                blank_gap = 0
                while next_idx < len(lines) and not lines[next_idx].strip():
                    blank_gap += 1
                    next_idx += 1
                    if blank_gap > 1:
                        break

                next_text = (
                    lines[next_idx].strip()
                    if next_idx < len(lines)
                    else ""
                )

                next_words = re.findall(
                    r"\b[\wÀ-ÖØ-öø-ÿ'-]+\b",
                    next_text,
                    flags=re.UNICODE,
                )

                # Title-like capitalization is only supporting evidence;
                # acronyms and mixed-case scientific terms remain allowed.
                alpha_tokens = [
                    w for w in words
                    if any(ch.isalpha() for ch in w)
                ]
                title_like = bool(alpha_tokens) and (
                    sum(
                        1
                        for w in alpha_tokens
                        if w[:1].isupper() or w.isupper()
                    )
                    >= max(1, len(alpha_tokens) // 2)
                )

                sentence_like = bool(
                    re.search(r"[.!?]\s*$", original_line)
                    or re.match(
                        r"""(?ix)^\s*
                        (?:we|this|these|those|it|there|our|the\s+(?:result|model|method))
                        \b""",
                        title,
                    )
                )

                math_like = bool(
                    re.search(
                        r"""(?x)
                        \\[A-Za-z]+
                        |[=<>]
                        |\$+
                        |\{[^}]*\}
                        |\[[^\]]*\]
                        """,
                        title,
                    )
                )

                structural_unnumbered = (
                    seen_body_structure
                    and prev_blank
                    and 1 <= len(words) <= 12
                    and len(title) <= 120
                    and len(next_words) >= 8
                    and title_like
                    and not sentence_like
                    and not math_like
                )

                if not structural_unnumbered:
                    continue

            # Generic custom headings need enough alphabetic content.
            alpha_words = [
                w for w in words
                if any(ch.isalpha() for ch in w)
            ]
            if not alpha_words:
                continue

            # A single-letter/number fragment is not a useful section title.
            if len("".join(alpha_words)) < 3:
                continue

            if appendix_prefix:
                title = f"{appendix_prefix}: {title}"

            value = normalize_string(title)
            if value:
                emitted.append((page_no, value))
                if is_known or num or markdown:
                    seen_body_structure = True

    return emitted



def _extract_caption(record):
    """
    Generic CAPTION_V1 for scholarly documents.

    Principles:
    - use explicit scholarly caption markers only;
    - support common figure/table-like object captions;
    - preserve the actual OCR page number;
    - do not infer captions from surrounding prose or object contents;
    - do not use document-, corpus-, filename- or Dev5-specific rules.
    """
    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    caption_start = re.compile(
        r"""(?ix)^\s*
        (?:
            # Figure/table-like captions normally use an explicit
            # separator after the object identifier.
            (?:
                fig(?:ure)?\.?
                |table\.?
                |tab\.?
                |scheme\.?
                |plate\.?
            )
            \s*
            (?:
                [A-Z]?\s*\d+(?:[.\-]\d+)*
                |[IVXLCDM]+
            )
            \s*
            [.:;\-–—)]
            \s*
            \S

            |

            # Algorithm/listing captions are also commonly typeset
            # as "Algorithm 1 Title" without punctuation after the
            # identifier.
            (?:
                algorithm\.?
                |listing\.?
            )
            \s*
            (?:
                [A-Z]?\s*\d+(?:[.\-]\d+)*
                |[IVXLCDM]+
            )
            (?:
                \s*[.:;\-–—)]\s*
                |\s+
            )
            \S
        )
        """
    )

    extracted = []

    for page_no, page_text in pages:
        for raw_line in page_text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue

            # Explicit object-caption marker must begin the line.
            # This avoids ordinary prose references such as
            # "As shown in Figure 2 ..." or "... see Table 3".
            if not caption_start.match(line):
                continue

            value = normalize_string(line)
            if value:
                extracted.append((page_no, value))

    return extracted


def _extract_equation(record):
    """
    Generic scientific-document equation extraction.

    V3:
    - extract explicit display mathematics only;
    - treat one display block as one logical equation by default;
    - split only clearly independent equation statements inside a display;
    - preserve matrix/cases/array structures as single equations;
    - merge syntactic continuation lines and adjacent continuation displays;
    - remove equation-number/layout metadata;
    - convert LaTeX to rendered mathematical text generically;
    - mirror eSciBench equation GT handling of "_" and "^";
    - use the benchmark's official normalize_string.
    """
    from pylatexenc.latex2text import LatexNodes2Text

    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    latex_to_text = LatexNodes2Text()

    env_name = (
        r"(?:equation|align|gather|multline|flalign|alignat|eqnarray)"
        r"\*?"
    )

    begin_env_re = re.compile(
        rf"\\begin\{{(?P<env>{env_name})\}}",
        re.IGNORECASE,
    )

    # Newlines inside these structures are normally mathematical layout,
    # not boundaries between independently extracted equations.
    structural_env_re = re.compile(
        r"\\begin\{(?:"
        r"matrix|pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|"
        r"smallmatrix|array|cases|aligned|alignedat|split"
        r")\*?\}",
        re.IGNORECASE,
    )

    continuation_start_re = re.compile(
        r"""(?x)^\s*
        (?:
            =
            |\+
            |-
            |[*/]
            |&
            |\\left\b
            |\\right\b
            |\\middle\b
            |\\(?:cdot|times|div|pm|mp|leq|geq|neq|approx|sim|propto)\b
            |\\(?:to|rightarrow|leftarrow|Rightarrow|Leftarrow|leftrightarrow)\b
            |\\(?:overset|underset)\b
        )
        """
    )

    relation_re = re.compile(
        r"""(?x)
        (?:
            =
            |\\(?:leq|geq|neq|approx|sim|propto)\b
            |\\(?:to|rightarrow|leftarrow|Rightarrow|Leftarrow|leftrightarrow)\b
            |\\in\b
            |\\subset(?:eq)?\b
            |\\supset(?:eq)?\b
        )
        """
    )

    def delimiter_balance(text):
        t = text or ""

        # Ignore TeX sizing commands but retain the visible delimiters.
        t = re.sub(r"\\(?:left|right|middle)\b", "", t)

        # Escaped braces are literal mathematical braces, not grouping syntax.
        t = t.replace(r"\{", "").replace(r"\}", "")

        return (
            t.count("(") - t.count(")"),
            t.count("[") - t.count("]"),
            t.count("{") - t.count("}"),
        )

    def previous_incomplete(text):
        value = (text or "").strip()
        if not value:
            return True

        if any(x > 0 for x in delimiter_balance(value)):
            return True

        core = re.sub(r"\\\\\s*$", "", value).rstrip()

        if re.search(
            r"""(?x)
            (?:
                =
                |\+
                |-
                |[*/]
                |&
                |\\(?:cdot|times|div|pm|mp|leq|geq|neq|approx|sim|propto)
                |\\(?:to|rightarrow|leftarrow|Rightarrow|Leftarrow|leftrightarrow)
            )\s*$
            """,
            core,
        ):
            return True

        # TeX continuation constructs such as "\left." commonly indicate
        # that the logical expression is continued in the next physical line
        # or display block.
        if re.search(r"\\(?:left|right)\.\s*$", core):
            return True

        return False

    def independent_statement(line):
        """
        Conservative test for a complete mathematical statement.

        We split physical lines only when both neighboring lines independently
        contain a mathematical relation/assignment. This avoids treating
        ordinary layout wrapping as multiple equations.
        """
        value = (line or "").strip()
        if not value:
            return False

        if continuation_start_re.match(value):
            return False

        if previous_incomplete(value):
            return False

        return bool(relation_re.search(value))

    def split_display_body(body):
        """
        Segment one display block into logical equations.

        Default: one display == one equation.
        Exception: split only clearly independent complete mathematical
        statements on separate physical lines.
        """
        body = (body or "").strip()
        if not body:
            return []

        # Matrix/cases/array-like structures use physical lines for layout.
        if structural_env_re.search(body):
            return [body]

        physical = [line.strip() for line in body.splitlines() if line.strip()]
        if len(physical) <= 1:
            return physical

        logical = []
        current = physical[0]

        for nxt in physical[1:]:
            if continuation_start_re.match(nxt) or previous_incomplete(current):
                current = re.sub(r"\\\\\s*$", " ", current).rstrip()
                current = f"{current} {nxt}".strip()
                continue

            # Split only when there is positive evidence that BOTH sides are
            # independently complete equation statements.
            if independent_statement(current) and independent_statement(nxt):
                logical.append(current)
                current = nxt
            else:
                current = re.sub(r"\\\\\s*$", " ", current).rstrip()
                current = f"{current} {nxt}".strip()

        if current:
            logical.append(current)

        return logical

    def clean_equation(text):
        value = (text or "").strip()
        if not value:
            return ""

        # Equation numbering / cross-reference metadata is not mathematical
        # equation content.
        value = re.sub(r"\\(?:notag|nonumber)\b", " ", value)
        value = re.sub(r"\\tag\s*\{[^{}]*\}", " ", value)
        value = re.sub(r"\\label\s*\{[^{}]*\}", " ", value)

        # Preserve the conventional probability operator name. pylatexenc
        # otherwise drops the unknown command itself while keeping arguments.
        value = re.sub(r"\\Pr\b", "Pr", value)

        # Remove TeX row separators that only encode visual layout.
        value = re.sub(r"\\\\", " ", value)

        try:
            value = latex_to_text.latex_to_text(value)
        except Exception:
            # Generic safe fallback: retain OCR content rather than inventing
            # or dropping an equation when conversion encounters malformed TeX.
            pass

        # Official eSciBench equation GT performs these replacements before
        # normalize_string. Apply the same representation rule to predictions.
        value = value.replace("_", " ").replace("^", " ")

        value = re.sub(r"[ \t]*\n[ \t]*", " ", value)
        value = re.sub(r"[ \t]+", " ", value).strip()

        return normalize_string(value) if value else ""

    def collect_display_blocks(page_text):
        """
        Return display blocks in document order, including source spans so
        adjacent displays can be merged only when no prose intervenes.
        """
        blocks = []
        i = 0
        n = len(page_text)

        while i < n:
            bracket_pos = page_text.find(r"\[", i)
            dollar_pos = page_text.find("$$", i)
            env_match = begin_env_re.search(page_text, i)

            candidates = []
            if bracket_pos != -1:
                candidates.append((bracket_pos, "bracket", None))
            if dollar_pos != -1:
                candidates.append((dollar_pos, "dollar", None))
            if env_match:
                candidates.append((env_match.start(), "env", env_match))

            if not candidates:
                break

            pos, kind, match = min(candidates, key=lambda x: x[0])

            if kind == "bracket":
                close = page_text.find(r"\]", pos + 2)
                if close == -1:
                    i = pos + 2
                    continue

                blocks.append({
                    "start": pos,
                    "end": close + 2,
                    "body": page_text[pos + 2:close],
                })
                i = close + 2
                continue

            if kind == "dollar":
                close = page_text.find("$$", pos + 2)
                if close == -1:
                    i = pos + 2
                    continue

                blocks.append({
                    "start": pos,
                    "end": close + 2,
                    "body": page_text[pos + 2:close],
                })
                i = close + 2
                continue

            env = match.group("env")
            close_re = re.compile(
                rf"\\end\{{{re.escape(env)}\}}",
                re.IGNORECASE,
            )
            close_match = close_re.search(page_text, match.end())

            if not close_match:
                i = match.end()
                continue

            blocks.append({
                "start": pos,
                "end": close_match.end(),
                "body": page_text[match.end():close_match.start()],
            })
            i = close_match.end()

        return blocks

    extracted = []

    for page_no, page_text in pages:
        blocks = collect_display_blocks(page_text)
        pending = None
        pending_end = None

        for block in blocks:
            pieces = split_display_body(block["body"])
            if not pieces:
                continue

            # If the previous display is syntactically incomplete and the
            # displays are adjacent (only whitespace between them), merge them.
            if pending is not None:
                gap = page_text[pending_end:block["start"]]
                first = pieces[0]

                # Cross-display merging must be substantially more
                # conservative than within-display line continuation.
                # A new display may legitimately begin with unary + or -,
                # so that alone is not evidence that it belongs to the
                # preceding display. Merge only when the preceding display
                # is syntactically incomplete, or when the following display
                # begins with a strong TeX delimiter continuation.
                strong_cross_display_continuation = re.match(
                    r"""(?x)^\s*
                    (?:
                        \\left\s*\.
                        |\\right\s*\.
                        |\\middle\b
                    )
                    """,
                    first,
                )

                if (
                    not gap.strip()
                    and (
                        previous_incomplete(pending)
                        or strong_cross_display_continuation
                    )
                ):
                    pending = f"{pending} {first}".strip()
                    pieces = pieces[1:]
                else:
                    value = clean_equation(pending)
                    if value:
                        extracted.append((page_no, value))
                    pending = None
                    pending_end = None

            for piece in pieces:
                if pending is None:
                    pending = piece
                    pending_end = block["end"]
                    continue

                if continuation_start_re.match(piece) or previous_incomplete(pending):
                    pending = f"{pending} {piece}".strip()
                    pending_end = block["end"]
                else:
                    value = clean_equation(pending)
                    if value:
                        extracted.append((page_no, value))
                    pending = piece
                    pending_end = block["end"]

            if pending is not None:
                pending_end = block["end"]

        if pending is not None:
            value = clean_equation(pending)
            if value:
                extracted.append((page_no, value))

    return extracted


def _extract_table(record):
    """
    Generic scientific-document table extraction.

    V1:
    - detect explicit HTML and Markdown tables;
    - treat one table as one extracted logical unit;
    - preserve reading order of rows/cells;
    - merge multiline content inside each cell;
    - discard empty cells and Markdown separator rows;
    - merge all non-empty cells of one table into one normalized string;
    - use the benchmark's official normalize_string.
    """
    import html as html_lib

    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    extracted = []

    def clean_cell(value):
        value = value or ""

        # Convert common HTML line/block boundaries to spaces before
        # stripping tags so multiline cell content remains readable.
        value = re.sub(
            r"(?is)<\s*(?:br|p|div|li)\b[^>]*>",
            " ",
            value,
        )
        value = re.sub(
            r"(?is)</\s*(?:p|div|li)\s*>",
            " ",
            value,
        )

        value = re.sub(r"(?is)<[^>]+>", " ", value)
        value = html_lib.unescape(value)

        value = re.sub(r"\s+", " ", value).strip()
        return value

    def finalize_cells(page_no, cells):
        cleaned = []

        for cell in cells:
            value = clean_cell(cell)
            if value:
                cleaned.append(value)

        if not cleaned:
            return

        merged = " ".join(cleaned)
        merged = normalize_string(merged)

        if merged:
            extracted.append((page_no, merged))

    html_table_re = re.compile(
        r"(?is)<table\b[^>]*>.*?</table\s*>"
    )
    html_cell_re = re.compile(
        r"(?is)<(?:td|th)\b[^>]*>(.*?)</(?:td|th)\s*>"
    )

    for page_no, page_text in pages:
        occupied = []

        # -------------------------------------------------------------
        # HTML TABLES
        # -------------------------------------------------------------
        for match in html_table_re.finditer(page_text):
            table_html = match.group(0)
            cells = html_cell_re.findall(table_html)

            # Generic fallback for imperfect OCR HTML: if explicit cells
            # cannot be parsed, retain the textual contents of the table
            # rather than silently dropping it.
            if cells:
                finalize_cells(page_no, cells)
            else:
                fallback = clean_cell(table_html)
                if fallback:
                    fallback = normalize_string(fallback)
                    if fallback:
                        extracted.append((page_no, fallback))

            occupied.append((match.start(), match.end()))

        # -------------------------------------------------------------
        # MARKDOWN TABLES
        # -------------------------------------------------------------
        lines = page_text.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i]

            # A Markdown table requires a pipe-containing header row
            # immediately followed by the conventional separator row.
            if "|" not in line or i + 1 >= len(lines):
                i += 1
                continue

            separator = lines[i + 1]

            sep_cells = [
                x.strip()
                for x in separator.strip().strip("|").split("|")
            ]

            is_separator = (
                "|" in separator
                and len(sep_cells) >= 2
                and all(
                    re.fullmatch(r":?-{3,}:?", x or "") is not None
                    for x in sep_cells
                )
            )

            if not is_separator:
                i += 1
                continue

            table_lines = [line]
            j = i + 2

            while j < len(lines):
                candidate = lines[j]

                if "|" not in candidate or not candidate.strip():
                    break

                table_lines.append(candidate)
                j += 1

            cells = []

            for row in table_lines:
                row_cells = row.strip().strip("|").split("|")

                for cell in row_cells:
                    value = cell.strip()
                    if value:
                        cells.append(value)

            finalize_cells(page_no, cells)
            i = j

        # -------------------------------------------------------------
        # PLAIN-TEXT TABLE RESCUE
        # -------------------------------------------------------------
        # OCR may preserve a visually structured table as plain text rather
        # than HTML/Markdown. Rescue only a conservative structural form:
        #
        #   Table N: caption
        #
        #   field label: value
        #
        #   field label: value
        #
        # Require at least two consecutive field:value paragraphs immediately
        # after an explicit table caption. This avoids treating arbitrary
        # colon-containing prose as a table.
        plain_caption_re = re.compile(
            r"(?im)^\s*Table\s+"
            r"(?:[A-Z]?\d+[A-Za-z]?|[IVXLC]+)"
            r"\s*[:.]\s*.+?\s*$"
        )

        plain_field_re = re.compile(
            r"^\s*(?P<label>[^:\n]{1,80})\s*:\s*(?P<value>\S.+)$",
            re.DOTALL,
        )

        for caption_match in plain_caption_re.finditer(page_text):
            after = page_text[caption_match.end():]

            paragraphs = [
                part.strip()
                for part in re.split(r"\n\s*\n", after)
                if part.strip()
            ]

            fields = []

            for paragraph in paragraphs:
                field_match = plain_field_re.match(paragraph)
                if not field_match:
                    break

                field_label = re.sub(
                    r"\s+",
                    " ",
                    field_match.group("label"),
                ).strip()

                field_value = field_match.group("value").strip()

                # Figure/image captions can also contain colons. They are not
                # table fields and must terminate this rescue candidate.
                if re.match(
                    r"(?i)^(?:!\[|fig(?:ure)?\.?\b)",
                    field_label,
                ):
                    fields = []
                    break

                fields.append(f"{field_label}: {field_value}")

            if len(fields) < 2:
                continue

            merged = normalize_string(
                re.sub(r"\s+", " ", " ".join(fields)).strip()
            )

            if merged:
                extracted.append((page_no, merged))

    return extracted



def _extract_reference(record):
    """
    Generic scientific bibliography/reference extraction.

    Structural evidence:
    1. Explicit References/Bibliography/Literature Cited heading.
    2. A long monotonically increasing [n] citation run near document end,
       allowing numbered bibliographies whose heading is absent from OCR.

    Entry boundaries:
    - numbered bibliographies: each [n], n., or n) begins a new entry;
      physical continuation lines are merged;
    - unnumbered bibliographies after an explicit heading: when the section
      contains a sustained sequence of citation-like physical lines, each
      such line is treated as one bibliography entry;
    - otherwise use conservative paragraph-level fallback.

    No bibliography content is synthesized.
    """
    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    heading_re = re.compile(
        r"(?im)^\s*(?:#{1,6}\s*)?"
        r"(?:references|bibliography|literature\s+cited)\s*$"
    )

    numbered_start_re = re.compile(
        r"""(?x)^\s*
        (?:
            \[\s*(\d+)[A-Za-z]?\s*\]
            |
            (\d+)[A-Za-z]?\s*[.)]
        )
        \s*
        """
    )

    bracket_start_re = re.compile(r"^\s*\[\s*(\d+)\s*\]\s+\S")

    def norm(value):
        value = re.sub(r"\s+", " ", value).strip()
        return normalize_string(value) if value else ""

    def split_numbered(lines):
        out = []
        current = None

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if numbered_start_re.match(line):
                if current:
                    value = numbered_start_re.sub("", current, count=1)
                    value = norm(value)
                    if value:
                        out.append(value)
                current = line
            elif current is not None:
                current = f"{current} {line}"

        if current:
            value = numbered_start_re.sub("", current, count=1)
            value = norm(value)
            if value:
                out.append(value)

        return out

    def citation_like_unnumbered(line):
        line = line.strip()
        if not line:
            return False

        # Generic bibliographic evidence: author-like beginning plus a
        # publication year somewhere in the entry.
        if not re.search(r"\b(?:18|19|20)\d{2}[a-z]?\b", line, re.I):
            return False

        if len(line.split()) < 5:
            return False

        return bool(
            re.match(
                r"^(?:[A-ZÀ-ÖØ-Þ][^\s,]{0,40}"
                r"|[—–-])",
                line,
            )
        )

    extracted = []

    # Flatten page text while retaining page ownership of each physical line.
    all_lines = []
    for page_no, page_text in pages:
        for line in page_text.splitlines():
            all_lines.append((page_no, line))

    full_text = "\n".join(line for _, line in all_lines)
    heading_match = heading_re.search(full_text)

    # ------------------------------------------------------------------
    # A) Explicit bibliography heading.
    # ------------------------------------------------------------------
    if heading_match:
        before = full_text[:heading_match.end()]
        start_line = before.count("\n")

        section = all_lines[start_line:]
        section_lines = [line for _, line in section]

        numbered_count = sum(
            1 for line in section_lines if numbered_start_re.match(line)
        )

        if numbered_count >= 2:
            # Preserve page number from each numbered entry start.
            current = None
            current_page = None

            for page_no, raw_line in section:
                line = raw_line.strip()
                if not line:
                    continue

                if numbered_start_re.match(line):
                    if current:
                        value = numbered_start_re.sub("", current, count=1)
                        value = norm(value)
                        if value:
                            extracted.append((current_page, value))
                    current = line
                    current_page = page_no
                elif current is not None:
                    current = f"{current} {line}"

            if current:
                value = numbered_start_re.sub("", current, count=1)
                value = norm(value)
                if value:
                    extracted.append((current_page, value))

            return extracted

        # Unnumbered bibliography represented as one citation per physical
        # line. Require a sustained citation-like sequence so ordinary prose
        # after a heading is not split indiscriminately.
        candidates = [
            (page_no, line.strip())
            for page_no, line in section
            if line.strip()
        ]
        citation_flags = [
            citation_like_unnumbered(line) for _, line in candidates
        ]

        if (
            len(candidates) >= 3
            and sum(citation_flags) >= 3
            and sum(citation_flags) / len(candidates) >= 0.60
        ):
            for page_no, line in candidates:
                if citation_like_unnumbered(line):
                    value = norm(line)
                    if value:
                        extracted.append((page_no, value))
            return extracted

        # Conservative paragraph fallback.
        joined = "\n".join(line for _, line in section)
        paragraphs = [
            re.sub(r"\s+", " ", part).strip()
            for part in re.split(r"\n\s*\n", joined)
            if part.strip()
        ]
        fallback_page = section[0][0] if section else 0

        for paragraph in paragraphs:
            if len(paragraph.split()) < 3:
                continue
            value = norm(paragraph)
            if value:
                extracted.append((fallback_page, value))

        return extracted

    # ------------------------------------------------------------------
    # B) No explicit heading: detect a strong numbered bibliography run.
    #    Require a monotonically increasing [n] run, at least 3 entries,
    #    beginning in the final 20% of the OCR text.
    # ------------------------------------------------------------------
    starts = []
    for i, (page_no, line) in enumerate(all_lines):
        m = bracket_start_re.match(line)
        if m:
            starts.append((i, page_no, int(m.group(1))))

    runs = []
    current = []

    for item in starts:
        if not current:
            current = [item]
        elif item[2] == current[-1][2] + 1:
            current.append(item)
        else:
            if len(current) >= 3:
                runs.append(current)
            current = [item]

    if len(current) >= 3:
        runs.append(current)

    strong_runs = [
        run for run in runs
        if run[0][0] / max(1, len(all_lines)) >= 0.80
    ]

    if not strong_runs:
        return []

    # Prefer the longest structurally supported terminal run.
    run = max(strong_runs, key=len)
    run_start = run[0][0]
    bibliography_lines = all_lines[run_start:]

    current = None
    current_page = None

    for page_no, raw_line in bibliography_lines:
        line = raw_line.strip()
        if not line:
            continue

        if bracket_start_re.match(line):
            if current:
                value = numbered_start_re.sub("", current, count=1)
                value = norm(value)
                if value:
                    extracted.append((current_page, value))
            current = line
            current_page = page_no
        elif current is not None:
            current = f"{current} {line}"

    if current:
        value = numbered_start_re.sub("", current, count=1)
        value = norm(value)
        if value:
            extracted.append((current_page, value))

    return extracted


def _extract_list(record):
    """
    Generic scientific-document list extraction.

    V2 principle:
    - A list is emitted as ONE complete structured object, not one prediction
      per item.
    - Detect only explicit typographic bullet runs, because numbered markers
      are structurally ambiguous with scientific section/subsection numbering,
      theorem steps, equations, and procedural prose.
    - Require at least two explicit bullet items.
    - Merge physical continuation lines into their preceding item.
    - Preserve list/item order.
    - Canonicalize the visible bullet marker to "*" while preserving OCR text.
    - Do not synthesize semantic content absent from OCR.
    """
    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    bullet_re = re.compile(
        r"^\s*(?:[-*+•◦▪▫‣⁃])\s+(.+?)\s*$"
    )

    structural_boundary_re = re.compile(
        r"""(?ix)^\s*
        (?:
            #{1,6}\s+\S
            |
            (?:abstract|references|bibliography|literature\s+cited|keywords?)\s*$
            |
            (?:[IVXLCDM]+|\d+)(?:\.\d+)*[.)]?\s+[A-Z][A-Za-z]
        )
        """
    )

    extracted = []

    for page_no, page_text in pages:
        lines = page_text.splitlines()
        i = 0

        while i < len(lines):
            first = bullet_re.match(lines[i])
            if not first:
                i += 1
                continue

            items = []
            current = [first.group(1).strip()]
            j = i + 1

            while j < len(lines):
                raw = lines[j]
                stripped = raw.strip()

                nxt = bullet_re.match(raw)
                if nxt:
                    items.append(current)
                    current = [nxt.group(1).strip()]
                    j += 1
                    continue

                if not stripped:
                    # Blank line terminates the current explicit bullet run.
                    break

                if structural_boundary_re.match(raw):
                    break

                # Non-marker line directly inside a bullet run is interpreted
                # as physical-line continuation of the current item.
                current.append(stripped)
                j += 1

            items.append(current)

            # Require at least two explicit bullet items.
            if len(items) >= 2:
                normalized_items = []

                for parts in items:
                    item = re.sub(r"\s+", " ", " ".join(parts)).strip()
                    if not item:
                        continue

                    item = normalize_string(item)
                    if item:
                        normalized_items.append(item)

                if len(normalized_items) >= 2:
                    # Canonical list representation: one output containing
                    # all items in sequence.
                    value = " ".join(f"* {item}" for item in normalized_items)
                    value = normalize_string(value)

                    if value:
                        extracted.append((page_no, value))

                i = max(j, i + 1)
            else:
                i += 1

    return extracted


def _extract_header(record):
    """
    Generic scientific-document running-header extraction.

    V4:
    - Header must be explicitly present in OCR.
    - Inspect only the first few non-empty lines of each page.
    - Require recurrence across at least three distinct pages.
    - Reject body text, sections, captions, equations, HTML/table markup,
      page numbers, affiliations, emails, and trivial fragments.
    - Never reconstruct or synthesize header text absent from OCR.
    """
    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    page_number_re = re.compile(r"^\s*[\[(]?\d{1,4}[\])\.]?\s*$")
    email_re = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")

    html_re = re.compile(
        r"(?i)^\s*</?(?:table|tr|td|th|thead|tbody|tfoot|div|span|p)\b"
    )

    caption_re = re.compile(
        r"(?ix)^\s*(?:fig(?:ure)?\.?\s*\d+|table\s*\d+)\b"
    )

    section_re = re.compile(
        r"""(?ix)^\s*
        (?:
            \d+(?:\.\d+)*\.?\s+[A-Z]
            |
            [IVXLCDM]+\.?\s+[A-Z]
            |
            [A-Z]\.\s+[A-Z]
        )
        """
    )

    excluded_re = re.compile(
        r"""(?ix)^\s*
        (?:
            abstract\b
            |keywords?\b
            |contents\b
            |references\b
            |bibliography\b
            |literature\s+cited\b
            |acknowledg(?:e)?ments?\b
            |definition\b
            |theorem\b
            |lemma\b
            |proposition\b
            |corollary\b
        )
        """
    )

    affiliation_terms = (
        "university", "department", "faculty", "school",
        "institute", "laboratory", "laboratoire",
        "centre", "center", "hospital", "college"
    )

    candidates = []

    for page_no, page_text in pages:
        lines = [x.strip() for x in page_text.splitlines() if x.strip()]
        if not lines:
            continue

        # Running headers are expected at the very top of the page.
        for pos, line in enumerate(lines[:4]):
            if len(line) < 8 or len(line) > 150:
                continue

            if page_number_re.fullmatch(line):
                continue
            if email_re.search(line):
                continue
            if html_re.match(line):
                continue
            if caption_re.match(line):
                continue
            if section_re.match(line):
                continue
            if excluded_re.match(line):
                continue

            lower = line.lower()
            if any(term in lower for term in affiliation_terms):
                continue

            # Reject equation/math-heavy lines.
            if re.search(r"(?:\\\[|\\\]|\\begin|\\end|[=∑∫√∂∇])", line):
                continue

            words = line.split()

            # Trivial fragments and paragraph-like lines are not headers.
            if len(words) < 2 or len(words) > 14:
                continue

            if len(line) > 110 and line.endswith((".", ";", ":")):
                continue

            text = normalize_string(line)
            if text:
                candidates.append((page_no, pos, text))

    pages_by_text = {}
    for page_no, _, text in candidates:
        pages_by_text.setdefault(text, set()).add(page_no)

    extracted = []
    seen = set()

    for page_no, _, text in candidates:
        # Strong generic running-header evidence.
        if len(pages_by_text[text]) < 3:
            continue

        key = (page_no, text)
        if key in seen:
            continue

        seen.add(key)
        extracted.append((page_no, text))

    return extracted


def _extract_footer(record):
    """
    Generic scientific-document footer / footnote extraction.

    Principles:
    - Footers may appear as explicit LaTeX-style footnotes embedded in OCR
      reading order rather than literally at the end of a page.
    - Extract explicit footnote content without synthesizing missing text.
    - Also accept standalone scientific author/contact notes.
    - Repeated bottom-of-page text is accepted only with recurrence evidence.
    - Reject ordinary body prose, references, captions, equations and page
      numbers.
    """
    pages = _get_olmocr_pages(record)
    if not pages:
        return []

    extracted = []

    # ---------------------------------------------------------------
    # 1. Explicit LaTeX-style \footnote{...}
    #    Balanced-brace parser, so nested braces do not break extraction.
    # ---------------------------------------------------------------
    for page_no, page_text in pages:
        marker = r"\footnote{"
        pos = 0

        while True:
            start = page_text.find(marker, pos)
            if start == -1:
                break

            content_start = start + len(marker)
            depth = 1
            i = content_start

            while i < len(page_text) and depth > 0:
                ch = page_text[i]

                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1

                i += 1

            if depth == 0:
                value = page_text[content_start:i - 1]
                value = re.sub(r"\s+", " ", value).strip()

                if value:
                    value = normalize_string(value)
                    if value:
                        extracted.append((page_no, value))

                pos = i
            else:
                # Malformed/unclosed footnote: do not invent a boundary.
                break

    # ---------------------------------------------------------------
    # 2. Explicit standalone author/contact notes.
    # ---------------------------------------------------------------
    note_re = re.compile(
        r"""(?ix)^\s*
        (?:
            electronic\s+address
            |corresponding\s+author
            |author\s+correspondence
            |correspondence\s+to
        )
        \s*[:\-]\s*(.+?)\s*$
        """
    )

    for page_no, page_text in pages:
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            m = note_re.match(line)
            if not m:
                continue

            value = normalize_string(line)
            if value:
                extracted.append((page_no, value))

    # ---------------------------------------------------------------
    # 3. Repeated compact bottom-of-page material.
    # ---------------------------------------------------------------
    page_number_re = re.compile(r"^\s*[\[(]?\d{1,4}[\])\.]?\s*$")
    reference_re = re.compile(r"^\s*(?:\[\d+\]|\d+[.)])\s+")
    caption_re = re.compile(r"(?ix)^\s*(?:fig(?:ure)?\.?\s*\d+|table\s*\d+)\b")
    equation_re = re.compile(r"(?:\\\[|\\\]|\\begin|\\end|[=∑∫√∂∇])")

    bottom_candidates = []

    for page_no, page_text in pages:
        lines = [x.strip() for x in page_text.splitlines() if x.strip()]
        if not lines:
            continue

        for line in lines[-4:]:
            if len(line) < 8 or len(line) > 180:
                continue
            if page_number_re.fullmatch(line):
                continue
            if reference_re.match(line):
                continue
            if caption_re.match(line):
                continue
            if equation_re.search(line):
                continue

            words = line.split()
            if len(words) < 2 or len(words) > 18:
                continue

            value = normalize_string(line)
            if value:
                bottom_candidates.append((page_no, value))

    pages_by_text = {}
    for page_no, value in bottom_candidates:
        pages_by_text.setdefault(value, set()).add(page_no)

    for page_no, value in bottom_candidates:
        if len(pages_by_text[value]) >= 2:
            extracted.append((page_no, value))

    # Exact deduplication while preserving order.
    seen = set()
    result = []

    for page_no, value in extracted:
        key = (page_no, value)
        if key in seen:
            continue
        seen.add(key)
        result.append((page_no, value))

    return result

def extract_raw(base_dir, label, pdf):
    """
    eSciBench extractor interface for the clean OLMOCR wrapper.

    Final evaluation remains locked to the fixed Test96 dataset.
    """
    record = _get_olmocr_record(base_dir, pdf)

    if label == "title":
        extracted_titles = _extract_title(record)

        return True, [
            (pdf.pdf_name, 0, label, title)
            for title in extracted_titles
        ]

    if label == "footer":
        extracted_footers = _extract_footer(record)

        return True, [
            (pdf.pdf_name, page_no, label, footer)
            for page_no, footer in extracted_footers
        ]

    if label == "header":
        extracted_headers = _extract_header(record)

        return True, [
            (pdf.pdf_name, page_no, label, header)
            for page_no, header in extracted_headers
        ]

    if label == "author":
        extracted_authors = _extract_author(record)

        return True, [
            (pdf.pdf_name, 0, label, author)
            for author in extracted_authors
        ]

    if label == "affiliation":
        extracted_affiliations = _extract_affiliation(record)

        return True, [
            (pdf.pdf_name, 0, label, affiliation)
            for affiliation in extracted_affiliations
        ]

    if label == "email":
        extracted_emails = _extract_email(record)

        return True, [
            (pdf.pdf_name, 0, label, email)
            for email in extracted_emails
        ]

    if label == "keyword":
        extracted_keywords = _extract_keyword(record)

        return True, [
            (pdf.pdf_name, 0, label, keyword)
            for keyword in extracted_keywords
        ]

    if label == "pub_date":
        extracted_pub_dates = _extract_pub_date(record)

        return True, [
            (pdf.pdf_name, 0, label, pub_date)
            for pub_date in extracted_pub_dates
        ]

    if label == "abstract":
        extracted_abstracts = _extract_abstract(record)

        return True, [
            (pdf.pdf_name, 0, label, abstract)
            for abstract in extracted_abstracts
        ]

    if label == "section":
        extracted_sections = _extract_section(record)

        return True, [
            (pdf.pdf_name, page_no, label, section)
            for page_no, section in extracted_sections
        ]

    if label == "caption":
        extracted_captions = _extract_caption(record)

        return True, [
            (pdf.pdf_name, page_no, label, caption)
            for page_no, caption in extracted_captions
        ]

    if label == "equation":
        extracted_equations = _extract_equation(record)

        return True, [
            (pdf.pdf_name, page_no, label, equation)
            for page_no, equation in extracted_equations
        ]

    if label == "table":
        extracted_tables = _extract_table(record)

        return True, [
            (pdf.pdf_name, page_no, label, table)
            for page_no, table in extracted_tables
        ]

    if label == "reference":
        extracted_references = _extract_reference(record)

        return True, [
            (pdf.pdf_name, page_no, label, reference)
            for page_no, reference in extracted_references
        ]

    if label in {"list", "list"}:
        extracted_lists = _extract_list(record)

        return True, [
            (pdf.pdf_name, page_no, label, item)
            for page_no, item in extracted_lists
        ]

    return True, []
