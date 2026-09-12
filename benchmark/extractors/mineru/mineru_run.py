import html
import json
import re
from pathlib import Path
from html.parser import HTMLParser

from benchmark.normalisation import normalize_string
from pylatexenc.latex2text import LatexNodes2Text


# Labels implemented in this first MinerU wrapper version.
SUPPORTED_LABELS = {
    "title",
    "abstract",
    "header",
    "table",
    "caption",
    "reference",
    "author",
    "affiliation",
    "email",
    "keyword",
    "pub_date",
    "section",
    "footer",
    "list",
    "equation",
}


def _document_stem(pdf) -> str:
    """
    Return the PDF filename without its final .pdf suffix.
    """
    name = str(pdf.pdf_name)
    if name.lower().endswith(".pdf"):
        return name[:-4]
    return name


def _middle_json_path(pdf) -> Path:
    """
    Resolve the MinerU middle.json generated for this document.

    Expected Dev5 structure:
        Dev5_MinerU_orginal/
            Mineru_run.py
            raw_outputs/
                <document>/
                    auto/
                        <document>_middle.json
    """
    stem = _document_stem(pdf)
    return (
        Path(__file__).resolve().parent
        / "raw_outputs"
        / stem
        / "auto"
        / f"{stem}_middle.json"
    )


def _load_middle_json(pdf) -> dict:
    path = _middle_json_path(pdf)

    if not path.is_file():
        raise FileNotFoundError(
            f"MinerU middle.json not found for {pdf.pdf_name}: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _walk_nodes(obj):
    """
    Recursively yield every dictionary in a MinerU structure.

    This is structural only: no document-specific text, filename,
    coordinate, journal, or Dev5-specific condition is used.
    """
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_nodes(value)

    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_nodes(item)


def _collect_content(obj) -> str:
    """
    Reconstruct textual content from MinerU nodes by collecting
    their leaf 'content' fields in reading/storage order.

    If a parent and one of its children expose the exact same
    content value, the value is not duplicated consecutively.
    """
    parts = []

    def walk(x):
        if isinstance(x, dict):
            content = x.get("content")

            if isinstance(content, str) and content.strip():
                value = content.strip()

                if not parts or parts[-1] != value:
                    parts.append(value)

            for key, value in x.items():
                if key == "content":
                    continue
                walk(value)

        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)

    return " ".join(parts)


def _clean_text(text: str) -> str:
    """
    Apply only generic representation cleanup before the official
    eSciBench normalization.
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    # MinerU may preserve presentational HTML tags such as <sup>.
    # Remove the tags while preserving their textual content.
    text = re.sub(r"<[^>]+>", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    return normalize_string(text)


def _block_text(block: dict) -> str:
    """
    Extract and normalize text from one semantic MinerU block.
    """
    text = _collect_content(block)
    return _clean_text(text)


def _append_result(results, pdf, label, text):
    """
    Add one extraction in the tuple format required by the
    official eSciBench run_benchmark.py.
    """
    if not text:
        return

    results.append(
        (
            pdf.pdf_name,
            0,
            label,
            text,
        )
    )



_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])"
)

_AFFILIATION_HINT_RE = re.compile(
    r"\b(?:"
    r"university|universit[eé]|college|school|department|dept\.?|"
    r"faculty|institute|institution|laboratory|laboratoire|lab\.?|"
    r"centre|center|academy|hospital|clinic|"
    r"physics|mathematics|engineering|science|sciences|"
    r"research|cnrs|inria|cern|cea"
    r")\b",
    re.IGNORECASE,
)

_ABSTRACT_MARKER_RE = re.compile(r"^\s*abstract[\s.:—-]*$", re.IGNORECASE)

_KEYWORD_MARKER_RE = re.compile(
    r"^\s*(?:keywords?|key\s+words?)\s*[.:—-]*\s*",
    re.IGNORECASE,
)

_PUBDATE_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"publication\s+date|published(?:\s+online)?|"
    r"preprint\s+online\s+version|online\s+publication|date"
    r")\s*[:.-]\s*(.+?)\s*$",
    re.IGNORECASE,
)


def _page0_para_blocks(data: dict) -> list[dict]:
    pages = data.get("pdf_info", [])
    if not pages:
        return []
    return pages[0].get("para_blocks", [])


def _page0_extra_blocks(data: dict) -> list[dict]:
    pages = data.get("pdf_info", [])
    if not pages:
        return []

    page = pages[0]
    return (
        list(page.get("para_blocks", []))
        + list(page.get("discarded_blocks", []))
    )


def _frontmatter_blocks(data: dict) -> list[dict]:
    """
    Return page-0 blocks after the document title and before the
    abstract.  This boundary is structural and independent of any
    document identity or corpus-specific phrase.
    """
    blocks = _page0_para_blocks(data)
    out = []
    seen_document_title = False

    for block in blocks:
        block_type = block.get("type")
        raw = _collect_content(block).strip()
        clean = _clean_text(raw)

        if not seen_document_title:
            if block_type == "title":
                seen_document_title = True
            continue

        if block_type == "abstract":
            break

        if block_type == "title" and _ABSTRACT_MARKER_RE.match(clean or ""):
            break

        out.append(block)

    return out


def _raw_without_tags(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_email_text(text: str) -> str:
    text = _EMAIL_RE.sub(" ", text or "")
    text = re.sub(r"\bE-?mail\s*:\s*", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" ,;:-")


def _clean_author_candidate(text: str) -> str:
    text = _raw_without_tags(text)
    text = re.sub(r"^\s*(?:and|&)\s+", "", text, flags=re.IGNORECASE)

    # Generic scholarly membership suffixes are metadata, not names.
    text = re.sub(
        r",?\s*(?:(?:senior|student|fellow)\s+)?member\s+"
        r"(?:IEEE|ACM)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip(" ,;:*†‡")

    if not text:
        return ""

    if (
        "@" in text
        or _AFFILIATION_HINT_RE.search(text)
        or re.search(r"\d", text)
    ):
        return ""

    words = re.findall(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*\.?", text, re.UNICODE)

    if not (2 <= len(words) <= 8):
        return ""

    return _clean_text(text)


def _authors_from_block(raw: str) -> list[str]:
    authors = []

    if re.search(r"<sup\b", raw, flags=re.IGNORECASE):
        # Each text segment immediately before a superscript is a
        # generic author candidate.  This handles author lists with
        # affiliation markers without depending on marker values.
        pieces = re.split(
            r"<sup\b[^>]*>.*?</sup>",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )

        for piece in pieces[:-1]:
            candidate = _clean_author_candidate(piece)
            if candidate and candidate not in authors:
                authors.append(candidate)

        return authors

    plain = _raw_without_tags(raw)

    if (
        not plain
        or "@" in plain
        or _AFFILIATION_HINT_RE.search(plain)
        or _PUBDATE_MARKER_RE.match(plain)
        or _KEYWORD_MARKER_RE.match(plain)
    ):
        return []

    # Generic coordination used in author lines without superscripts.
    pieces = re.split(r"\s+(?:and|&)\s+", plain, flags=re.IGNORECASE)

    for piece in pieces:
        candidate = _clean_author_candidate(piece)
        if candidate and candidate not in authors:
            authors.append(candidate)

    return authors


def _extract_author(data: dict, pdf) -> list[tuple]:
    results = []
    seen = set()

    for block in _frontmatter_blocks(data):
        raw = _collect_content(block)

        for author in _authors_from_block(raw):
            if author not in seen:
                seen.add(author)
                _append_result(results, pdf, "author", author)

    return results


def _extract_affiliation(data: dict, pdf) -> list[tuple]:
    results = []

    sup_re = re.compile(
        r"<sup\b[^>]*>(.*?)</sup>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    institutional_re = re.compile(
        r"\b(?:"
        r"university|université|college|school|department|dept\.?|faculty|"
        r"institute|institution|laboratory|laboratoire|lab\.?|centre|center|"
        r"academy|hospital|clinic|physics|mathematics|engineering|"
        r"sciences?|research"
        r")\b",
        flags=re.IGNORECASE,
    )

    membership_re = re.compile(
        r",\s*((?:(?:student|graduate|senior|life|associate|fellow)\s+)*"
        r"member\b[^,;]*)\s*$",
        flags=re.IGNORECASE,
    )

    def norm_marker(value: str) -> str:
        value = _raw_without_tags(value)
        value = re.sub(r"[\s,;]+", "", value)
        return value.strip()

    def markers(raw: str) -> list[str]:
        out = []

        for value in sup_re.findall(raw):
            value = norm_marker(value)
            if value:
                out.append(value)

        return out

    def starts_with_sup(raw: str):
        return re.match(
            r"^\s*<sup\b[^>]*>(.*?)</sup>",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )

    blocks = _frontmatter_blocks(data)

    # Identify superscript markers that really introduce affiliation
    # blocks in the front matter.  This prevents symbols used only for
    # corresponding-author/equal-contribution notes from being treated
    # automatically as affiliation markers.
    affiliation_markers = set()

    for block in blocks:
        raw = _collect_content(block).strip()
        match = starts_with_sup(raw)

        if match:
            marker = norm_marker(match.group(1))
            if marker:
                affiliation_markers.add(marker)

    # Build generic author -> affiliation-marker multiplicities.
    author_marker_counts = {}

    for block in blocks:
        raw = _collect_content(block).strip()

        if not raw or starts_with_sup(raw):
            continue

        block_markers = [
            marker
            for marker in markers(raw)
            if marker in affiliation_markers
        ]

        if not block_markers:
            continue

        authors = _authors_from_block(raw)

        if not authors:
            continue

        # In conventional author lists, affiliation superscripts appear
        # in author order.  Only actual affiliation markers are retained.
        usable = block_markers[:len(authors)]

        for marker in usable:
            author_marker_counts[marker] = (
                author_marker_counts.get(marker, 0) + 1
            )

    entries = []

    def emit(text: str, marker=None, repeat=None):
        text = _strip_email_text(_raw_without_tags(text)).strip()
        text = re.sub(r"\s*[;,]\s*$", "", text).strip()
        text = _clean_text(text)

        if not text:
            return

        if repeat is None:
            if marker:
                repeat = max(
                    1,
                    author_marker_counts.get(marker, 1),
                )
            else:
                repeat = 1

        entries.extend([text] * max(1, repeat))

    current_text = ""
    current_marker = None
    active_marker = None
    pending_author_count = 0

    def flush_current():
        nonlocal current_text, current_marker

        if current_text:
            emit(current_text, marker=current_marker)

        current_text = ""
        current_marker = None

    for block in blocks:
        raw = _collect_content(block).strip()

        if not raw:
            continue

        plain = _raw_without_tags(raw).strip()

        has_sup = bool(sup_re.search(raw))

        # Metadata boundaries terminate affiliation continuation.
        # An email-containing block is NOT discarded when it also has
        # a superscript marker, because scientific front matter may be
        # collapsed as: Author<sup>marker</sup> Affiliation Email.
        if (
            _PUBDATE_MARKER_RE.match(plain)
            or _KEYWORD_MARKER_RE.match(plain)
            or (_EMAIL_RE.search(plain) and not has_sup)
        ):
            flush_current()
            active_marker = None
            pending_author_count = 0
            continue

        start_match = starts_with_sup(raw)

        # -------------------------------------------------------------
        # 1. Explicit marker-started affiliation.
        # -------------------------------------------------------------
        if start_match:
            flush_current()

            marker = norm_marker(start_match.group(1))

            if marker not in affiliation_markers:
                active_marker = None
                pending_author_count = 0
                continue

            text = re.sub(
                r"^\s*<sup\b[^>]*>.*?</sup>",
                "",
                raw,
                count=1,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()

            active_marker = marker
            pending_author_count = 0

            # A trailing comma is a strong generic signal that the
            # affiliation continues on the following physical line.
            if text.rstrip().endswith(","):
                current_text = text
                current_marker = marker
            else:
                # Semicolon or sentence ending closes this affiliation.
                # active_marker is deliberately retained because a
                # following unmarked institutional line may represent
                # another affiliation of the same author.
                emit(text, marker=marker)

            continue

        # -------------------------------------------------------------
        # 2. Mixed author + superscript + affiliation in one block.
        # -------------------------------------------------------------
        if has_sup:
            flush_current()

            membership_match = membership_re.search(plain)

            if membership_match:
                emit(membership_match.group(1))

            parts = re.split(
                r"<sup\b[^>]*>.*?</sup>",
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            )

            prefix = parts[0].strip() if parts else ""
            tail = parts[-1].strip() if parts else ""

            prefix_plain = _raw_without_tags(prefix).strip()
            tail_plain = _strip_email_text(
                _raw_without_tags(tail)
            ).strip()

            # MinerU can collapse scientific front matter into:
            # Author<sup>marker</sup> Affiliation Email
            # Parse the author-side and affiliation-side independently.
            prefix_authors = _authors_from_block(prefix_plain)

            if (
                tail_plain
                and institutional_re.search(tail_plain)
            ):
                author_count = len(prefix_authors)

                # A non-empty prefix immediately before a superscript
                # represents at least one author even if the generic
                # author parser cannot classify the whole name form.
                if author_count < 1 and prefix_plain:
                    author_count = 1

                emit(
                    tail_plain,
                    repeat=max(1, author_count),
                )

                pending_author_count = 0
                active_marker = None

            else:
                pending_author_count = len(prefix_authors)
                active_marker = None

            continue

        # -------------------------------------------------------------
        # 3. Plain author line or plain affiliation/address line.
        # -------------------------------------------------------------
        authors = _authors_from_block(raw)

        membership_match = membership_re.search(plain)

        if membership_match:
            emit(membership_match.group(1))

        if authors and not institutional_re.search(plain):
            flush_current()
            active_marker = None
            pending_author_count += len(authors)
            continue

        text = _strip_email_text(plain).strip()

        if not text:
            continue

        # Continuation of a marker affiliation whose previous physical
        # line ended with a comma.
        if current_text:
            current_text = f"{current_text} {text}".strip()

            if not text.rstrip().endswith(","):
                flush_current()

            continue

        if institutional_re.search(text):
            # An unmarked affiliation immediately following an explicit
            # marker affiliation remains attached to that same marker.
            if active_marker:
                emit(text, marker=active_marker)

            # Otherwise, an unmarked affiliation immediately below one
            # or more author lines is shared by those authors.
            elif pending_author_count:
                emit(text, repeat=pending_author_count)
                pending_author_count = 0

            else:
                emit(text)

            continue

        # A non-author, non-institutional block ends implicit context.
        active_marker = None
        pending_author_count = 0

    flush_current()

    # -------------------------------------------------------------
    # 4. Author-affiliation relations in page footnotes.
    # -------------------------------------------------------------
    page0 = (data.get("pdf_info") or [{}])[0]

    for block in page0.get("discarded_blocks", []):
        # Only genuine page footnotes are considered here.
        # arXiv side notes, headers and page numbers are ignored.
        if block.get("type") != "page_footnote":
            continue

        raw = _raw_without_tags(
            _collect_content(block)
        ).strip()

        if not raw:
            continue

        relation = re.search(
            r"^(.*?)\b(?:are|is)\s+with\b\s+(.+)$",
            raw,
            flags=re.IGNORECASE,
        )

        if not relation:
            continue

        author_part = relation.group(1).strip()
        affiliation_part = relation.group(2).strip()

        author_count = len(_authors_from_block(author_part))

        if author_count < 1:
            author_count = 1

        # Remove trailing contact information while preserving the
        # institutional affiliation itself.
        affiliation_part = re.sub(
            r"\s*\([^()]*\b(?:phone|tel\.?|telephone|email|e-mail)"
            r"\b[^()]*\)\s*\.?\s*$",
            "",
            affiliation_part,
            flags=re.IGNORECASE,
        )

        affiliation_part = _strip_email_text(
            affiliation_part
        ).strip()

        if (
            affiliation_part
            and institutional_re.search(affiliation_part)
        ):
            emit(
                affiliation_part,
                repeat=author_count,
            )

    # No global deduplication: repeated affiliations are intentional
    # when several authors share the same affiliation.
    for affiliation in entries:
        _append_result(
            results,
            pdf,
            "affiliation",
            affiliation,
        )

    return results

def _extract_email(data: dict, pdf) -> list[tuple]:
    results = []
    seen = set()

    # Scientific papers sometimes compress several e-mail local parts
    # that share one domain, for example:
    # {name1, name2}@domain.org
    shared_domain_re = re.compile(
        r"\{([^{}]+)\}\s*@\s*"
        r"([A-Za-z0-9.-]+\.[A-Za-z]{2,})",
        re.IGNORECASE,
    )

    def emit(email_addr: str):
        email_addr = _clean_text(email_addr)

        if email_addr and email_addr not in seen:
            seen.add(email_addr)
            _append_result(results, pdf, "email", email_addr)

    for block in _page0_extra_blocks(data):
        raw = html.unescape(_collect_content(block))

        # 1. Expand several local parts sharing the same domain.
        shared_spans = []

        for match in shared_domain_re.finditer(raw):
            shared_spans.append(match.span())

            local_group = match.group(1)
            domain = match.group(2).strip()

            for local in re.split(r"[,;]", local_group):
                local = local.strip()

                if re.fullmatch(
                    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+",
                    local,
                ):
                    emit(f"{local}@{domain}")

        # 2. Remove already-expanded shared-domain expressions so the
        # ordinary e-mail regex cannot produce partial duplicates.
        ordinary_raw = raw

        for left, right in reversed(shared_spans):
            ordinary_raw = (
                ordinary_raw[:left]
                + " "
                + ordinary_raw[right:]
            )

        # 3. Extract ordinary complete e-mail addresses.
        for email_addr in _EMAIL_RE.findall(ordinary_raw):
            emit(email_addr)

    return results

def _split_keywords(text: str) -> list[str]:
    text = _KEYWORD_MARKER_RE.sub("", text or "", count=1).strip()

    if not text:
        return []

    # Generic delimiters commonly used for keyword lists.
    parts = re.split(r"\s*(?:,|;|—|–|\u2014|\u2013)\s*", text)

    return [
        _clean_text(part)
        for part in parts
        if _clean_text(part)
    ]


def _extract_keyword(data: dict, pdf) -> list[tuple]:
    results = []
    blocks = _page0_para_blocks(data)
    seen = set()

    for i, block in enumerate(blocks):
        text = _raw_without_tags(_collect_content(block))

        match = _KEYWORD_MARKER_RE.match(text)
        if not match:
            continue

        payload = text[match.end():].strip()

        # MinerU may make "Keywords:" a separate title block and put
        # the actual list in the immediately following text block.
        if not payload and i + 1 < len(blocks):
            next_block = blocks[i + 1]
            if next_block.get("type") == "text":
                payload = _raw_without_tags(_collect_content(next_block))

        for keyword in _split_keywords(payload):
            if keyword and keyword not in seen:
                seen.add(keyword)
                _append_result(results, pdf, "keyword", keyword)

    return results


def _extract_pub_date(data: dict, pdf) -> list[tuple]:
    """
    Pub_date V3 — generic scientific-article front-matter rule.

    - inspect only the first two pages;
    - detect date expressions independently of explicit labels;
    - strongly prefer publication-specific context;
    - reject arXiv version/submission dates;
    - reject editorial-history dates such as received, accepted,
      revised and submitted;
    - allow a short standalone front-matter date as a weaker fallback;
    - emit only the highest-ranked candidate.
    """

    month = (
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    )

    date_patterns = [
        re.compile(
            rf"\b({month}\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+\d{{4}})\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(\d{{1,2}}(?:st|nd|rd|th)?\s+{month}[,]?\s+\d{{4}})\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b({month}\s+\d{{4}})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b"
        ),
        re.compile(
            r"\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{4})\b"
        ),
    ]

    publication_re = re.compile(
        r"\b(?:"
        r"publication\s+date|date\s+of\s+publication|"
        r"published(?:\s+online)?|online\s+publication|"
        r"publication|preprint\s+online\s+version"
        r")\b",
        re.IGNORECASE,
    )

    editorial_re = re.compile(
        r"\b(?:received|accepted|revised|submitted|resubmitted)\b",
        re.IGNORECASE,
    )

    arxiv_re = re.compile(r"\barxiv\b", re.IGNORECASE)

    candidates = []
    order = 0

    pages = data.get("pdf_info", []) or []

    for page_index, page in enumerate(pages[:2]):
        blocks = []

        for block in page.get("para_blocks", []) or []:
            if isinstance(block, dict):
                blocks.append(block)

        for key in ("discarded_blocks", "preproc_blocks"):
            for block in page.get(key, []) or []:
                if isinstance(block, dict):
                    blocks.append(block)

        for block in blocks:
            raw = _collect_content(block)
            if not raw:
                continue

            plain = _clean_text(_raw_without_tags(raw))
            if not plain:
                continue

            if arxiv_re.search(plain):
                continue

            if editorial_re.search(plain):
                continue

            found_dates = []
            for pattern in date_patterns:
                for match in pattern.finditer(plain):
                    value = _clean_text(match.group(1))
                    if value:
                        found_dates.append(value)

            if not found_dates:
                continue

            block_type = str(block.get("type", "") or "").lower()
            text_len = len(plain)
            has_publication_context = bool(publication_re.search(plain))

            for value in found_dates:
                score = 0

                # Strongest evidence: explicit publication semantics.
                if has_publication_context:
                    score += 100

                # Front page is preferred over page 2.
                if page_index == 0:
                    score += 20
                else:
                    score += 10

                # Short metadata/front-matter blocks are safer than prose.
                if text_len <= 80:
                    score += 20
                elif text_len <= 160:
                    score += 10
                else:
                    score -= 30

                # Typical metadata-like block types get a mild boost.
                if block_type in {
                    "text",
                    "header",
                    "footer",
                    "aside_text",
                    "title",
                }:
                    score += 5

                # A standalone date is permitted only as a weak fallback.
                normalized_plain = re.sub(r"\s+", " ", plain).strip(" .,:;-–—")
                normalized_value = re.sub(r"\s+", " ", value).strip()

                if normalized_plain.lower() == normalized_value.lower():
                    score += 25

                # Without publication semantics, avoid long sentence-like text.
                if not has_publication_context:
                    if text_len > 100:
                        continue
                    if plain.count(".") >= 2:
                        continue

                candidates.append((score, order, value))
                order += 1

    if not candidates:
        return []

    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_score, _, best_value = candidates[0]

    # Weak/noisy date candidates are not emitted.
    if best_score < 35:
        return []

    results = []
    _append_result(results, pdf, "pub_date", best_value)
    return results

def _extract_section(data: dict, pdf) -> list[tuple]:
    """
    Section headings:
      - use MinerU semantic title blocks with hierarchy level=2;
      - preserve each section occurrence independently;
      - exclude only clear non-section metadata/caption structures;
      - remove conventional section-number prefixes because eSciBench
        evaluates the section title separately from its number.
    """
    results = []

    metadata_heading_re = re.compile(
        r"^\s*(?:"
        r"keywords?"
        r"|key\s+words?"
        r"|contents?"
        r")\s*[:.]?\s*$",
        re.IGNORECASE,
    )

    caption_heading_re = re.compile(
        r"^\s*(?:table|fig(?:ure)?)\s*[\dIVXLCDM]+(?:[.:)]|\s)",
        re.IGNORECASE,
    )

    internal_enumeration_re = re.compile(
        r"^\s*\d+\)\s+"
    )

    arabic_number_re = re.compile(
        r"^\s*\d+(?:\.\d+)*(?:[.)])?\s+"
    )

    roman_number_re = re.compile(
        r"^\s*[IVXLCDM]+[.)]\s+",
        re.IGNORECASE,
    )

    letter_number_re = re.compile(
        r"^\s*[A-Z][.)]\s+"
    )

    for page in data.get("pdf_info", []):
        for block in page.get("para_blocks", []) or []:
            if not isinstance(block, dict):
                continue

            if block.get("type") != "title":
                continue

            if block.get("level") != 2:
                continue

            value = _clean_text(_block_text(block))
            if not value:
                continue

            if metadata_heading_re.match(value):
                continue

            if caption_heading_re.match(value):
                continue

            # Number-parenthesis headings are commonly lower-level
            # enumerated items rather than scientific section headings.
            if internal_enumeration_re.match(value):
                continue

            value = arabic_number_re.sub("", value, count=1)
            value = roman_number_re.sub("", value, count=1)
            value = letter_number_re.sub("", value, count=1)
            value = _clean_text(value)

            if value:
                _append_result(
                    results,
                    pdf,
                    "section",
                    value,
                )

    return results



def _extract_equation(data: dict, pdf) -> list[tuple]:
    """
    Equation V2:
      - use MinerU para_blocks classified as interline_equation;
      - convert MinerU LaTeX locally to plain mathematical text;
      - remove equation tags such as \\tag{1}, which are not part of the
        official eSciBench equation ground-truth representation;
      - apply the same local normalization semantics as the official
        normalize_string without modifying benchmark/normalisation.py;
      - preserve each complete MinerU equation block as one entry;
      - merge only adjacent fragments when the first fragment is visibly
        incomplete and the following fragment is a continuation.
    """
    results = []

    equation_converter = LatexNodes2Text()

    def normalize_equation(raw: str) -> str:
        if not raw:
            return ""

        value = str(raw)

        # Equation numbers/tags are not represented in official eSciBench GT.
        value = re.sub(
            r"\\tag\s*\{[^{}]*\}",
            " ",
            value,
            flags=re.IGNORECASE,
        )

        # Convert LaTeX commands/structures to readable mathematical text.
        try:
            value = equation_converter.latex_to_text(value)
        except Exception:
            pass

        # Match the official normalize_string semantics locally.
        value = (
            value
            .lower()
            .replace("’", "'")
            .replace("“", "'")
            .replace("”", "'")
            .replace("_", " ")
            .replace("^", " ")
        )

        value = " ".join(value.split())
        return value

    def raw_is_incomplete(raw: str) -> bool:
        """
        Conservative continuation test.

        A fragment is considered incomplete only when its LaTeX structure
        clearly remains open at the end, or when it ends with an explicit
        continuation operator/separator.
        """
        if not raw:
            return False

        value = raw.strip()

        # Unbalanced structural delimiters.
        pairs = {
            "{": "}",
            "[": "]",
            "(": ")",
        }

        for left, right in pairs.items():
            if value.count(left) > value.count(right):
                return True

        # Explicit mathematical continuation markers.
        if re.search(
            r"(?:\\\\|\\begin\s*\{|[=+\-*/])\s*$",
            value,
            flags=re.IGNORECASE,
        ):
            return True

        return False

    pending_raw = None

    def emit(raw: str):
        value = normalize_equation(raw)

        if value:
            _append_result(
                results,
                pdf,
                "equation",
                value,
            )

    for page in data.get("pdf_info", []) or []:
        blocks = []

        for block in page.get("para_blocks", []) or []:
            if not isinstance(block, dict):
                continue

            if block.get("type") != "interline_equation":
                continue

            raw = _collect_content(block)
            if not raw:
                continue

            blocks.append(raw)

        i = 0
        while i < len(blocks):
            current = blocks[i]

            # If a previous fragment was waiting for a continuation,
            # merge only when this fragment starts like a continuation.
            if pending_raw is not None:
                continuation = current.lstrip()

                # Merge only when the first fragment is already known to
                # be incomplete and the next fragment has an explicit
                # continuation marker.  Do not treat "=" as a continuation
                # marker because it can begin an independent equation.
                starts_continuation = bool(
                    re.match(
                        r"^(?:"
                        r"\\\\"
                        r"|[&+*/,\-;]"
                        r"|\\(?:qquad|quad|frac|sqrt|left|right|begin)"
                        r")",
                        continuation,
                        flags=re.IGNORECASE,
                    )
                )

                if starts_continuation:
                    pending_raw = pending_raw + " " + current
                    if raw_is_incomplete(current):
                        i += 1
                        continue

                    emit(pending_raw)
                    pending_raw = None
                    i += 1
                    continue

                emit(pending_raw)
                pending_raw = None

            if raw_is_incomplete(current) and i + 1 < len(blocks):
                pending_raw = current
            else:
                emit(current)

            i += 1

        if pending_raw is not None:
            emit(pending_raw)
            pending_raw = None

    return results

def _extract_list(data: dict, pdf) -> list[tuple]:
    """
    List V4 — conservative line/block-start sequence rule.

    Generic rules:
      - inspect only MinerU para_blocks with semantic type "text";
      - recognize markers only at the beginning of a block or a new line;
      - support bullets, numeric markers (1. / 1)) and alphabetic
        markers (a. / a));
      - numeric sequences must start at 1 and increase exactly by 1;
      - alphabetic sequences must start at a and increase exactly by 1;
      - bullet sequences require at least two items;
      - support multiple items inside one MinerU text block;
      - support lists split across consecutive text blocks;
      - any ordinary intervening text block terminates the current sequence;
      - title/reference/index/equation/etc. are excluded by construction.
    """
    results = []

    line_marker_re = re.compile(
        r"(?m)^[ \t]*("
        r"[•▪◦‣][ \t]+"
        r"|(?:\d+)[\.\)][ \t]+"
        r"|(?:[A-Za-z])[\.\)][ \t]+"
        r")"
    )

    numeric_re = re.compile(r"^\s*(\d+)[\.\)]\s+$")
    alpha_re = re.compile(r"^\s*([A-Za-z])[\.\)]\s+$")
    bullet_re = re.compile(r"^\s*[•▪◦‣]\s+$")

    def marker_info(marker):
        if bullet_re.match(marker):
            return ("bullet", None)

        m = numeric_re.match(marker)
        if m:
            return ("numeric", int(m.group(1)))

        m = alpha_re.match(marker)
        if m:
            return ("alpha", ord(m.group(1).lower()) - ord("a") + 1)

        return (None, None)

    def split_items(plain):
        matches = list(line_marker_re.finditer(plain))
        if not matches:
            return []

        items = []

        for i, match in enumerate(matches):
            marker = match.group(1)
            kind, value = marker_info(marker)

            if kind is None:
                continue

            end_pos = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(plain)
            )

            item_text = plain[match.start():end_pos].strip()
            if not item_text:
                continue

            items.append({
                "kind": kind,
                "value": value,
                "text": item_text,
            })

        return items

    def valid_sequence(items):
        if len(items) < 2:
            return False

        kinds = {item["kind"] for item in items}
        if len(kinds) != 1:
            return False

        kind = items[0]["kind"]

        if kind == "bullet":
            return True

        if kind == "numeric":
            if items[0]["value"] != 1:
                return False

        elif kind == "alpha":
            if items[0]["value"] != 1:
                return False

        else:
            return False

        for prev, cur in zip(items, items[1:]):
            if cur["value"] != prev["value"] + 1:
                return False

        return True

    def emit(items):
        if not valid_sequence(items):
            return

        _append_result(
            results,
            pdf,
            "list",
            "\n".join(item["text"] for item in items),
        )

    for page in data.get("pdf_info", []) or []:
        current = []

        for block in page.get("para_blocks", []) or []:
            if not isinstance(block, dict):
                emit(current)
                current = []
                continue

            # Strict semantic filter.
            if block.get("type") != "text":
                emit(current)
                current = []
                continue

            raw = _collect_content(block)
            if not raw:
                emit(current)
                current = []
                continue

            # Preserve line boundaries for list-marker detection.
            plain = _raw_without_tags(raw)
            plain = re.sub(r"[ \t]+", " ", plain or "").strip()

            if not plain:
                emit(current)
                current = []
                continue

            items = split_items(plain)

            if not items:
                # Ordinary prose terminates a list.
                emit(current)
                current = []
                continue

            for item in items:
                if not current:
                    current = [item]
                    continue

                prev = current[-1]

                # Different marker families cannot belong to one list.
                if item["kind"] != prev["kind"]:
                    emit(current)
                    current = [item]
                    continue

                if item["kind"] == "bullet":
                    current.append(item)
                    continue

                # Numeric/alphabetic continuation.
                if item["value"] == prev["value"] + 1:
                    current.append(item)
                    continue

                # Restart/discontinuity => close previous candidate.
                emit(current)
                current = [item]

        emit(current)

    return results

def _extract_footer(data: dict, pdf) -> list[tuple]:
    """
    Footer V2:
      - use MinerU discarded blocks classified as page_footnote;
      - restrict candidates to the bottom 20% of the page;
      - preserve explicit copyright/legal footer material;
      - use leading superscript markers as generic footnote starts;
      - merge following unmarked page_footnote fragments on the same page
        until another explicit footnote marker begins;
      - exclude clear author-affiliation relation blocks;
      - do not treat page_number blocks as footer.
    """
    results = []

    footer_marker_re = re.compile(
        r"(?:"
        r"\bcopyright\b"
        r"|©"
        r"|\ball\s+rights\s+reserved\b"
        r")",
        re.IGNORECASE,
    )

    footnote_start_re = re.compile(
        r"^\s*<sup\b[^>]*>.*?</sup>",
        re.IGNORECASE | re.DOTALL,
    )

    affiliation_relation_re = re.compile(
        r"^.*?\b(?:are|is)\s+with\b\s+.+$",
        re.IGNORECASE | re.DOTALL,
    )

    def clean_footnote(raw: str) -> str:
        raw = footnote_start_re.sub("", raw, count=1)
        raw = _raw_without_tags(raw)
        return _clean_text(raw)

    for page in data.get("pdf_info", []) or []:
        page_size = page.get("page_size") or []
        if len(page_size) < 2:
            continue

        try:
            page_height = float(page_size[1])
        except (TypeError, ValueError):
            continue

        if page_height <= 0:
            continue

        blocks = []

        for block in page.get("discarded_blocks", []) or []:
            if not isinstance(block, dict):
                continue

            if block.get("type") != "page_footnote":
                continue

            bbox = block.get("bbox") or []
            if len(bbox) < 4:
                continue

            try:
                x0 = float(bbox[0])
                y0 = float(bbox[1])
            except (TypeError, ValueError):
                continue

            if y0 / page_height < 0.80:
                continue

            raw = _collect_content(block).strip()
            if not raw:
                continue

            blocks.append((y0, x0, raw))

        blocks.sort(key=lambda item: (item[0], item[1]))

        current_parts = []

        def flush_current():
            nonlocal current_parts

            if not current_parts:
                return

            value = clean_footnote(" ".join(current_parts))

            if value:
                _append_result(
                    results,
                    pdf,
                    "footer",
                    value,
                )

            current_parts = []

        for _, _, raw in blocks:
            plain = _clean_text(_raw_without_tags(raw))

            if not plain:
                continue

            # A mixed page-footnote block can contain a true author note
            # followed by copyright/legal footer text. Preserve only the
            # legal footer portion in that case.
            legal_match = footer_marker_re.search(raw)

            if legal_match:
                flush_current()

                legal_value = _clean_text(
                    _raw_without_tags(raw[legal_match.start():])
                )

                if legal_value:
                    _append_result(
                        results,
                        pdf,
                        "footer",
                        legal_value,
                    )

                continue

            # Page-footnote blocks that explicitly encode an
            # author-affiliation relation belong to front matter rather
            # than Footer.
            if affiliation_relation_re.match(plain):
                flush_current()
                continue

            if footnote_start_re.match(raw):
                flush_current()
                current_parts = [raw]
                continue

            # Unmarked page_footnote fragments following an explicit
            # footnote start belong to the same physical footnote.
            if current_parts:
                current_parts.append(raw)

        flush_current()

    return results


def _extract_title(data: dict, pdf) -> list[tuple]:
    """
    Article title:
      - use page 0 only;
      - select MinerU semantic blocks type='title';
      - require the document-level hierarchy level=1.

    This uses MinerU's semantic hierarchy rather than lexical
    patterns or document-specific rules.
    """
    results = []

    pages = data.get("pdf_info", [])
    if not pages:
        return results

    page0 = pages[0]

    for block in page0.get("para_blocks", []):
        if not isinstance(block, dict):
            continue

        if block.get("type") != "title":
            continue

        if block.get("level") != 1:
            continue

        _append_result(
            results,
            pdf,
            "title",
            _block_text(block),
        )

    return results


def _extract_abstract(data: dict, pdf) -> list[tuple]:
    """
    Abstract:
      extract MinerU blocks explicitly typed as 'abstract'.
    """
    results = []

    for page in data.get("pdf_info", []):
        for block in page.get("para_blocks", []):
            for node in _walk_nodes(block):
                if node.get("type") == "abstract":
                    _append_result(
                        results,
                        pdf,
                        "abstract",
                        _block_text(node),
                    )

    return results


def _extract_header(data: dict, pdf) -> list[tuple]:
    """
    Header:
      extract MinerU blocks explicitly typed as 'header' from
      discarded_blocks. Multiple header fragments on the same page
      are joined into one logical page header.
    """
    results = []

    for page in data.get("pdf_info", []):
        page_headers = []

        for block in page.get("discarded_blocks", []):
            for node in _walk_nodes(block):
                if node.get("type") == "header":
                    text = _block_text(node).strip()
                    if text:
                        page_headers.append(text)

        if page_headers:
            _append_result(
                results,
                pdf,
                "header",
                " ".join(page_headers),
            )

    return results


def _extract_caption(data: dict, pdf) -> list[tuple]:
    """
    Caption:
      extract MinerU semantic captions.

      - Caption fragments inside one table are joined.
      - Image-caption fragments belonging to one logical multi-panel
        figure are grouped around the nearest explicit Fig./Figure anchor.
      - No document-specific identifiers or fixed coordinates are used.
    """
    results = []

    def get_caption_nodes(block, allowed_types):
        found = []

        def walk(obj):
            if isinstance(obj, dict):
                if obj is not block and obj.get("type") in allowed_types:
                    text = _block_text(obj).strip()
                    if text:
                        found.append((obj, text))
                    return

                for value in obj.values():
                    walk(value)

            elif isinstance(obj, list):
                for value in obj:
                    walk(value)

        walk(block)
        return found

    def center(node):
        bbox = node.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            try:
                return (
                    (float(bbox[0]) + float(bbox[2])) / 2.0,
                    (float(bbox[1]) + float(bbox[3])) / 2.0,
                )
            except (TypeError, ValueError):
                pass
        return (0.0, 0.0)

    def compact(text):
        return re.sub(r"\s+", "", text).lower()

    figure_anchor_re = re.compile(
        r"^\s*fig(?:ure)?[\s.:\-]*\d+\b",
        re.IGNORECASE,
    )

    for page in data.get("pdf_info", []):
        image_fragments = []

        for block_order, block in enumerate(page.get("para_blocks", [])):
            block_type = block.get("type")

            # One logical caption per table object.
            if block_type == "table":
                captions = get_caption_nodes(
                    block,
                    {"table_caption"},
                )

                texts = []
                seen = set()

                for _, text in captions:
                    key = compact(text)
                    if key and key not in seen:
                        seen.add(key)
                        texts.append(text)

                if texts:
                    _append_result(
                        results,
                        pdf,
                        "caption",
                        " ".join(texts),
                    )

            elif block_type in {"image", "chart", "figure"}:
                captions = get_caption_nodes(
                    block,
                    {
                        "image_caption",
                        "chart_caption",
                        "figure_caption",
                    },
                )

                for local_order, (node, text) in enumerate(captions):
                    x, y = center(node)

                    image_fragments.append(
                        {
                            "text": text,
                            "x": x,
                            "y": y,
                            "block_order": block_order,
                            "local_order": local_order,
                            "anchor": bool(figure_anchor_re.match(text)),
                        }
                    )

        if not image_fragments:
            continue

        anchors = [
            i
            for i, frag in enumerate(image_fragments)
            if frag["anchor"]
        ]

        # Without an explicit figure-number anchor, keep MinerU's
        # semantic caption blocks independent.
        if not anchors:
            for frag in image_fragments:
                _append_result(
                    results,
                    pdf,
                    "caption",
                    frag["text"],
                )
            continue

        groups = {i: [] for i in anchors}

        # Associate every fragment with the closest numbered figure
        # caption using relative page layout.
        for i, frag in enumerate(image_fragments):
            nearest = min(
                anchors,
                key=lambda a: (
                    abs(image_fragments[a]["y"] - frag["y"]),
                    abs(
                        image_fragments[a]["block_order"]
                        - frag["block_order"]
                    ),
                    abs(a - i),
                ),
            )
            groups[nearest].append(i)

        for anchor_i in anchors:
            member_ids = groups[anchor_i]

            # Restore natural reading order of all pieces assigned to
            # this logical figure.
            member_ids.sort(
                key=lambda i: (
                    image_fragments[i]["y"],
                    image_fragments[i]["x"],
                    image_fragments[i]["block_order"],
                    image_fragments[i]["local_order"],
                )
            )

            texts = []
            compact_texts = []

            for i in member_ids:
                text = image_fragments[i]["text"]
                key = compact(text)

                if not key:
                    continue

                # Avoid emitting a short fragment separately when it is
                # already contained in a fuller caption fragment.
                if any(
                    key == old or key in old
                    for old in compact_texts
                ):
                    continue

                # If this fuller fragment contains an earlier shorter
                # fragment, replace the shorter duplicate.
                keep_texts = []
                keep_compact = []

                for old_text, old_key in zip(texts, compact_texts):
                    if old_key in key:
                        continue
                    keep_texts.append(old_text)
                    keep_compact.append(old_key)

                texts = keep_texts
                compact_texts = keep_compact

                texts.append(text)
                compact_texts.append(key)

            if texts:
                _append_result(
                    results,
                    pdf,
                    "caption",
                    " ".join(texts),
                )

    return results

def _extract_reference(data: dict, pdf) -> list[tuple]:
    """
    Bibliographical references.

    MinerU usually emits one bibliographical entry per ``ref_text`` block.
    Occasionally a single reference is split into consecutive ``ref_text``
    blocks, including across a page boundary.  Merge only when generic
    textual evidence indicates that the previous entry is incomplete and
    the following block is not an explicit new reference.
    """

    def explicit_reference_start(text: str) -> bool:
        text = text.strip()

        # Explicit numbered bibliography markers.
        if re.match(r"^(?:\[\s*\d+\s*\]|\(\s*\d+\s*\)|\d+\s*[.)])", text):
            return True

        # Repeated-author bibliography notation, e.g. "—. 2013, ...".
        if re.match(r"^[—–-]\s*\.\s*(?:18|19|20)\d{2}[a-z]?\b", text):
            return True

        return False

    def looks_like_independent_reference(text: str) -> bool:
        text = text.strip()

        if explicit_reference_start(text):
            return True

        # Unnumbered bibliographies normally identify an entry by an
        # author-like prefix followed fairly early by a publication year.
        prefix = text[:180]
        if re.search(r"\b(?:18|19|20)\d{2}[a-z]?\b", prefix):
            return True

        return False

    def should_merge(previous: str, current: str) -> bool:
        previous = previous.strip()
        current = current.strip()

        if not previous or not current:
            return False

        # Never absorb a block carrying an explicit new-reference marker.
        if explicit_reference_start(current):
            return False

        # Strong evidence that the previous reference was cut mid-entry.
        if re.search(r"(?:\band|&|;|,|:)\s*$", previous, flags=re.IGNORECASE):
            return True

        # A previous block without terminal punctuation can continue into a
        # block that does not itself look like a complete new citation.
        if not re.search(r"[.!?]\s*$", previous):
            if not looks_like_independent_reference(current):
                return True

        return False

    raw_refs = []

    for page in data.get("pdf_info", []):
        for block in page.get("para_blocks", []):
            for node in _walk_nodes(block):
                if node.get("type") == "ref_text":
                    text = _block_text(node).strip()
                    if text:
                        raw_refs.append(text)

    merged_refs = []

    for text in raw_refs:
        if merged_refs and should_merge(merged_refs[-1], text):
            merged_refs[-1] = f"{merged_refs[-1]} {text}".strip()
        else:
            merged_refs.append(text)

    results = []
    for text in merged_refs:
        _append_result(
            results,
            pdf,
            "reference",
            text,
        )

    return results



# ----------------------------------------------------------------------
# TABLE V2
# Generic MinerU HTML-table parsing.
#
# - Operates only on the semantic top-level table block selected by
#   _extract_table().
# - Reads td/th cells in HTML order.
# - Converts <eq> LaTeX generically to plain scientific text.
# - Does not contain document names, expected values, coordinates,
#   article-specific phrases, or Dev5-specific conditions.
# ----------------------------------------------------------------------

_LATEX_TO_TEXT = LatexNodes2Text()


def _normalize_table_latex_source(latex: str) -> str:
    """
    Repair spacing artifacts that MinerU introduces inside scientific
    LaTeX table cells before LatexNodes2Text conversion.

    This operates only on equation content, not ordinary prose.
    """
    text = latex

    # OCR/model spacing between digits inside one mathematical token:
    #   2 6 -> 26
    #   1 0 -> 10
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)

    # Compact simple \mathrm{...} contents:
    #   \mathrm { e V } -> \mathrm{eV}
    #   \mathrm { k e V } -> \mathrm{keV}
    def compact_mathrm(match):
        content = re.sub(r"\s+", "", match.group(1))
        return r"\mathrm{" + content + "}"

    text = re.sub(
        r"\\mathrm\s*\{\s*([^{}]+?)\s*\}",
        compact_mathrm,
        text,
    )

    # Compact simple alphabetic sub/superscript tokens:
    #   _ { t h } -> _{th}
    def compact_script(match):
        operator = match.group(1)
        content = re.sub(r"\s+", "", match.group(2))
        return operator + "{" + content + "}"

    text = re.sub(
        r"([_^])\s*\{\s*([A-Za-z](?:\s+[A-Za-z])*)\s*\}",
        compact_script,
        text,
    )

    return text


def _normalize_table_math_text(text: str) -> str:
    """
    Normalize punctuation/operator spacing produced by LaTeX conversion.
    Applied only to converted table-equation text.
    """
    text = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", text)
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "-", text)

    text = re.sub(r"\s*_\s*", "_", text)
    text = re.sub(r"\s*\^\s*", "^", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*×\s*", "×", text)
    text = re.sub(r"\s*=\s*", "=", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_table_text(text: str) -> str:
    """
    Table-specific prose cleanup.

    Repairs PDF line-wrap hyphenation such as:
        re- ported -> reported

    A hyphen without following whitespace is preserved, so ordinary
    compounds such as state-of-the-art are not changed.
    """
    text = re.sub(
        r"(?<=[A-Za-z])-\s+(?=[A-Za-z])",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_bitstream_table_artifact(cells: list[str]) -> bool:
    """
    Reject table-like MinerU regions that are actually binary bitstreams.

    Generic evidence:
      - every non-empty cell contains only binary digits 0/1;
      - there are multiple cells;
      - the candidate contains a substantial amount of binary data.

    No document identity, GT text, coordinates, or expected values are used.
    """
    if not cells:
        return False

    checked = 0
    total_binary_digits = 0

    for cell in cells:
        compact = re.sub(r"\s+", "", cell)

        if not compact:
            continue

        if re.fullmatch(r"[01]+", compact) is None:
            return False

        checked += 1
        total_binary_digits += len(compact)

    return checked >= 2 and total_binary_digits >= 32


class _MinerUTableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.cells = []
        self._in_cell = False
        self._in_eq = False
        self._cell_parts = []
        self._eq_parts = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()

        if tag in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []

        elif tag == "eq" and self._in_cell:
            self._in_eq = True
            self._eq_parts = []

        elif tag == "br" and self._in_cell and not self._in_eq:
            self._cell_parts.append(" ")

    def handle_data(self, data):
        if not self._in_cell:
            return

        if self._in_eq:
            self._eq_parts.append(data)
        else:
            self._cell_parts.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()

        if tag == "eq" and self._in_cell and self._in_eq:
            latex = "".join(self._eq_parts).strip()

            if latex:
                try:
                    normalized_latex = _normalize_table_latex_source(latex)
                    converted = _LATEX_TO_TEXT.latex_to_text(normalized_latex)
                    converted = _normalize_table_math_text(converted)
                except Exception:
                    # Generic lossless fallback: retain the source math
                    # rather than silently dropping table content.
                    converted = latex

                self._cell_parts.append(converted)

            self._eq_parts = []
            self._in_eq = False

        elif tag in {"td", "th"} and self._in_cell:
            cell = " ".join(self._cell_parts)
            cell = re.sub(r"\s+", " ", cell).strip()

            if cell:
                self.cells.append(cell)

            self._cell_parts = []
            self._in_cell = False
            self._in_eq = False
            self._eq_parts = []


def _table_html_from_block(block: dict) -> str:
    """
    Return the first semantic HTML table representation found inside one
    top-level MinerU table block.

    The recursive walk stays inside that physical top-level table block;
    nested type='table' nodes are therefore not emitted as extra tables.
    """
    seen = set()

    for node in _walk_nodes(block):
        value = node.get("html")

        if not isinstance(value, str):
            continue

        if "<table" not in value.lower():
            continue

        if value in seen:
            continue

        seen.add(value)
        return value

    return ""


class _MinerUTableRowParser(_MinerUTableHTMLParser):
    """
    Parse the same normalized td/th text as _MinerUTableHTMLParser,
    while preserving HTML row boundaries.
    """
    def __init__(self):
        super().__init__()
        self.rows = []
        self._row_cells = None

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()

        if tag_lower == "tr":
            self._row_cells = []

        super().handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        tag_lower = tag.lower()

        before = len(self.cells)
        super().handle_endtag(tag)

        if (
            tag_lower in {"td", "th"}
            and self._row_cells is not None
            and len(self.cells) > before
        ):
            self._row_cells.append(self.cells[-1])

        if tag_lower == "tr" and self._row_cells is not None:
            self.rows.append(self._row_cells)
            self._row_cells = None


def _table_rows_from_html(table_html: str) -> list[list[str]]:
    parser = _MinerUTableRowParser()
    parser.feed(table_html)
    parser.close()
    return parser.rows


def _split_collapsed_decimal_sequence(text: str, count: int):
    """
    Conservatively recover N concatenated positive decimal values.

    Example structural shape:
        41.2540.95
    contains two decimal points, hence two candidate decimal values.

    Recovery is accepted only when all inferred integer parts have the
    same digit width as the first integer part. This avoids arbitrary
    GT-driven splitting.
    """
    compact = re.sub(r"\s+", "", text)

    if count < 2:
        return None

    if re.fullmatch(r"[0-9.]+", compact) is None:
        return None

    if compact.count(".") != count:
        return None

    chunks = compact.split(".")

    if len(chunks) != count + 1:
        return None

    if any(not chunk.isdigit() for chunk in chunks):
        return None

    integer_width = len(chunks[0])

    if integer_width < 1:
        return None

    values = []
    current_integer = chunks[0]

    # Every middle chunk contains:
    #   fractional digits of the previous number
    #   +
    #   integer digits of the next number.
    for bridge in chunks[1:-1]:
        if len(bridge) <= integer_width:
            return None

        fraction = bridge[:-integer_width]
        next_integer = bridge[-integer_width:]

        if not fraction or not next_integer:
            return None

        # Keep this high-confidence rather than guessing very long
        # fractional strings.
        if len(fraction) > 4:
            return None

        values.append(current_integer + "." + fraction)
        current_integer = next_integer

    final_fraction = chunks[-1]

    if not final_fraction or len(final_fraction) > 4:
        return None

    values.append(current_integer + "." + final_fraction)

    if len(values) != count:
        return None

    return values


def _split_collapsed_label_cell(text: str, count: int):
    """
    Split a collapsed textual row label only when generic textual
    boundaries produce exactly the number of subrows inferred
    independently from the numeric columns.
    """
    value = re.sub(r"\s+", " ", text).strip()

    if not value or count < 2:
        return None

    # Strategy 1:
    # preserve parenthesized citation/name groups as independent units,
    # together with any leading/trailing textual units.
    matches = list(re.finditer(r"\([^()]*\)", value))

    if matches:
        parts = []
        pos = 0

        for match in matches:
            prefix = value[pos:match.start()].strip()

            if prefix:
                parts.append(prefix)

            parts.append(match.group(0).strip())
            pos = match.end()

        tail = value[pos:].strip()

        if tail:
            parts.append(tail)

        if len(parts) == count and all(parts):
            return parts

    # Strategy 2:
    # detect a missing separator where lowercase/digit text is immediately
    # followed by a new uppercase token:
    #     "... coherenceRNES ..." -> "... coherence" + "RNES ..."
    parts = [
        part.strip()
        for part in re.split(
            r"(?<=[a-z0-9)])(?=[A-Z][A-Za-z])",
            value,
        )
        if part.strip()
    ]

    if len(parts) == count:
        return parts

    return None


def _reconstruct_collapsed_table_rows(rows: list[list[str]]):
    """
    Reconstruct high-confidence logical rows collapsed by MinerU.

    Detection and reconstruction use only the raw HTML structure:
      - one textual leading cell;
      - at least two following numeric cells;
      - every numeric cell contains the same number (>1) of concatenated
        decimal values;
      - each numeric cell can be split conservatively using consistent
        integer width;
      - the leading textual cell can independently be split into exactly
        the same number of labels.

    Rows that do not satisfy all conditions remain unchanged.
    """
    output = []
    changed = False

    for row in rows:
        if len(row) < 3:
            output.extend(row)
            continue

        label_cell = row[0]
        numeric_cells = row[1:]

        decimal_counts = []

        for cell in numeric_cells:
            compact = re.sub(r"\s+", "", cell)

            if re.fullmatch(r"[0-9.]+", compact) is None:
                decimal_counts = []
                break

            decimal_counts.append(compact.count("."))

        if (
            not decimal_counts
            or len(set(decimal_counts)) != 1
            or decimal_counts[0] < 2
        ):
            output.extend(row)
            continue

        subrow_count = decimal_counts[0]

        numeric_columns = [
            _split_collapsed_decimal_sequence(cell, subrow_count)
            for cell in numeric_cells
        ]

        if any(column is None for column in numeric_columns):
            output.extend(row)
            continue

        labels = _split_collapsed_label_cell(
            label_cell,
            subrow_count,
        )

        if labels is None:
            output.extend(row)
            continue

        # Convert column-oriented collapsed content back to logical
        # row-major reading order.
        for i in range(subrow_count):
            output.append(labels[i])

            for column in numeric_columns:
                output.append(column[i])

        changed = True

    if not changed:
        return None

    return output


def _table_cells_from_html(table_html: str) -> list[str]:
    """
    Parse non-empty td/th cells in their original HTML reading order.
    """
    parser = _MinerUTableHTMLParser()
    parser.feed(table_html)
    parser.close()
    return parser.cells

_TABLE_CAPTION_ANCHOR_RE = re.compile(
    r"^\s*table\s+(?:[A-Z]?\d+[A-Z]?|[IVXLCDM]+)\b",
    flags=re.IGNORECASE,
)


def _bbox_values(block: dict):
    """
    Return (x0, y0, x1, y1) for a valid MinerU bbox, otherwise None.
    """
    bbox = block.get("bbox")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x0, y0, x1, y1 = map(float, bbox)
    except (TypeError, ValueError):
        return None

    if x1 <= x0 or y1 <= y0:
        return None

    return x0, y0, x1, y1


def _horizontal_overlap_ratio(a: dict, b: dict) -> float:
    """
    Horizontal overlap relative to the narrower of two blocks.

    This is scale-independent and therefore does not rely on fixed page
    coordinates or a particular article layout.
    """
    box_a = _bbox_values(a)
    box_b = _bbox_values(b)

    if box_a is None or box_b is None:
        return 0.0

    ax0, _, ax1, _ = box_a
    bx0, _, bx1, _ = box_b

    overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    narrower = min(ax1 - ax0, bx1 - bx0)

    if narrower <= 0:
        return 0.0

    return overlap / narrower


def _vertically_contiguous(previous: dict, current: dict) -> bool:
    """
    Test whether two consecutive blocks belong to a locally continuous
    vertical region.

    The permitted gap is derived from the heights of the two blocks rather
    than from an absolute PDF coordinate. This keeps the rule generic across
    page sizes and layouts.
    """
    prev_box = _bbox_values(previous)
    curr_box = _bbox_values(current)

    if prev_box is None or curr_box is None:
        return False

    _, _, _, prev_y1 = prev_box
    _, curr_y0, _, curr_y1 = curr_box

    prev_y0 = prev_box[1]

    prev_height = prev_y1 - prev_y0
    curr_height = curr_y1 - curr_y0

    gap = curr_y0 - prev_y1

    # Current block must follow the previous block in reading direction.
    if gap < -0.15 * min(prev_height, curr_height):
        return False

    # A local gap no larger than half the larger neighbouring block height
    # is treated as continuous. No document-specific coordinate is used.
    max_gap = 0.5 * max(prev_height, curr_height)

    return gap <= max_gap


def _recover_caption_anchored_table_texts(page: dict) -> list[str]:
    """
    Recover table bodies that MinerU segmented as ordinary text blocks.

    Generic pattern:
      1. a standalone top-level title whose text begins with a conventional
         scientific table identifier (Table 1, Table IV, Table S2, ...);
      2. one or more immediately following top-level text blocks;
      3. those text blocks remain in the same horizontal layout region and
         are vertically contiguous;
      4. collection stops at the first structural/non-text boundary.

    The caption itself is excluded because eSciBench evaluates captions
    separately.

    No document names, article-specific vocabulary, expected GT text,
    fixed PDF coordinates, or Dev5-specific values are used.
    """
    blocks = page.get("para_blocks", [])
    recovered = []

    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue

        if block.get("type") != "title":
            continue

        anchor_text = _block_text(block)

        if not _TABLE_CAPTION_ANCHOR_RE.match(anchor_text):
            continue

        anchor_box = _bbox_values(block)

        if anchor_box is None:
            continue

        body_blocks = []
        previous = block

        for candidate in blocks[index + 1:]:
            if not isinstance(candidate, dict):
                break

            # Any structural/non-text block closes this candidate region.
            if candidate.get("type") != "text":
                break

            candidate_text = _block_text(candidate)

            # Empty text blocks do not provide recoverable table content.
            if not candidate_text:
                break

            # Caption and body must occupy the same local horizontal region.
            if _horizontal_overlap_ratio(block, candidate) < 0.5:
                break

            # Consecutive body pieces must also remain mutually aligned.
            if body_blocks and _horizontal_overlap_ratio(
                body_blocks[-1], candidate
            ) < 0.5:
                break

            if not _vertically_contiguous(previous, candidate):
                break

            body_blocks.append(candidate)
            previous = candidate

        if not body_blocks:
            continue

        table_text = " ".join(
            _block_text(body_block)
            for body_block in body_blocks
        )

        table_text = re.sub(r"\s+", " ", table_text).strip()

        if table_text:
            recovered.append(table_text)

    return recovered


def _extract_table(data: dict, pdf) -> list[tuple]:
    """
    TABLE V3 — semantic-table extraction plus generic recovery of table
    bodies that MinerU segmented as ordinary text.

    Path A — MinerU semantic tables:
      1. use each physical top-level type='table' block once;
      2. parse its nested HTML td/th cells in reading order;
      3. convert equation-cell LaTeX generically;
      4. exclude captions.

    Path B — segmentation recovery:
      1. detect a standalone conventional Table <id> caption represented
         by MinerU as a top-level title;
      2. merge immediately following text blocks while their geometry
         remains locally continuous in the same horizontal region;
      3. stop at the first structural/non-text boundary;
      4. exclude the caption itself.

    No GT-derived text, document identity, fixed coordinates, or
    article-specific content is used.
    """
    results = []

    for page in data.get("pdf_info", []):
        # V2 semantic-table path remains unchanged.
        for block in page.get("para_blocks", []):
            if not isinstance(block, dict):
                continue

            if block.get("type") != "table":
                continue

            table_html = _table_html_from_block(block)

            if table_html:
                cells = _table_cells_from_html(table_html)

                # Generic rejection of bitstream-like segmentation artifacts.
                if _is_bitstream_table_artifact(cells):
                    continue

                # Preserve HTML row structure long enough to recover
                # high-confidence collapsed logical rows.
                rows = _table_rows_from_html(table_html)
                reconstructed = _reconstruct_collapsed_table_rows(rows)

                if reconstructed is not None:
                    cells = reconstructed

                table_text = " ".join(cells)
            else:
                table_text = _block_text(block)

            table_text = _normalize_table_text(table_text)

            _append_result(
                results,
                pdf,
                "table",
                table_text,
            )

        # V3 generic recovery path for tables MinerU segmented as text.
        for table_text in _recover_caption_anchored_table_texts(page):
            table_text = _normalize_table_text(table_text)

            _append_result(
                results,
                pdf,
                "table",
                table_text,
            )

    return results

def extract_raw(base_dir: str, label: str, pdf):
    """
    eSciBench extractor interface.

    Returns:
        (flag, extraction_tuple)

    flag:
        True only for labels implemented by this MinerU wrapper version.

    extraction_tuple:
        list of:
            (pdf_name, page, label, extracted_text)
    """
    if label not in SUPPORTED_LABELS:
        return False, []

    data = _load_middle_json(pdf)

    extractors = {
        "title": _extract_title,
        "abstract": _extract_abstract,
        "header": _extract_header,
        "table": _extract_table,
        "caption": _extract_caption,
        "reference": _extract_reference,
        "author": _extract_author,
        "affiliation": _extract_affiliation,
        "email": _extract_email,
        "keyword": _extract_keyword,
        "pub_date": _extract_pub_date,
        "section": _extract_section,
        "footer": _extract_footer,
        "list": _extract_list,
        "equation": _extract_equation,
    }

    results = extractors[label](data, pdf)

    return True, results
