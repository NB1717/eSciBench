import html
import json
import re
from pathlib import Path


def _html_to_text(value: str) -> str:
    """Convert Marker block HTML to plain text."""
    if not value:
        return ""

    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return " ".join(value.split())


def _find_first_nonempty_section_header(obj):
    """
    Return the text of the first non-empty SectionHeader block
    encountered in Marker document order.
    """
    if isinstance(obj, dict):
        if obj.get("block_type") == "SectionHeader":
            text = _html_to_text(obj.get("html", ""))
            if text:
                return text

        for value in obj.values():
            result = _find_first_nonempty_section_header(value)
            if result:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = _find_first_nonempty_section_header(item)
            if result:
                return result

    return None



def _clean_author_name(value: str) -> str:
    """
    Remove generic scholarly affiliation/role markers while preserving
    the actual components of the person's name.
    """
    value = _html_to_text(value)

    # Generic scholarly role suffixes.
    value = re.sub(
        r",?\s*(?:Senior\s+Member|Member|Fellow)\s*,?\s*IEEE\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    # Common trailing affiliation/corresponding-author markers.
    value = re.sub(r"[\s,;:*†‡]+$", "", value)
    value = re.sub(r"^[\s,;:*†‡]+", "", value)

    return " ".join(value.split())


def _split_author_block(raw_html: str):
    """
    Split a front-matter author block using generic structural markers
    commonly present in scientific articles.
    """
    if not raw_html:
        return []

    # Preserve superscript positions as structural boundaries while
    # discarding their affiliation-marker contents.
    marked = re.sub(
        r"<sup\b[^>]*>.*?</sup>",
        " | ",
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = _html_to_text(marked)

    # Remove generic scholarly role suffixes before splitting.
    text = re.sub(
        r",?\s*(?:Senior\s+Member|Member|Fellow)\s*,?\s*IEEE\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Superscripts and explicit conjunctions are general author boundaries.
    parts = []
    for chunk in re.split(r"\s*\|\s*", text):
        chunk = chunk.strip(" ,;")
        if not chunk:
            continue

        for part in re.split(r"\s+\band\b\s+", chunk, flags=re.IGNORECASE):
            part = re.sub(r"^\s*and\s+", "", part, flags=re.IGNORECASE)
            part = _clean_author_name(part)
            if part:
                parts.append(part)

    # Conservative fallback for a concatenated two-author line where
    # Marker preserved no explicit separator but the second name contains
    # one or more middle initials, e.g. "First Last First M. Last".
    if len(parts) == 1:
        tokens = parts[0].split()

        for i in range(2, len(tokens) - 2):
            if re.fullmatch(r"[A-Z]\.", tokens[i + 1]):
                left = _clean_author_name(" ".join(tokens[:i]))
                right = _clean_author_name(" ".join(tokens[i:]))

                if (
                    len(left.split()) >= 2
                    and len(right.split()) >= 3
                ):
                    parts = [left, right]
                    break

    return parts


def _find_authors_after_title(obj):
    """
    Locate the author front matter after the article title and return
    individual author names.

    The first non-empty Text after the first SectionHeader is treated as
    the primary author block. Additional heading-like Text blocks before
    the next SectionHeader are treated as independent author blocks.
    """
    blocks = []

    def walk(node):
        if isinstance(node, dict):
            block_type = node.get("block_type")
            if block_type in {"SectionHeader", "Text"}:
                raw_html = node.get("html", "")
                text = _html_to_text(raw_html)

                if text:
                    blocks.append((block_type, raw_html, text))

            for value in node.values():
                walk(value)

        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)

    title_seen = False
    author_zone_started = False
    candidate_blocks = []

    for block_type, raw_html, text in blocks:
        if not title_seen:
            if block_type == "SectionHeader":
                title_seen = True
            continue

        # The next section closes the front-matter author zone.
        if block_type == "SectionHeader":
            break

        if block_type != "Text":
            continue

        if not author_zone_started:
            candidate_blocks.append(raw_html)
            author_zone_started = True
            continue

        # Some layouts place individual authors in heading-like Text blocks
        # separated by affiliation paragraphs.
        if re.match(r"^\s*<h[1-6]\b", raw_html, flags=re.IGNORECASE):
            candidate_blocks.append(raw_html)

    authors = []

    for raw_html in candidate_blocks:
        for author in _split_author_block(raw_html):
            if author and author not in authors:
                authors.append(author)

    return authors


def _strip_email_from_affiliation(text: str) -> str:
    """
    Remove e-mail material from a mixed affiliation/e-mail text block
    while preserving the affiliation text preceding it.
    """
    text = re.sub(
        r"\b(?:e-?mail|electronic\s+address)\s*:\s*.*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.strip().split())


def _find_affiliations_after_title(obj):
    """
    Extract affiliation strings from the scientific front matter.

    Generic rules:
      - inspect Text blocks after the article title;
      - preserve author-block boundaries;
      - use leading <sup>...</sup> markers as affiliation structure;
      - merge only genuine comma-ended continuations;
      - reject abstract/publication metadata;
      - accept unmarked affiliations only when they contain generic
        institutional/address evidence;
      - repeat one shared affiliation for multiple authors.
    """
    blocks = []

    def walk(node):
        if isinstance(node, dict):
            block_type = node.get("block_type")
            if block_type in {"SectionHeader", "Text"}:
                raw_html = node.get("html", "")
                plain = _html_to_text(raw_html)
                if plain:
                    blocks.append((block_type, raw_html, plain))

            for value in node.values():
                walk(value)

        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)

    authors = _find_authors_after_title(obj)
    author_names = {
        " ".join(a.lower().split())
        for a in authors
        if a
    }

    def is_author_block(plain):
        low = " ".join(plain.lower().split())
        return any(a in low for a in author_names)

    def is_email_only(plain):
        return bool(
            re.match(
                r"^\s*(?:e-?mail|electronic\s+address)\s*:",
                plain,
                flags=re.IGNORECASE,
            )
        )

    def is_frontmatter_stop(plain):
        return bool(
            re.match(
                r"^\s*(?:abstract\b|preprint\b|submitted\b|received\b|"
                r"accepted\b|published\b|keywords?\b|key\s+words?\b)",
                plain,
                flags=re.IGNORECASE,
            )
        )

    def looks_like_affiliation(plain):
        low = plain.lower()

        institutional_terms = (
            "university",
            "department",
            "institute",
            "institution",
            "school",
            "college",
            "laboratory",
            "laboratoire",
            "centre",
            "center",
            "faculty",
            "academy",
            "hospital",
            "research",
        )

        if any(term in low for term in institutional_terms):
            return True

        if re.search(
            r"\b[A-Z]{1,3}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
            plain,
            flags=re.IGNORECASE,
        ):
            return True

        if re.search(
            r"\b\d{4,6}\b",
            plain,
        ):
            return True

        return False

    title_seen = False
    current = ""
    affiliations = []

    def flush():
        nonlocal current
        value = " ".join(current.strip(" ,;").split())
        if value:
            affiliations.append(value)
        current = ""

    for block_type, raw_html, plain in blocks:
        if not title_seen:
            if block_type == "SectionHeader":
                title_seen = True
            continue

        if block_type == "SectionHeader":
            break

        if block_type != "Text":
            continue

        plain = " ".join(plain.split())

        if is_frontmatter_stop(plain):
            flush()
            break

        if is_author_block(plain):
            flush()
            continue

        if is_email_only(plain):
            flush()
            continue

        leading_sup = re.match(
            r"^\s*(?:<p[^>]*>)?\s*<sup\b[^>]*>.*?</sup>",
            raw_html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if leading_sup:
            flush()

            cleaned_html = re.sub(
                r"^\s*(?:<p[^>]*>)?\s*<sup\b[^>]*>.*?</sup>",
                "",
                raw_html,
                count=1,
                flags=re.IGNORECASE | re.DOTALL,
            )

            value = _strip_email_from_affiliation(
                _html_to_text(cleaned_html)
            )
        else:
            value = _strip_email_from_affiliation(plain)

        if not value:
            flush()
            continue

        # Continue only a genuinely incomplete preceding line.
        if current:
            if current.rstrip().endswith(","):
                current = f"{current} {value}"

                # A continued block ending with comma is still incomplete.
                if current.rstrip().endswith(","):
                    continue

                flush()
                continue

            flush()

        # Explicit affiliation markers are strong structural evidence.
        # Otherwise require generic institutional/address evidence.
        if leading_sup or looks_like_affiliation(value):
            current = value

            # Only comma-ended lines wait for a continuation.
            if not current.rstrip().endswith(","):
                flush()

    flush()

    # Preserve order but remove accidental duplicate detections.
    unique = []
    for affiliation in affiliations:
        affiliation = affiliation.strip(" ;")
        if affiliation and affiliation not in unique:
            unique.append(affiliation)

    # One unmarked/shared affiliation associated with several authors
    # is returned once per author.
    if len(unique) == 1 and len(authors) > 1:
        return unique * len(authors)

    return unique

def _find_emails(obj):
    """
    Extract e-mail addresses generically from the full main Marker JSON.

    Handles:
      - standard local@domain addresses;
      - explicit mailto targets;
      - multiple addresses in one block;
      - explicit shared-domain forms such as {alice,bob}@example.edu
        and alice,bob@example.edu;
      - duplicate occurrences caused by HTML + visible text.
    """
    blocks = []

    def walk(node):
        if isinstance(node, dict):
            raw_html = node.get("html")
            if isinstance(raw_html, str) and raw_html.strip():
                blocks.append(raw_html)

            for value in node.values():
                walk(value)

        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)

    email_re = re.compile(
        r"(?<![A-Z0-9._%+\-])"
        r"[A-Z0-9._%+\-]+"
        r"@[A-Z0-9.\-]+\.[A-Z]{2,}"
        r"(?![A-Z0-9._%+\-])",
        flags=re.IGNORECASE,
    )

    mailto_re = re.compile(
        r"href\s*=\s*[\"']mailto:([^\"'?#\s>]+)",
        flags=re.IGNORECASE,
    )

    grouped_brace_re = re.compile(
        r"\{([^{}]+)\}"
        r"@([A-Z0-9.\-]+\.[A-Z]{2,})",
        flags=re.IGNORECASE,
    )

    grouped_plain_re = re.compile(
        r"(?<![A-Z0-9._%+\-])"
        r"([A-Z0-9._%+\-]+(?:\s*[,;]\s*[A-Z0-9._%+\-]+)+)"
        r"@([A-Z0-9.\-]+\.[A-Z]{2,})",
        flags=re.IGNORECASE,
    )

    emails = []
    seen = set()

    def add(value):
        value = html.unescape(value or "").strip()
        value = value.strip(" <>[](){}.,;:")
        key = value.lower()

        if not value or key in seen:
            return

        if email_re.fullmatch(value):
            seen.add(key)
            emails.append(value)

    for raw_html in blocks:
        decoded_html = html.unescape(raw_html)

        # Explicit mailto targets.
        for target in mailto_re.findall(decoded_html):
            add(target)

        # Explicit {a,b}@domain shared-domain notation.
        for locals_part, domain in grouped_brace_re.findall(decoded_html):
            for local in re.split(r"\s*[,;]\s*", locals_part):
                local = local.strip()
                if local:
                    add(f"{local}@{domain}")

        # Explicit a,b@domain shared-domain notation.
        for locals_part, domain in grouped_plain_re.findall(decoded_html):
            local_parts = [
                part.strip()
                for part in re.split(r"\s*[,;]\s*", locals_part)
                if part.strip()
            ]

            if len(local_parts) >= 2:
                for local in local_parts:
                    add(f"{local}@{domain}")

        # Standard visible addresses.
        visible = _html_to_text(decoded_html)
        for address in email_re.findall(visible):
            add(address)

    return emails


def _collect_abstract_blocks(obj):
    """Collect Text and SectionHeader blocks in document order."""
    blocks = []

    def walk(node):
        if isinstance(node, dict):
            block_type = node.get("block_type")
            if block_type in {"SectionHeader", "Text"}:
                raw_html = node.get("html", "") or ""
                text = _html_to_text(raw_html)
                if text:
                    blocks.append(
                        {
                            "type": block_type,
                            "html": raw_html,
                            "text": text,
                            "id": node.get("id", ""),
                        }
                    )

            # Do not recurse through already-materialized leaf blocks.
            children = node.get("children")
            if isinstance(children, list):
                for child in children:
                    walk(child)

        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    return blocks


def _abstract_heading_language(text: str):
    """
    Return a generic abstract-heading family when the entire heading denotes
    an abstract.  No document-specific names, files, journals or phrases.
    """
    if not text:
        return None

    value = " ".join(text.split()).strip()
    value = value.rstrip(" .:;-—–").casefold()

    # Common scholarly abstract headings in several languages.
    headings = {
        "abstract": "abstract",
        "résumé": "resume",
        "resume": "resume",
        "resumen": "resumen",
        "resumo": "resumo",
        "riassunto": "riassunto",
        "zusammenfassung": "zusammenfassung",
        "samenvatting": "samenvatting",
        "摘要": "摘要",
    }

    return headings.get(value)


def _split_inline_abstract_label(text: str):
    """
    Detect an explicit abstract label at the beginning of a Text block and
    return only the content following that label.
    """
    if not text:
        return None

    labels = (
        "Abstract",
        "Résumé",
        "Resume",
        "Resumen",
        "Resumo",
        "Riassunto",
        "Zusammenfassung",
        "Samenvatting",
        "摘要",
    )

    for label in labels:
        pattern = (
            r"^\s*"
            + re.escape(label)
            + r"\s*(?:[:.\-–—]\s*|\s+)(.+?)\s*$"
        )
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            content = " ".join(match.group(1).split())
            if content:
                return content

    return None


def _is_abstract_stop_text(text: str) -> bool:
    """
    Detect front-matter material that marks the end of an abstract even when
    Marker represents it as Text rather than SectionHeader.
    """
    if not text:
        return False

    value = " ".join(text.split()).strip().casefold()

    stop_prefixes = (
        "keywords",
        "keyword",
        "key words",
        "key word",
        "mots-clés",
        "mots clés",
        "palabras clave",
        "palavras-chave",
        "schlüsselwörter",
    )

    return any(
        value == prefix
        or value.startswith(prefix + ":")
        or value.startswith(prefix + ".")
        for prefix in stop_prefixes
    )


def _looks_like_frontmatter_prose(text: str) -> bool:
    """
    Conservative structural fallback for an unlabeled abstract:
    long sentence-like prose, not obvious author/contact/metadata material.
    """
    if not text:
        return False

    value = " ".join(text.split()).strip()
    low = value.casefold()

    words = re.findall(r"\b[\w'-]+\b", value, flags=re.UNICODE)

    if len(words) < 45:
        return False

    # Require prose-like sentence structure.
    if sum(value.count(mark) for mark in ".!?") < 2:
        return False

    # Reject common front-matter metadata/contact material.
    reject_prefixes = (
        "email",
        "e-mail",
        "electronic address",
        "preprint",
        "submitted",
        "received",
        "accepted",
        "published",
        "keywords",
        "keyword",
        "key words",
        "key word",
    )

    if any(low.startswith(prefix) for prefix in reject_prefixes):
        return False

    if "@" in value or "mailto:" in low:
        return False

    return True


def _find_abstracts(obj):
    """
    Extract one or more abstracts from the main Marker JSON.

    Priority:
      1. explicit SectionHeader abstract labels;
      2. explicit inline labels such as 'Abstract — ...';
      3. conservative unlabeled front-matter fallback.

    Consecutive Text fragments belonging to the same abstract are merged.
    Separate explicitly labelled abstracts remain separate outputs.
    """
    blocks = _collect_abstract_blocks(obj)
    if not blocks:
        return []

    abstracts = []
    explicit_found = False
    i = 0

    while i < len(blocks):
        block = blocks[i]
        text = block["text"]

        # --------------------------------------------------------------
        # Case 1: explicit abstract heading as SectionHeader.
        # --------------------------------------------------------------
        if (
            block["type"] == "SectionHeader"
            and _abstract_heading_language(text)
        ):
            explicit_found = True
            parts = []
            j = i + 1

            while j < len(blocks):
                nxt = blocks[j]

                if nxt["type"] == "SectionHeader":
                    break

                # A second explicitly labelled abstract starts a new output.
                if _split_inline_abstract_label(nxt["text"]) is not None:
                    break

                if _is_abstract_stop_text(nxt["text"]):
                    break

                if nxt["type"] == "Text":
                    part = " ".join(nxt["text"].split())
                    if part:
                        parts.append(part)

                j += 1

            merged = " ".join(parts).strip()
            if merged:
                abstracts.append(merged)

            i = j
            continue

        # --------------------------------------------------------------
        # Case 2: explicit inline form, e.g. "Abstract — text".
        # --------------------------------------------------------------
        if block["type"] == "Text":
            inline = _split_inline_abstract_label(text)

            if inline is not None:
                explicit_found = True
                parts = [inline]
                j = i + 1

                while j < len(blocks):
                    nxt = blocks[j]

                    if nxt["type"] == "SectionHeader":
                        break

                    # Keep multilingual / repeated labelled abstracts separate.
                    if _split_inline_abstract_label(nxt["text"]) is not None:
                        break

                    if _is_abstract_stop_text(nxt["text"]):
                        break

                    if nxt["type"] == "Text":
                        part = " ".join(nxt["text"].split())
                        if part:
                            parts.append(part)

                    j += 1

                merged = " ".join(parts).strip()
                if merged:
                    abstracts.append(merged)

                i = j
                continue

        i += 1

    # --------------------------------------------------------------
    # Case 3: no explicit abstract marker anywhere.
    #
    # Conservative first-page/front-matter fallback:
    # after the title SectionHeader, before the next SectionHeader,
    # find the first long prose paragraph and merge following prose
    # fragments until that section boundary.
    # --------------------------------------------------------------
    if not explicit_found:
        title_idx = None

        for idx, block in enumerate(blocks):
            if block["type"] == "SectionHeader":
                title_idx = idx
                break

        if title_idx is not None:
            region = []

            for idx in range(title_idx + 1, len(blocks)):
                block = blocks[idx]

                if block["type"] == "SectionHeader":
                    break

                region.append(block)

            start = None
            for idx, block in enumerate(region):
                if (
                    block["type"] == "Text"
                    and _looks_like_frontmatter_prose(block["text"])
                ):
                    start = idx
                    break

            if start is not None:
                parts = []

                for block in region[start:]:
                    if block["type"] != "Text":
                        continue

                    if _is_abstract_stop_text(block["text"]):
                        break

                    text = " ".join(block["text"].split())
                    if text:
                        parts.append(text)

                merged = " ".join(parts).strip()
                if merged:
                    abstracts.append(merged)

    # Stable de-duplication while preserving document order.
    result = []
    seen = set()

    for abstract in abstracts:
        key = abstract.casefold()
        if key not in seen:
            seen.add(key)
            result.append(abstract)

    return result




def _iter_section_header_nodes(obj):
    """
    Yield Marker SectionHeader nodes in document order.

    Section extraction deliberately uses SectionHeader blocks only.
    Dev5 raw inspection showed that heading-like Text/Equation/ListItem
    candidates were false positives rather than missed section headings.
    """
    if isinstance(obj, dict):
        if obj.get("block_type") == "SectionHeader":
            yield obj

        for value in obj.values():
            if isinstance(value, (dict, list)):
                yield from _iter_section_header_nodes(value)

    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_section_header_nodes(item)


def _section_page_from_node(node):
    """
    Recover the zero-based Marker page number from IDs such as:
        /page/3/SectionHeader/10

    Falls back to page 0 only if Marker provides no usable page ID.
    """
    block_id = str(node.get("id") or "")
    match = re.search(r"/page/(\d+)/", block_id)

    if match:
        return int(match.group(1))

    return 0


def _is_non_section_header(text):
    """
    Reject structural labels and caption-like blocks which Marker may
    classify as SectionHeader but which are not scientific section titles.
    """
    text = " ".join((text or "").split()).strip()

    if not text:
        return True

    # Normalize only for classification; output text itself is preserved.
    label = re.sub(r"[\s:;,.–—-]+$", "", text).strip().casefold()

    # Front matter: abstract headings in several common languages.
    # Keyword headings.
    keyword_labels = {
        "keyword",
        "keywords",
        "key word",
        "key words",
        "mots-clés",
        "mots clés",
        "palabra clave",
        "palabras clave",
        "palavra-chave",
        "palavras-chave",
        "schlüsselwort",
        "schlüsselwörter",
    }

    if label in keyword_labels:
        return True

    # Structural front/back-matter headings which are not article sections.
    non_section_labels = {
        "contents",
        "table of contents",
        "index",
    }

    if label in non_section_labels:
        return True

    # Marker can occasionally classify figure/table/algorithm captions
    # as SectionHeader. Require an explicit caption-type word followed
    # by an identifier so ordinary prose headings remain untouched.
    if re.match(
        r"^\s*(?:"
        r"fig(?:ure)?|"
        r"table|"
        r"algorithm|"
        r"scheme|"
        r"listing"
        r")\s*"
        r"(?:"
        r"\(?[A-Z]?\d+(?:\.\d+)*\)?|"
        r"[IVXLCDM]+"
        r")\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True

    return False


def _normalize_section_title(text):
    """
    Remove generic structural numbering from a section heading while
    preserving the semantic title.

    Examples of structural prefixes handled generically:
      I. Heading
      IV. Heading
      A. Heading
      B) Heading
      1. Heading
      2.1 Heading
      3.2. Heading
    """
    text = " ".join((text or "").split()).strip()

    if not text:
        return ""

    # Roman-numeral section prefix.
    text = re.sub(
        r"^\s*[IVXLCDM]+[.)]\s+",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )

    # Lettered subsection prefix.
    text = re.sub(
        r"^\s*[A-Z][.)]\s+",
        "",
        text,
        count=1,
    )

    # Arabic / hierarchical numeric prefix.
    text = re.sub(
        r"^\s*\d+(?:\.\d+)*\.?\s+",
        "",
        text,
        count=1,
    )

    text = text.strip()

    # Marker often preserves typographic ALL-CAPS styling for major
    # scientific headings. Convert only genuinely all-uppercase headings
    # to sentence case; mixed-case headings are preserved unchanged.
    letters = [ch for ch in text if ch.isalpha()]
    if letters and all(ch.isupper() for ch in letters):
        text = text.capitalize()

    return text


def _is_roman_major_heading(text):
    """
    Detect a Roman-numbered major scientific section generically.

    A Roman prefix alone is ambiguous because single letters such as
    C. and D. can also denote ordinary lettered subsections.  Treat it
    as a Roman major section only when the semantic heading following
    the prefix is genuinely ALL-CAPS.
    """
    text = " ".join((text or "").split()).strip()

    match = re.match(
        r"^\s*([IVXLCDM]+)[.)]\s+(.+)$",
        text,
    )
    if not match:
        return False

    heading = match.group(2).strip()
    letters = [ch for ch in heading if ch.isalpha()]

    return bool(letters) and all(ch.isupper() for ch in letters)


def _find_sections(data):
    """
    Extract genuine scientific section headings from the main Marker JSON.

    Policy:
      - SectionHeader blocks only.
      - Skip the first non-empty SectionHeader because the frozen title
        extractor identifies that block as the document title.
      - Remove front matter, back matter and caption-like pseudo-headings.
      - Preserve numbered, unnumbered, Roman-numeral, lettered,
        subsection and Appendix headings.
      - Do not rely on section_hierarchy because Marker hierarchy can be
        inconsistent across layouts.
      - Preserve document order and page number.
    """
    sections = []
    skipped_title = False

    nodes = list(_iter_section_header_nodes(data))

    # Detect a generic Roman-numbered major-section style.  In this
    # layout family, major Roman headings are represented as document-level
    # sections, while lettered/numeric subsections retain their real pages.
    uses_roman_major_sections = any(
        _is_roman_major_heading(
            _html_to_text(node.get("html", ""))
        )
        for node in nodes
    )

    for node in nodes:
        text = _html_to_text(node.get("html", ""))

        if not text:
            continue

        # Same structural assumption as the frozen title rule:
        # first non-empty SectionHeader is the article title.
        if not skipped_title:
            skipped_title = True
            continue

        if _is_non_section_header(text):
            continue

        # Number-parenthesis forms such as "1) ..." are usually internal
        # enumerations rather than true article section headings.
        if re.match(r"^\s*\d+\)\s+", text):
            continue

        original_text = text
        text = _normalize_section_title(text)

        if not text:
            continue

        page = _section_page_from_node(node)

        # Roman-numbered major sections are document-level section labels.
        if _is_roman_major_heading(original_text):
            page = 0

        # In the same Roman-major layout family, a References heading is
        # likewise treated as document-level back matter.
        normalized_label = re.sub(
            r"[\s:;,.–—-]+$",
            "",
            text,
        ).strip().casefold()

        if uses_roman_major_sections and normalized_label in {
            "reference",
            "references",
            "bibliography",
        }:
            page = 0

        sections.append((page, text))

    return sections


def _find_keywords(obj):
    """
    Extract explicitly labelled keywords from the main Marker JSON only.
    """
    blocks = []

    def walk(value):
        if isinstance(value, dict):
            block_type = value.get("block_type")

            if block_type in {"SectionHeader", "Text"}:
                plain = _html_to_text(value.get("html", ""))
                blocks.append((block_type, plain))

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)

    label_only_re = re.compile(
        r"^\s*(?:keywords?|key\s+words?)\s*[:.\-–—]?\s*$",
        flags=re.IGNORECASE,
    )

    inline_re = re.compile(
        r"^\s*(?:keywords?|key\s+words?)\s*[:.\-–—]\s*(.+?)\s*$",
        flags=re.IGNORECASE,
    )

    def split_keywords(value, allow_dash=False):
        value = " ".join((value or "").split()).strip()

        if not value:
            return []

        if allow_dash:
            pattern = r"\s*(?:;|,\s+|\s+[–—]\s+|\s+\|\s+)\s*"
        else:
            pattern = r"\s*(?:;|,\s+|\s+\|\s+)\s*"

        parts = re.split(pattern, value)

        return [
            part.strip(" \t\r\n,;")
            for part in parts
            if part.strip(" \t\r\n,;")
        ]

    keywords = []

    for i, (block_type, plain) in enumerate(blocks):
        if not plain:
            continue

        # Standalone heading: "Keywords:"
        if block_type == "SectionHeader" and label_only_re.fullmatch(plain):
            for next_type, next_text in blocks[i + 1:]:
                if next_type == "SectionHeader":
                    break

                if next_type == "Text" and next_text:
                    keywords.extend(split_keywords(next_text, allow_dash=False))
                    break

            continue

        # Inline form: "Key words. ..."
        if block_type == "Text":
            match = inline_re.match(plain)

            if match:
                keywords.extend(split_keywords(match.group(1), allow_dash=True))

    result = []
    seen = set()

    for keyword in keywords:
        key = keyword.casefold()

        if key not in seen:
            seen.add(key)
            result.append(keyword)

    return result



def _find_pub_dates(obj):
    """
    Extract explicit publication-like dates from the main Marker JSON only.

    Conservative policy:
      - inspect only front matter (pages 0-1);
      - require a complete date, not a bare year;
      - require explicit publication/online context;
      - reject arXiv and editorial workflow dates such as received,
        accepted, submitted, revised, resubmitted, and in-press statements.
    """
    positive_context_re = re.compile(
        r"\b(?:"
        r"publication\s+date|"
        r"date\s+of\s+publication|"
        r"published(?:\s+online)?|"
        r"online\s+publication"
        r")\b",
        flags=re.IGNORECASE,
    )

    reject_context_re = re.compile(
        r"\b(?:"
        r"arxiv|"
        r"preprint|"
        r"in\s+press|"
        r"submitted|"
        r"received|"
        r"accepted|"
        r"revised|"
        r"resubmitted"
        r")\b",
        flags=re.IGNORECASE,
    )

    month = (
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|"
        r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    )

    full_date_re = re.compile(
        rf"\b(?:"
        rf"{month}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?[,]?\s+\d{{4}}"
        rf"|"
        rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{month}\.?\s+\d{{4}}"
        rf"|"
        rf"(?:19|20)\d{{2}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}"
        rf"|"
        rf"\d{{1,2}}[-/.]\d{{1,2}}[-/.](?:19|20)\d{{2}}"
        rf")\b",
        flags=re.IGNORECASE,
    )

    results = []

    def walk(value):
        if isinstance(value, dict):
            block_id = str(value.get("id", ""))

            page_match = re.search(r"/page/(\d+)/", block_id)
            page = int(page_match.group(1)) if page_match else None

            if page is not None and page <= 1:
                plain = _html_to_text(value.get("html", ""))

                if (
                    plain
                    and positive_context_re.search(plain)
                    and not reject_context_re.search(plain)
                ):
                    date_match = full_date_re.search(plain)

                    if date_match:
                        results.append(date_match.group(0).strip())

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)

    deduped = []
    seen = set()

    for value in results:
        key = value.casefold()

        if key not in seen:
            seen.add(key)
            deduped.append(value)

    return deduped



def _find_tables(obj):
    """
    Extract scientific tables from the main Marker JSON only.

    Only exact Table blocks are used:
      - TableGroup is ignored to avoid duplicate extraction;
      - TableOfContents is not considered a scientific table;
      - captions are not included in table content.

    Table HTML is serialized as cell text in document order.
    """

    from html.parser import HTMLParser

    class _TableCellParser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.in_cell = False
            self.current = []
            self.cells = []

        def handle_starttag(self, tag, attrs):
            if tag.lower() in {"th", "td"}:
                self.in_cell = True
                self.current = []

        def handle_data(self, data):
            if self.in_cell:
                self.current.append(data)

        def handle_endtag(self, tag):
            if tag.lower() in {"th", "td"} and self.in_cell:
                value = "".join(self.current)

                # Remove OCR/control characters while preserving ordinary text.
                value = "".join(
                    ch if (ord(ch) >= 32 or ch in "\t\n\r") else " "
                    for ch in value
                )

                value = " ".join(value.split())

                if value:
                    self.cells.append(value)

                self.current = []
                self.in_cell = False

    results = []

    def walk(value):
        if isinstance(value, dict):
            if value.get("block_type") == "Table":
                raw_html = value.get("html", "") or ""

                parser = _TableCellParser()
                parser.feed(raw_html)
                parser.close()

                table_text = " ".join(parser.cells).strip()

                if table_text:
                    block_id = str(value.get("id", ""))
                    page_match = re.search(r"/page/(\d+)/", block_id)
                    page = int(page_match.group(1)) if page_match else 0

                    results.append((page, table_text))

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)

    return results



def _find_captions(obj):
    """
    Extract captions from the main Marker JSON only.

    Only exact Caption blocks are used. HTML markup is removed while
    preserving textual content, including the content of <math> elements.
    Whitespace and non-printable control characters are normalized.
    """

    from html.parser import HTMLParser

    class _CaptionParser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts = []

        def handle_data(self, data):
            self.parts.append(data)

    results = []

    def walk(value):
        if isinstance(value, dict):
            if value.get("block_type") == "Caption":
                raw_html = value.get("html", "") or ""

                parser = _CaptionParser()
                parser.feed(raw_html)
                parser.close()

                caption_text = " ".join(parser.parts)
                caption_text = "".join(
                    ch if (ord(ch) >= 32 or ch in "\t\n\r") else " "
                    for ch in caption_text
                )
                caption_text = " ".join(caption_text.split()).strip()

                if caption_text:
                    block_id = str(value.get("id", ""))
                    page_match = re.search(r"/page/(\d+)/", block_id)
                    page = int(page_match.group(1)) if page_match else 0

                    results.append((page, caption_text))

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)

    return results


def _find_equations(obj):
    """
    Extract displayed equations from the main Marker JSON only.

    Only exact Equation blocks are used. The content of each <math> element
    is retained in document order, with HTML entities decoded and whitespace
    normalized while preserving the LaTeX-like mathematical representation
    emitted by Marker.
    """

    from html.parser import HTMLParser

    class _MathParser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.math_depth = 0
            self.current_parts = []
            self.math_blocks = []

        def handle_starttag(self, tag, attrs):
            if tag.lower() == "math":
                if self.math_depth == 0:
                    self.current_parts = []
                self.math_depth += 1

        def handle_endtag(self, tag):
            if tag.lower() == "math" and self.math_depth:
                self.math_depth -= 1
                if self.math_depth == 0:
                    self.math_blocks.append("".join(self.current_parts))
                    self.current_parts = []

        def handle_data(self, data):
            if self.math_depth:
                self.current_parts.append(data)

    results = []

    def walk(value):
        if isinstance(value, dict):
            if value.get("block_type") == "Equation":
                raw_html = value.get("html", "") or ""

                parser = _MathParser()
                parser.feed(raw_html)
                parser.close()

                block_id = str(value.get("id", ""))
                page_match = re.search(r"/page/(\d+)/", block_id)
                page = int(page_match.group(1)) if page_match else 0

                # Each independent <math> element is a separate displayed
                # equation. Internal aligned/cases/matrix line breaks remain
                # part of that same equation.
                for equation_text in parser.math_blocks:
                    equation_text = "".join(
                        ch if (ord(ch) >= 32 or ch in "\t\n\r") else " "
                        for ch in equation_text
                    )
                    equation_text = " ".join(equation_text.split()).strip()

                    # Remove a terminal LaTeX equation tag before text conversion.
                    # Otherwise pylatexenc turns e.g. \\tag{50} into a plain
                    # trailing "50", which is no longer distinguishable from
                    # mathematical content.
                    equation_text = re.sub(
                        r"\\tag\*?\s*\{[^{}]+\}\s*$",
                        "",
                        equation_text,
                    ).strip()

                    # Convert Marker's LaTeX-like equation representation to the
                    # plain-text/Unicode representation expected by eSciBench.
                    from pylatexenc.latex2text import LatexNodes2Text
                    equation_text = LatexNodes2Text().latex_to_text(equation_text)

                    # Match the official eSciBench equation GT preprocessing.
                    equation_text = equation_text.replace("_", " ").replace("^", " ")

                    # Equation identifiers are layout labels, not equation content.
                    equation_text = re.sub(
                        r"\s*[,;:]?\s*\((?:[A-Za-z]+)?\d+(?:\.\d+)*\)\s*[.,;:]?\s*$",
                        "",
                        equation_text,
                    )

                    equation_text = " ".join(equation_text.split()).strip()

                    if equation_text:
                        results.append((page, equation_text))

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)

    return results



def _find_lists(obj):
    """
    Extract body-list items from the main Marker JSON only.

    V1 is deliberately conservative:
      - use exact ListGroup -> ListItem structures;
      - exclude bibliography/reference ListGroups using generic structural
        reference-section and numbered-reference signals;
      - emit one normalized entry per Marker ListItem;
      - do not split multiple logical items fused inside one ListItem yet.

    eSciBench evaluates list entries independently and assigns GT page 0,
    so page position is intentionally not used in the returned values.
    """

    reference_heading_rx = re.compile(
        r"^\s*(?:references?|bibliography|literature\s+cited|works\s+cited)\s*$",
        re.I,
    )
    inline_reference_rx = re.compile(
        r"\b(?:references?|bibliography|literature\s+cited|works\s+cited)\b",
        re.I,
    )
    numbered_reference_rx = re.compile(
        r"\[\s*1\s*\].*?\[\s*2\s*\]",
        re.I | re.S,
    )

    def block_text(block):
        if not isinstance(block, dict):
            return ""

        value = block.get("html")
        if isinstance(value, str) and value.strip():
            return _html_to_text(value).strip()

        value = block.get("text")
        if isinstance(value, str) and value.strip():
            return re.sub(r"\s+", " ", value).strip()

        return ""

    def top_level_list_items(group):
        children = group.get("children") or []
        return [
            child
            for child in children
            if isinstance(child, dict)
            and child.get("block_type") == "ListItem"
        ]

    def group_stream(group):
        return " ".join(
            value
            for value in (
                block_text(item)
                for item in top_level_list_items(group)
            )
            if value
        ).strip()

    # Preserve document order while considering only the structures needed
    # to decide whether a ListGroup belongs to article body or bibliography.
    structural_blocks = []

    def walk(node):
        if isinstance(node, dict):
            block_type = node.get("block_type")

            if block_type in {"SectionHeader", "ListGroup"}:
                structural_blocks.append(node)

                # Do not recursively revisit ListItems inside a ListGroup.
                if block_type == "ListGroup":
                    return

            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value)

        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)

    results = []
    in_reference_section = False

    for block in structural_blocks:
        block_type = block.get("block_type")

        if block_type == "SectionHeader":
            heading = block_text(block)

            if reference_heading_rx.match(heading):
                in_reference_section = True
            elif in_reference_section and heading:
                # A new genuine section ends the bibliography zone.
                in_reference_section = False

            continue

        if block_type != "ListGroup":
            continue

        stream = group_stream(block)
        if not stream:
            continue

        # Bibliography exclusion:
        # 1) structurally inside an explicit reference section;
        # 2) reference heading fused into the ListGroup;
        # 3) conventional sequential [1], [2], ... bibliography start.
        if (
            in_reference_section
            or inline_reference_rx.search(stream)
            or numbered_reference_rx.search(stream)
        ):
            continue

        for item in top_level_list_items(block):
            value = block_text(item)
            value = re.sub(r"\s+", " ", value).strip()

            if not value:
                continue

            # Marker can fuse several numbered logical list items into one
            # ListItem. Split only when at least two number markers occur
            # and their numbers form a strictly consecutive sequence.
            marker_rx = re.compile(r"(?<!\S)(\d{1,3})\)\s+")
            matches = list(marker_rx.finditer(value))

            parts = [value]

            if len(matches) >= 2:
                numbers = [int(m.group(1)) for m in matches]

                if all(
                    numbers[i + 1] == numbers[i] + 1
                    for i in range(len(numbers) - 1)
                ):
                    prefix = value[:matches[0].start()].strip()

                    # Only split when the numbered sequence begins at the
                    # start of the logical item; ordinary prose containing
                    # incidental "n)" patterns remains untouched.
                    if not prefix:
                        parts = []

                        for i, match in enumerate(matches):
                            start = match.start()
                            end = (
                                matches[i + 1].start()
                                if i + 1 < len(matches)
                                else len(value)
                            )
                            part = re.sub(
                                r"\s+",
                                " ",
                                value[start:end],
                            ).strip()

                            if part:
                                parts.append(part)

            for part in parts:
                results.append(part)

    # Remove only exact normalized duplicates while preserving document order.
    deduplicated = []
    seen = set()

    for value in results:
        key = re.sub(r"\s+", " ", value).strip()

        # Deduplication key only: Marker can differ solely by an
        # artificial space before punctuation in otherwise identical items.
        compare_key = re.sub(r"\s+([,.;:])", r"\1", key)

        if not compare_key or compare_key in seen:
            continue

        seen.add(compare_key)
        deduplicated.append(key)

    return deduplicated


def _find_references(obj):
    """
    Extract bibliography entries from the main Marker JSON only.

    Generic strategies:
      - locate bibliography ListGroups structurally;
      - join continuation ListGroups before numbered splitting;
      - split sequential [n] bibliography entries across arbitrary
        ListItem/ListGroup boundaries;
      - for unnumbered author-year bibliographies, use conservative
        author-year starts supported by bibliographic boundary context.
    """

    reference_heading_rx = re.compile(
        r"^\s*(references?|bibliography|literature\s+cited|works\s+cited)\s*$",
        re.I,
    )
    inline_reference_rx = re.compile(
        r"\b(?:references?|bibliography|literature\s+cited|works\s+cited)\b",
        re.I,
    )

    def clean_text(value):
        value = _html_to_text(value or "")
        value = "".join(
            ch if (ord(ch) >= 32 or ch in "\t\n\r") else " "
            for ch in value
        )
        return " ".join(value.split()).strip()

    def block_text(block):
        html = block.get("html")
        if isinstance(html, str) and html.strip():
            return clean_text(html)

        value = block.get("text")
        if isinstance(value, str) and value.strip():
            return " ".join(value.split()).strip()

        return ""

    def iter_pages(value):
        if isinstance(value, dict):
            if value.get("block_type") == "Page":
                yield value
                return
            for child in value.values():
                yield from iter_pages(child)
        elif isinstance(value, list):
            for child in value:
                yield from iter_pages(child)

    def direct_blocks(page):
        blocks = []
        for value in page.values():
            if isinstance(value, list):
                for item in value:
                    if (
                        isinstance(item, dict)
                        and isinstance(item.get("block_type"), str)
                    ):
                        blocks.append(item)
        return blocks

    def top_level_list_items(group):
        items = []
        for value in group.values():
            if isinstance(value, list):
                for item in value:
                    if (
                        isinstance(item, dict)
                        and item.get("block_type") == "ListItem"
                    ):
                        items.append(item)
        return items

    def group_stream(group):
        parts = []
        for item in top_level_list_items(group):
            value = block_text(item)
            if value:
                parts.append(value)
        return " ".join(parts).strip()

    def numbered_start(value):
        return bool(re.match(r"^\s*\[1\]\s+", value))

    def numbered_continuation(value):
        return bool(re.search(r"\[\d+\]\s+", value))

    def split_sequential_numbered(value):
        first = re.search(r"\[1\]\s*", value)
        if not first:
            return []

        starts = [(1, first.start(), first.end())]
        expected = 2
        search_pos = first.end()

        while True:
            match = re.search(
                rf"\[{expected}\]\s*",
                value[search_pos:],
            )
            if not match:
                break

            start = search_pos + match.start()
            end = search_pos + match.end()
            starts.append((expected, start, end))
            search_pos = end
            expected += 1

        entries = []
        for i, (_, start, _) in enumerate(starts):
            stop = starts[i + 1][1] if i + 1 < len(starts) else len(value)
            entry = value[start:stop].strip()
            if entry:
                entries.append(entry)

        return entries

    # Generic author-year start. This is only a candidate; boundary
    # validation below prevents arbitrary names inside titles/editors
    # from becoming reference starts.
    author_year_rx = re.compile(
        r"(?<![A-Za-z])"
        r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`.-]+"
        r"(?:"
            r",\s*(?:Jr\.,\s*)?[A-Z]"
            r"|,\s*[A-Z]\."
            r"|\s+[A-Z]\."
        r")"
        r".{0,150}?"
        r"\b(?:19|20)\d{2}[a-z]?\b"
    )

    def plausible_unnumbered_boundary(value, pos):
        if pos <= 0:
            return True

        before = value[max(0, pos - 90):pos].rstrip()

        # Strong generic bibliography endings commonly preceding the next
        # author entry: terminal punctuation, page/article numbers,
        # arXiv identifiers, or a completed numeric range.
        return bool(
            re.search(
                r"(?:"
                    r"[.!?]"
                    r"|(?:19|20)\d{2}[a-z]?[.,]?"
                    r"|\d+[A-Za-z]?[.,]?"
                    r"|\d+\s*[–-]\s*\d+[.,]?"
                    r"|arXiv[:\s]*\d{4}\.\d+(?:\s*\[[^\]]+\])?[.,]?"
                    r"|abs/\d{4}\.\d+[.,]?"
                r")\s*$",
                before,
                re.I,
            )
        )

    def split_unnumbered(value):
        candidates = list(author_year_rx.finditer(value))
        if not candidates:
            return [value] if value else []

        starts = []

        # The first bibliography entry begins at stream position zero.
        starts.append(0)

        for match in candidates:
            pos = match.start()

            if pos == 0:
                continue

            if plausible_unnumbered_boundary(value, pos):
                starts.append(pos)

        starts = sorted(set(starts))

        if len(starts) <= 1:
            return [value]

        entries = []
        for i, start in enumerate(starts):
            stop = starts[i + 1] if i + 1 < len(starts) else len(value)
            entry = value[start:stop].strip()
            if entry:
                entries.append(entry)

        return entries

    pages = list(iter_pages(obj))

    # Preserve document order so a numbered bibliography that Marker
    # continues in a following Text block can be joined safely.
    ordered_blocks = []
    for page in pages:
        ordered_blocks.extend(direct_blocks(page))

    block_positions = {
        id(block): index
        for index, block in enumerate(ordered_blocks)
    }

    selected = []
    seen = set()
    in_reference_section = False

    def add_group(group, stream_override=None):
        key = str(group.get("id", "")) or id(group)
        if key in seen:
            return
        seen.add(key)
        selected.append((group, stream_override))

    # Explicit bibliography section.
    for page in pages:
        for block in direct_blocks(page):
            block_type = block.get("block_type")

            if block_type == "SectionHeader":
                heading = block_text(block)

                if reference_heading_rx.match(heading):
                    in_reference_section = True
                    continue

                if in_reference_section and heading:
                    in_reference_section = False

            if block_type == "ListGroup" and in_reference_section:
                add_group(block)

    # Structural fallbacks when the heading was lost/fused by Marker.
    for page in pages:
        for block in direct_blocks(page):
            if block.get("block_type") != "ListGroup":
                continue

            stream = group_stream(block)
            if not stream:
                continue

            marker = inline_reference_rx.search(stream)

            # Inline heading is accepted only when followed by bibliography
            # content; the text before the heading is discarded.
            if marker:
                after = stream[marker.end():].strip(" :.-")
                if after:
                    add_group(block, after)
                continue

            if (
                re.match(r"^\s*\[1\]\s+", stream)
                and re.search(r"\[2\]\s+", stream)
                and re.search(r"\[3\]\s+", stream)
            ):
                add_group(block)

    def last_sequential_number(value):
        first = re.search(r"\[1\]\s*", value)
        if not first:
            return 0

        expected = 2
        search_pos = first.end()
        last = 1

        while True:
            match = re.search(
                rf"\[{expected}\]\s*",
                value[search_pos:],
            )
            if not match:
                break

            search_pos += match.end()
            last = expected
            expected += 1

        return last

    def append_numbered_text_continuation(group, stream):
        """
        Marker can move the continuation of a numbered bibliography from
        ListGroup into a following Text block. Accept such a continuation
        only when the Text begins with the exact next sequential [n+1].
        """
        last_number = last_sequential_number(stream)
        if not last_number:
            return stream

        position = block_positions.get(id(group))
        if position is None:
            return stream

        combined = stream
        expected = last_number + 1

        for block in ordered_blocks[position + 1:]:
            value = block_text(block)

            # Empty page furniture does not interrupt continuation.
            if not value:
                continue

            if block.get("block_type") != "Text":
                break

            if not re.match(
                rf"^\s*\[{expected}\]\s+",
                value,
            ):
                break

            combined += " " + value

            new_last = last_sequential_number(combined)
            if new_last < expected:
                break

            expected = new_last + 1

        return combined

    streams = []
    for group, override in selected:
        stream = override if override is not None else group_stream(group)
        stream = " ".join(stream.split()).strip()

        if not stream:
            continue

        marker = inline_reference_rx.match(stream)
        if marker:
            stream = stream[marker.end():].strip(" :.-")

        if not stream:
            continue

        if numbered_start(stream):
            stream = append_numbered_text_continuation(group, stream)

        streams.append(stream)

    results = []
    used = set()

    # V2: numbered bibliography continuation is joined across selected
    # ListGroups before splitting. This removes arbitrary page/ListGroup
    # boundaries from the bibliography representation.
    i = 0
    while i < len(streams):
        stream = streams[i]

        if numbered_start(stream):
            combined = stream
            j = i + 1

            while j < len(streams):
                nxt = streams[j]

                # A continuation can begin with prose from the preceding
                # reference and contain later [n] markers.
                if numbered_continuation(nxt):
                    combined += " " + nxt
                    j += 1
                    continue

                break

            entries = split_sequential_numbered(combined)
            if entries:
                results.extend(entries)
                used.update(range(i, j))
                i = j
                continue

        i += 1

    # Remaining streams are handled as unnumbered bibliographies.
    for i, stream in enumerate(streams):
        if i in used:
            continue

        if numbered_start(stream):
            entries = split_sequential_numbered(stream)
            if entries:
                results.extend(entries)
                continue

        results.extend(split_unnumbered(stream))

    return results


def extract_raw(base_dir: str, label: str, pdf):
    """
    Marker 2.0.0 extractor for eSciBench.

    Raw source:
        main Marker JSON only:
        marker_outputs/<article>/<article>.json
    """

    if label not in {"title", "author", "affiliation", "email", "abstract", "section", "keyword", "pub_date", "table", "equation", "caption", "reference", "list"}:
        return True, []

    stem = Path(pdf.pdf_name).stem
    json_path = Path(base_dir) / "marker_outputs" / stem / f"{stem}.json"

    if not json_path.exists():
        return False, []

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, []

    if label == "title":
        title = _find_first_nonempty_section_header(data)

        if not title:
            return True, []

        return True, [(pdf.pdf_name, 0, label, title)]

    if label == "author":
        authors = _find_authors_after_title(data)

        if not authors:
            return True, []

        return True, [
            (pdf.pdf_name, 0, label, author)
            for author in authors
        ]

    if label == "affiliation":
        affiliations = _find_affiliations_after_title(data)

        if not affiliations:
            return True, []

        return True, [
            (pdf.pdf_name, 0, label, affiliation)
            for affiliation in affiliations
        ]

    if label == "section":
        sections = _find_sections(data)

        if not sections:
            return True, []

        return True, [
            (pdf.pdf_name, page, label, section)
            for page, section in sections
        ]

    if label == "abstract":
        abstracts = _find_abstracts(data)

        if not abstracts:
            return True, []

        return True, [
            (pdf.pdf_name, 0, label, abstract)
            for abstract in abstracts
        ]

    if label == "caption":
        captions = _find_captions(data)

        return True, [
            (pdf.pdf_name, page, label, caption_text)
            for page, caption_text in captions
        ]

    if label == "table":
        tables = _find_tables(data)

        return True, [
            (pdf.pdf_name, page, label, table_text)
            for page, table_text in tables
        ]

    if label == "equation":
        equations = _find_equations(data)

        return True, [
            (pdf.pdf_name, page, label, equation_text)
            for page, equation_text in equations
        ]

    if label == "list":
        list_items = _find_lists(data)

        return True, [
            (pdf.pdf_name, 0, label, list_text)
            for list_text in list_items
        ]

    if label == "reference":
        references = _find_references(data)

        return True, [
            (pdf.pdf_name, 0, label, reference_text)
            for reference_text in references
        ]

    if label == "pub_date":
        pub_dates = _find_pub_dates(data)

        if not pub_dates:
            return True, []

        return True, [
            (pdf.pdf_name, 0, label, pub_date)
            for pub_date in pub_dates
        ]

    if label == "keyword":
        keywords = _find_keywords(data)

        if not keywords:
            return True, []

        return True, [
            (pdf.pdf_name, 0, label, keyword)
            for keyword in keywords
        ]

    if label == "email":
        emails = _find_emails(data)

        if not emails:
            return True, []

        return True, [
            (pdf.pdf_name, 0, label, email)
            for email in emails
        ]

    return True, []
