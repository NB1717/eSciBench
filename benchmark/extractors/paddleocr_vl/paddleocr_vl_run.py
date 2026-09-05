from pylatexenc.latex2text import LatexNodes2Text
import json
from benchmark.normalisation import normalize_string


def extract_title(raw_json_path):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    if page.get("page_index") != 0:
        return []

    return [
        normalize_string(block.get("block_content", ""))
        for block in page.get("parsing_res_list", [])
        if block.get("block_label") == "doc_title"
        and block.get("block_content", "").strip()
    ]

import re
import unicodedata


def extract_email(raw_json_path):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    emails = []

    # Search every block on every page.
    for block in page.get("parsing_res_list", []):
        text = str(block.get("block_content", ""))

        if not text.strip():
            continue

        # Official normalization already applies Unicode NFKC.
        # This also makes matching consistent with benchmark normalization.
        text = normalize_string(text)

        # Remove invisible Unicode formatting characters that may split
        # an otherwise valid email address in OCR output.
        text = re.sub(r"[\u200b-\u200d\u2060\ufeff]", "", text)

        # {name1, name2}@domain.tld
        braced_pattern = (
            r"\{([^{}]+)\}\s*@\s*"
            r"([A-Za-z0-9.-]+\.[A-Za-z]{2,})"
        )

        for names, domain in re.findall(braced_pattern, text):
            for name in names.split(","):
                name = name.strip()
                if name:
                    emails.append(f"{name}@{domain}")

        # Remove braced-address form before ordinary extraction so that
        # the same address is not recovered twice.
        ordinary_text = re.sub(braced_pattern, "", text)

        # Ordinary email addresses.
        emails.extend(
            re.findall(
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                ordinary_text,
            )
        )

    return list(dict.fromkeys(emails))


def extract_affiliation(raw_json_path):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    if page.get("page_index") != 0:
        return []

    cue = re.compile(
        r"\b(university|college|school|department|institute|institution|"
        r"laboratory|laboratories|lab|centre|center|faculty|hospital|"
        r"academy|research|physics|sciences?|group|cnrs|inria|cern|commissariat)\b",
        re.I,
    )

    marker = re.compile(
        r"^\s*\$\s*\^\{([^}]+)\}\s*\$\s*"
    )
    any_marker = re.compile(
        r"\$\s*\^\{([^}]+)\}\s*\$"
    )

    email = re.compile(r"\S*@\S+")

    stop = re.compile(
        r"^\s*(preprint|submitted|received|accepted|published|"
        r"corresponding author|e-?mail|email)\b",
        re.I,
    )

    member = re.compile(
        r"\b(?:senior\s+member|member|fellow)\s*,?\s*IEEE\b",
        re.I,
    )

    author_names = extract_author(raw_json_path)
    blocks = page.get("parsing_res_list", [])

    def marker_tokens(content):
        """
        Convert marker contents such as:
        1
        a
        1,2
        a,b
        2,*
        into generic marker tokens.
        """
        tokens = re.split(r"[\s,;/]+", str(content).strip())
        return {
            token.strip().lower()
            for token in tokens
            if token.strip()
        }

    # ------------------------------------------------------------
    # Build author -> superscript-marker relationships.
    #
    # This is based only on document structure:
    # each extracted author is located in the raw author block and
    # superscript markers occurring after that author and before the
    # next author are assigned to that author.
    # ------------------------------------------------------------
    marker_to_authors = {}

    for b in blocks:
        label = b.get("block_label")

        if label in {"abstract", "paragraph_title"}:
            break

        if label != "text":
            continue

        text = str(b.get("block_content", "")).strip()
        if not text:
            continue

        author_spans = []

        for author in author_names:
            m = re.search(re.escape(author), text, flags=re.I)
            if m:
                author_spans.append((m.start(), m.end(), author))

        if not author_spans:
            continue

        author_spans.sort(key=lambda x: x[0])

        for i, (_, author_end, author) in enumerate(author_spans):
            next_start = (
                author_spans[i + 1][0]
                if i + 1 < len(author_spans)
                else len(text)
            )

            segment = text[author_end:next_start]

            for mm in any_marker.finditer(segment):
                for token in marker_tokens(mm.group(1)):
                    marker_to_authors.setdefault(token, set()).add(author)

    out = []
    main_affiliations = []
    memberships = []

    current = None
    current_multiplier = 1

    # Multiplier associated with the most recent explicit affiliation
    # marker. It can continue across an immediately following block.
    active_marker_multiplier = 1

    saw_affiliation_marker = False

    def flush():
        nonlocal current

        if current:
            value = current.strip(" ,;.")

            if value:
                main_affiliations.append(value)

                # Preserve benchmark multiplicity:
                # one affiliation may belong to several authors.
                out.extend([value] * max(1, current_multiplier))

        current = None

    for b in blocks:
        label = b.get("block_label")

        if label in {"abstract", "paragraph_title"}:
            break

        if label != "text":
            continue

        text = str(b.get("block_content", "")).strip()
        if not text:
            continue

        # Professional memberships may themselves be affiliation GT.
        for m in member.finditer(text):
            value = normalize_string(m.group(0))
            if value:
                memberships.append(value)

        ends_with_semicolon = text.rstrip().endswith(";")

        if stop.search(text):
            flush()
            active_marker_multiplier = 1
            continue

        text = re.sub(
            r"(?:E-?mail\s*:?\s*)?"
            r"\{?[\w.+-]+(?:\s*,\s*[\w.+-]+)*\}?"
            r"@[\w.-]+\.[A-Za-z]{2,}",
            "",
            text,
            flags=re.I,
        ).strip(" ,;.")

        if not text:
            continue

        marker_match = marker.match(text)
        has_marker = marker_match is not None

        if has_marker:
            saw_affiliation_marker = True

            # Precision-aware marker mapping:
            # only one simple alphanumeric affiliation marker is allowed
            # to create multiplicity. Symbolic markers such as "*" are
            # commonly author-status/correspondence markers and are not
            # reliable affiliation identities.
            affiliation_tokens = [
                token
                for token in marker_tokens(marker_match.group(1))
                if re.fullmatch(r"(?:[a-z]{1,3}|\d{1,3})", token, re.I)
            ]

            linked_authors = set()

            if len(affiliation_tokens) == 1:
                linked_authors.update(
                    marker_to_authors.get(affiliation_tokens[0], set())
                )

            # Duplicate only for an unambiguous one-marker mapping that
            # explicitly links more than one extracted author.
            active_marker_multiplier = (
                len(linked_authors)
                if len(linked_authors) > 1
                else 1
            )

        text = marker.sub("", text).strip()

        parts = [
            x.strip(" ,;.")
            for x in text.split(";")
            if x.strip(" ,;.")
        ]

        if cue.search(text) or has_marker:
            if has_marker:
                flush()

            for i, part in enumerate(parts):
                if i > 0:
                    flush()

                if current is None:
                    current = part
                    current_multiplier = active_marker_multiplier
                else:
                    current = (current + " " + part).strip()

            if ends_with_semicolon:
                flush()

        elif current:
            looks_like_continuation = (
                "," in text
                or bool(re.search(r"\b\d{3,}\b", text))
                or bool(re.search(
                    r"\b(road|street|avenue|boulevard|building|campus|"
                    r"city|state|province|country|kingdom|usa|uk|china|"
                    r"france|germany|canada|japan|switzerland)\b",
                    text,
                    re.I,
                ))
            )

            if looks_like_continuation and not has_marker:
                current = (current + " " + text).strip()
            else:
                flush()
                active_marker_multiplier = 1

    flush()

    # Professional membership/status affiliations.
    out.extend(memberships)

    # Existing conservative shared-affiliation rule:
    # several authors + one unmarked affiliation => shared affiliation.
    if (
        len(author_names) > 1
        and len(main_affiliations) == 1
        and not saw_affiliation_marker
    ):
        shared = main_affiliations[0]

        # One copy is already present.
        out.extend([shared] * (len(author_names) - 1))

    # ------------------------------------------------------------
    # Footnote affiliations
    # ------------------------------------------------------------
    for b in blocks:
        if b.get("block_label") != "footnote":
            continue

        text = str(b.get("block_content", "")).strip()

        if not text or not cue.search(text):
            continue

        shared_count = 1

        m_are_with = re.search(
            r"\bare with\b",
            text,
            flags=re.I,
        )

        if m_are_with:
            prefix = normalize_string(
                text[:m_are_with.start()]
            )

            matched_authors = 0

            for author in author_names:
                a = normalize_string(author)

                if a and a in prefix:
                    matched_authors += 1

            if matched_authors > 1:
                shared_count = matched_authors

            text = text[m_are_with.end():].strip()

        text = re.sub(
            r";\s*(?:e-?mail|email)\s*:[^)]*(\))",
            r"\1",
            text,
            flags=re.I,
        )

        text = re.sub(
            r"[;(]\s*(?:e-?mail|email)\s*:.*$",
            "",
            text,
            flags=re.I,
        ).strip(" ,;.")

        if text:
            out.extend([text] * shared_count)

    return [x for x in out if x]





def extract_author(raw_json_path):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    if page.get("page_index") != 0:
        return []

    affiliation = re.compile(
        r"\b(university|college|school|department|institute|institution|"
        r"laboratory|laboratories|lab|centre|center|faculty|hospital|academy|"
        r"research|physics|sciences?|group|cnrs|inria|cern|commissariat)\b",
        re.I,
    )
    metadata = re.compile(
        r"^\s*(preprint|received|accepted|published|publication|submitted|"
        r"revised|online\s+version)\b",
        re.I,
    )
    address = re.compile(
        r"\b\d{3,}\b.*[,;]|\b(road|street|avenue|boulevard|france|"
        r"united kingdom|usa|uk|china|switzerland)\b",
        re.I,
    )
    marker = re.compile(r"\s*\$\s*\^\{[^}]+\}\s*\$\s*")
    member = re.compile(
        r",?\s*(?:senior\s+member|member|fellow)\s*,?\s*IEEE\b.*$",
        re.I,
    )

    out = []
    seen_title = False

    for b in page.get("parsing_res_list", []):
        label = b.get("block_label")

        if label == "doc_title":
            seen_title = True
            continue

        if not seen_title:
            continue

        if label in {"abstract", "paragraph_title"}:
            break
        if label != "text":
            continue

        text = b.get("block_content", "").strip()
        if not text:
            continue

        if "@" in text or metadata.search(text):
            continue
        if affiliation.search(text) or address.search(text):
            continue

        text = member.sub("", text)
        text = marker.sub(" | ", text)
        text = re.sub(r"\s+\band\b\s+", " | ", text, flags=re.I)

        block_names = []
        for name in text.split("|"):
            name = re.sub(r"\s+", " ", name).strip(" ,;.")
            if name:
                block_names.append(name)

        # Conservative OCR line-break merge:
        # merge only an unfinished hyphen-ending author fragment with
        # the first author candidate of the immediately following block.
        if (
            out
            and block_names
            and out[-1].endswith("-")
        ):
            out[-1] = out[-1][:-1].rstrip() + block_names[0].lstrip()
            block_names = block_names[1:]

        out.extend(block_names)

    # Deduplicate authors using the benchmark's official normalization,
    # while preserving the first raw prediction string.
    from benchmark.normalisation import normalize_string

    deduped = []
    seen = set()
    for name in out:
        key = normalize_string(name)
        if key and key not in seen:
            seen.add(key)
            deduped.append(normalize_string(name))

    return deduped

def extract_keyword(raw_json_path):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    if page.get("page_index") != 0:
        return []

    from benchmark.normalisation import normalize_string

    # Generic multilingual keyword headings.
    heading = re.compile(
        r"^\s*(?:"
        r"key\s*words?"
        r"|keywords?"
        r"|index\s+terms?"
        r"|mots?\s*[-‐-‒–—]?\s*cl[ée]s?"
        r"|mots?\s*[-‐-‒–—]?\s*clefs?"
        r"|palabras?\s+claves?"
        r")\s*(?:[:.\-–—]\s*)?",
        re.I,
    )

    # Generic publication/article metadata which must not become keywords.
    metadata = re.compile(
        r"(?:"
        r"\bdoi\s*:?"
        r"|\barxiv\s*:?"
        r"|\bissn\b"
        r"|\bisbn\b"
        r"|\bvolume\b"
        r"|\bvol\.\s*\d"
        r"|\bissue\b"
        r"|\breceived\b"
        r"|\baccepted\b"
        r"|\bpublished\b"
        r"|\bpublication\b"
        r"|\bpreprint\b"
        r"|\bcopyright\b"
        r"|\b©\b"
        r"|https?://"
        r"|www\."
        r")",
        re.I,
    )

    out = []
    blocks = page.get("parsing_res_list", [])

    def clean_item(item):
        item = re.sub(r"\s+", " ", item).strip(" ,;:.|•·")
        if not item:
            return ""

        # Canonicalize Unicode dash variants inside a keyword.
        # ASCII hyphen is intentionally preserved.
        item = re.sub(r"\s*[–—−]\s*", "−", item)
        return normalize_string(item)

    def split_keyword_line(line):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            return []

        # Strong, language-independent list separators.
        if re.search(r"[;,•·|]", line):
            parts = re.split(r"\s*[;,•·|]\s*", line)

        else:
            # Dash is ambiguous: it may belong inside one keyword.
            # Treat it as a list separator only when the line has
            # several dash boundaries, i.e. clearly list-like.
            dash_boundaries = re.findall(r"\s+[—–]\s+", line)
            if len(dash_boundaries) >= 2:
                parts = re.split(r"\s+[—–]\s+", line)
            else:
                parts = [line]

        result = []
        for part in parts:
            item = clean_item(part)
            if item:
                result.append(item)
        return result

    for i, block in enumerate(blocks):
        text = str(block.get("block_content", "") or "").strip()
        if not text:
            continue

        m = heading.match(text)
        if not m:
            continue

        remainder = text[m.end():].strip()

        # ------------------------------------------------------------
        # Form 1: heading and keyword list in the same block.
        # ------------------------------------------------------------
        if remainder:
            for line in remainder.splitlines():
                line = line.strip()
                if not line:
                    continue
                if metadata.search(line):
                    break
                out.extend(split_keyword_line(line))
            continue

        # ------------------------------------------------------------
        # Form 2: standalone heading followed by keyword text block(s).
        # ------------------------------------------------------------
        for next_block in blocks[i + 1:]:
            label = next_block.get("block_label")
            next_text = str(next_block.get("block_content", "") or "").strip()

            if not next_text:
                continue

            # Structural end of the keyword zone.
            if label in {
                "paragraph_title",
                "doc_title",
                "abstract",
                "reference_content",
                "figure_title",
                "table",
            }:
                break

            if label != "text":
                continue

            consumed = False

            for line in next_text.splitlines():
                line = line.strip()
                if not line:
                    continue

                if metadata.search(line):
                    return list(dict.fromkeys(out))

                items = split_keyword_line(line)
                if items:
                    out.extend(items)
                    consumed = True

            # A standalone keyword heading normally owns the immediately
            # following textual keyword region. Do not drift into body text.
            if consumed:
                break

    # Normalize first, then deduplicate while preserving document order.
    deduped = []
    seen = set()

    for item in out:
        item = clean_item(item)
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped



def normalize_pub_date_text(text):
    """
    Generic normalization used only for publication-date recognition.

    It does not decide whether a date is a publication date.
    It only makes equivalent written date forms easier to recognize.
    """
    text = str(text or "")

    # Unicode compatibility normalization:
    # full-width digits/punctuation -> ordinary forms, etc.
    text = unicodedata.normalize("NFKC", text)

    # Unicode / non-breaking spaces.
    for ch in (
        "\u00a0", "\u2007", "\u2009", "\u200a",
        "\u202f", "\u205f", "\u3000"
    ):
        text = text.replace(ch, " ")

    # Unicode dash variants.
    for ch in (
        "\u2010", "\u2011", "\u2012", "\u2013",
        "\u2014", "\u2212", "\ufe58", "\ufe63", "\uff0d"
    ):
        text = text.replace(ch, "-")

    # Unicode punctuation variants commonly produced by OCR.
    replacements = {
        "／": "/",
        "．": ".",
        "，": ",",
        "：": ":",
        "﹕": ":",
        "；": ";",
        "（": "(",
        "）": ")",
        "［": "[",
        "］": "]",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove English ordinal suffixes:
    # 1st, 2nd, 3rd, 4th ... -> 1, 2, 3, 4 ...
    text = re.sub(
        r"(?i)\b(\d{1,2})(?:st|nd|rd|th)\b",
        r"\1",
        text,
    )

    # Normalize common abbreviated month punctuation:
    # Nov. -> Nov, Sept. -> Sept, etc.
    text = re.sub(
        r"(?i)\b("
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
        r")\.(?=\s|,|-|/|$)",
        r"\1",
        text,
    )

    # Normalize whitespace around date separators.
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"\s*\.\s*(?=\d)", ".", text)

    # Normalize comma spacing.
    text = re.sub(r"\s*,\s*", ", ", text)

    # Generic year-month-name-day form:
    # 2016 November 23 -> 23 November 2016
    month_name = (
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?|tember)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    )

    text = re.sub(
        rf"\b((?:19|20)\d{{2}})\s+({month_name})\s+(\d{{1,2}})\b",
        lambda m: f"{m.group(3)} {m.group(2)} {m.group(1)}",
        text,
        flags=re.I,
    )

    # Collapse repeated whitespace last.
    text = re.sub(r"\s+", " ", text).strip()

    return text

def extract_pub_date(raw_json_path):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    if page.get("page_index") != 0:
        return []

    blocks = page.get("parsing_res_list", [])

    MONTHS = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }

    MONTH_NAMES = {
        1: "january", 2: "february", 3: "march", 4: "april",
        5: "may", 6: "june", 7: "july", 8: "august",
        9: "september", 10: "october", 11: "november", 12: "december",
    }

    month_rx = (
        r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?|tember)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    )

    def norm(text):
        text = str(text or "")

        for ch in ("\u00a0", "\u2007", "\u2009", "\u202f"):
            text = text.replace(ch, " ")

        for ch in ("–", "—", "−", "‐", "‒"):
            text = text.replace(ch, "-")

        text = text.replace("，", ",")
        text = text.replace("：", ":")
        text = text.replace("／", "/")
        text = text.replace("．", ".")

        text = re.sub(
            r"(?i)\b(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.",
            r"\1",
            text,
        )

        text = re.sub(
            r"(?i)\b(\d{1,2})(?:st|nd|rd|th)\b",
            r"\1",
            text,
        )

        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def emit(year, month, day):
        try:
            year = int(year)
            month = int(month)
            day = int(day)
        except Exception:
            return None

        if not 1900 <= year <= 2099 or not 1 <= month <= 12:
            return None

        days = {
            1: 31, 2: 29, 3: 31, 4: 30,
            5: 31, 6: 30, 7: 31, 8: 31,
            9: 30, 10: 31, 11: 30, 12: 31,
        }

        if not 1 <= day <= days[month]:
            return None

        return normalize_string(
            f"{MONTH_NAMES[month]} {day}, {year}"
        )

    def extract_dates(text):
        text = normalize_pub_date_text(norm(text))
        found = []

        # --------------------------------------------------------
        # Generic scholarly date formats.
        #
        # Supported examples:
        #   2016-11-23
        #   2016/11/23
        #   2016.11.23
        #
        #   23 November 2016
        #   23 Nov 2016
        #   23 Nov. 2016
        #   23-Nov-2016
        #   23/Nov/2016
        #
        #   November 23, 2016
        #   Nov 23 2016
        #   Nov. 23, 2016
        #   Nov-23-2016
        #   Nov/23/2016
        # --------------------------------------------------------

        # day month year
        for m in re.finditer(
            rf"\b(\d{{1,2}})\s*[-/. ]\s*"
            rf"({month_rx})\.?"
            rf"\s*[-/,. ]+\s*((?:19|20)\d{{2}})\b",
            text,
            re.I,
        ):
            day, month, year = m.groups()

            key = month.lower().rstrip(".")
            month_num = MONTHS.get(key)

            if month_num:
                d = emit(year, month_num, day)
                if d:
                    found.append(d)

        # month day year
        for m in re.finditer(
            rf"\b({month_rx})\.?"
            rf"\s*[-/. ]\s*(\d{{1,2}})"
            rf"\s*(?:,|[-/. ])+\s*((?:19|20)\d{{2}})\b",
            text,
            re.I,
        ):
            month, day, year = m.groups()

            key = month.lower().rstrip(".")
            month_num = MONTHS.get(key)

            if month_num:
                d = emit(year, month_num, day)
                if d:
                    found.append(d)

        # ISO / year-month-day
        for m in re.finditer(
            r"\b((?:19|20)\d{2})\s*[-/.]\s*"
            r"(\d{1,2})\s*[-/.]\s*(\d{1,2})\b",
            text,
        ):
            year, month, day = m.groups()

            d = emit(year, month, day)
            if d:
                found.append(d)

        # Purely numeric day/month/year or month/day/year.
        # Accept only when the ordering is inherently unambiguous.
        for m in re.finditer(
            r"\b(\d{1,2})\s*[-/.]\s*"
            r"(\d{1,2})\s*[-/.]\s*((?:19|20)\d{2})\b",
            text,
        ):
            a, b, year = map(int, m.groups())

            if a > 12 and 1 <= b <= 12:
                # DD/MM/YYYY
                d = emit(year, b, a)

            elif b > 12 and 1 <= a <= 12:
                # MM/DD/YYYY
                d = emit(year, a, b)

            else:
                # Ambiguous numeric date such as 03/04/2016:
                # do not guess.
                d = None

            if d:
                found.append(d)

        # Deduplicate after canonicalization.
        out = []
        seen = set()

        for d in found:
            d = normalize_string(d)

            if d not in seen:
                seen.add(d)
                out.append(d)

        return out

    # Front matter only.
    front = []

    for i, b in enumerate(blocks):
        label = b.get("block_label")
        text = norm(b.get("block_content", ""))

        if label == "abstract":
            break

        if label == "paragraph_title" and re.search(
            r"\b(?:introduction|background|methods?|materials?|results?)\b",
            text,
            re.I,
        ):
            break

        front.append((i, b, text))

    if not front:
        return []

    strong_pub = re.compile(
        r"\b(?:"
        r"published\s+online|"
        r"first\s+published|"
        r"publication\s+date|"
        r"date\s+of\s+publication|"
        r"date\s+published|"
        r"available\s+online|"
        r"first\s+online|"
        r"online\s+first|"
        r"electronic(?:ally)?\s+published|"
        r"version\s+of\s+record|"
        r"published\s+ahead\s+of\s+print|"
        r"ahead\s+of\s+print|"
        r"epublished|"
        r"epub"
        r")\b",
        re.I,
    )

    medium_pub = re.compile(
        r"\b(?:"
        r"published|"
        r"publication|"
        r"online\s+publication|"
        r"online\s+version|"
        r"preprint\s+online\s+version|"
        r"posted\s+online|"
        r"released\s+online|"
        r"available\s+from|"
        r"dated(?:\s*:)?"
        r")\b",
        re.I,
    )

    editorial = re.compile(
        r"\b(?:"
        r"received|submitted|accepted|revised|revision|"
        r"resubmitted"
        r")\b",
        re.I,
    )

    version_workflow = re.compile(
        r"\b(?:"
        r"preprint|"
        r"arxiv"
        r")\b",
        re.I,
    )

    repository_meta = re.compile(
        r"\b(?:"
        r"arxiv\s*:|"
        r"biorxiv\s*:|"
        r"medrxiv\s*:|"
        r"preprint\b"
        r")",
        re.I,
    )

    # Independent evidence that this document carries a publication /
    # public-availability date signal.
    has_pub_signal = any(
        (strong_pub.search(text) or medium_pub.search(text))
        and not editorial.search(text)
        for _, _, text in front
    )

    # ------------------------------------------------------------
    # Priority 0:
    # Segmented scholarly publication metadata.
    #
    # A publication-history block may contain several independent
    # events, for example:
    #
    # Received DATE; Revised DATE; Accepted DATE; Published DATE
    #
    # Split those events first so editorial workflow dates do not
    # cause the entire block to be rejected.
    #
    # Editorial/repository cues are segmentation boundaries only;
    # they are never returned as publication dates here.
    # ------------------------------------------------------------
    publication_piece = re.compile(
        r"\b(?:"
        r"published(?:\s+online|\s+on)?|"
        r"first\s+published|"
        r"publication(?:\s+date)?|"
        r"date\s+of\s+publication|"
        r"date\s+published|"
        r"available\s+online|"
        r"online\s+publication|"
        r"online\s+first|"
        r"first\s+online|"
        r"electronic(?:ally)?\s+published|"
        r"published\s+ahead\s+of\s+print|"
        r"version\s+of\s+record|"
        r"posted\s+online|"
        r"released\s+online|"
        r"dated"
        r")\b",
        re.I,
    )

    workflow_piece = re.compile(
        r"\b(?:"
        r"received|accepted|revised|revision|submitted|resubmitted|"
        r"preprint|arxiv|this\s+version"
        r")\b",
        re.I,
    )

    event_split_rx = re.compile(
        r"""(?ix)
        (?=
            published\s+online |
            first\s+published |
            date\s+of\s+publication |
            date\s+published |
            publication\s+date |
            available\s+online |
            online\s+publication |
            online\s+first |
            first\s+online |
            electronically\s+published |
            electronic\s+published |
            published\s+ahead\s+of\s+print |
            version\s+of\s+record |
            posted\s+online |
            released\s+online |
            published |
            publication |
            dated |
            received |
            accepted |
            revised |
            revision |
            submitted |
            resubmitted |
            preprint |
            this\s+version |
            arxiv\s*:
        )
        """
    )

    for _, _, text in front:
        cleaned = text.replace("\r", "\n")
        cleaned = cleaned.replace("•", " ")
        cleaned = cleaned.replace("|", " ")
        cleaned = cleaned.replace("·", " ")

        pieces = []

        for line in re.split(r"[;\n]", cleaned):
            line = line.strip()

            if not line:
                continue

            line = line.strip(".,;:()[]{} ")

            for piece in event_split_rx.split(line):
                piece = piece.strip(" .,;:()[]{}")

                if piece:
                    pieces.append(piece)

        for piece in pieces:
            # Only explicit publication/public-availability events
            # are candidates.
            if not publication_piece.search(piece):
                continue

            # A workflow/repository-only fragment cannot become pub_date.
            if workflow_piece.search(piece) and not publication_piece.search(piece):
                continue

            dates = extract_dates(piece)

            if dates:
                return [dates[0]]

    # ------------------------------------------------------------
    # Priority 1:
    # Date directly associated with an explicit publication /
    # public-availability statement.
    #
    # These semantic contexts outrank editorial workflow and
    # repository/version dates.
    # ------------------------------------------------------------
    for _, _, text in front:
        if editorial.search(text):
            continue

        if strong_pub.search(text):
            dates = extract_dates(text)
            if dates:
                return [dates[0]]

    for _, _, text in front:
        if editorial.search(text):
            continue

        if medium_pub.search(text) and not version_workflow.search(text):
            dates = extract_dates(text)
            if dates:
                return [dates[0]]

    # ------------------------------------------------------------
    # Priority 2:
    # Repository/version dates (preprint/arXiv) are lower priority.
    # They are considered only when an independent publication /
    # availability signal exists and no higher-priority publication
    # date was found above.
    # ------------------------------------------------------------
    if has_pub_signal:
        for _, b, text in front:
            if b.get("block_label") not in {"aside_text", "header", "text"}:
                continue

            if not repository_meta.search(text):
                continue

            if editorial.search(text):
                continue

            dates = extract_dates(text)
            if dates:
                return [dates[0]]

    # ------------------------------------------------------------
    # Priority 3:
    # Short metadata/date block immediately adjacent to a publication cue.
    # ------------------------------------------------------------
    for pos, (_, b, text) in enumerate(front):
        if editorial.search(text):
            continue

        dates = extract_dates(text)

        if not dates:
            continue

        if len(text) > 120:
            continue

        neighbors = []

        if pos > 0:
            neighbors.append(front[pos - 1][2])

        if pos + 1 < len(front):
            neighbors.append(front[pos + 1][2])

        neighbor_text = " ".join(neighbors)

        if editorial.search(neighbor_text):
            continue

        if strong_pub.search(neighbor_text) or medium_pub.search(neighbor_text):
            return [dates[0]]

    return []

def extract_abstract(raw_json_path):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    if page.get("page_index") != 0:
        return []

    out = []

    for b in page.get("parsing_res_list", []):
        if b.get("block_label") != "abstract":
            continue

        text = b.get("block_content", "").strip()
        text = re.sub(r"^\s*abstract\s*[\-—–:.\s]+\s*", "", text, flags=re.I)

        # Apply the official eSciBench normalization to the prediction.
        text = normalize_string(text)

        if text:
            out.append(text)

    return out

def extract_caption(raw_json_path):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    out = []

    def canonicalize_caption_math(caption):
        caption = normalize_string(caption)

        # Remove inline LaTeX math delimiters while preserving content.
        caption = re.sub(r"\$\s*(.*?)\s*\$", r"\1", caption)

        # Canonicalize common mathematical symbols.
        caption = caption.replace(r"\alpha", "α")
        caption = caption.replace(r"\beta", "β")
        caption = caption.replace(r"\chi", "χ")
        caption = caption.replace(r"\delta", "δ")
        caption = caption.replace(r"\Delta", "δ")
        caption = caption.replace(r"\zeta", "ζ")
        caption = caption.replace(r"\lambda", "λ")
        caption = caption.replace(r"\omega", "ω")
        caption = caption.replace(r"\varphi", "φ")
        caption = caption.replace(r"\mu", "μ")
        caption = caption.replace(r"\sigma", "σ")
        caption = caption.replace(r"\times", "×")

        # Convert simple LaTeX fractions to plain-text numerator/denominator.
        caption = re.sub(
            r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
            r"\1/\2",
            caption,
        )

        # Remove purely typographical LaTeX wrappers.
        caption = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", caption)
        caption = caption.replace(r"\left", "")
        caption = caption.replace(r"\right", "")

        # Normalize common LaTeX spacing commands.
        caption = caption.replace(r"\,", "")
        caption = caption.replace(r"\!", "")
        caption = caption.replace(r"\;", " ")
        caption = caption.replace(r"\:", " ")
        caption = caption.replace("~", " ")

        # Canonicalize generic mathematical commands.
        caption = caption.replace(r"\log", "log")
        caption = caption.replace(r"\to", "→")
        caption = caption.replace(r"\in", "∈")
        caption = caption.replace(r"\geq", "≥")
        caption = caption.replace(r"\leq", "≤")
        caption = caption.replace(r"\pm", "±")

        # Flatten simple LaTeX superscript/subscript braces.
        caption = re.sub(r"\^\{([^{}]+)\}", r"^\1", caption)
        caption = re.sub(r"_\{([^{}]+)\}", r"_\1", caption)

        # Normalize spacing around common operators.
        caption = re.sub(r"\s*=\s*", "=", caption)
        caption = re.sub(r"\s+", " ", caption).strip()

        return caption

    for b in page.get("parsing_res_list", []):
        label = b.get("block_label", "")
        text = " ".join(str(b.get("block_content", "")).split()).strip()

        if not text:
            continue

        if label == "figure_title":
            m = re.match(
                r"^\s*(?:figure|fig\.?|table)\s*\d+[A-Za-z]?\s*[:.\-–—]?\s*(.+)$",
                text,
                re.I,
            )
            if not m:
                continue

            caption = m.group(1).strip()

            # A real caption must contain descriptive text, not only a marker.
            if not re.search(r"[A-Za-z]{2,}", caption):
                continue

            caption = canonicalize_caption_math(caption)

            if caption:
                out.append(caption)

        elif label == "algorithm":
            m = re.match(
                r"^\s*algorithm\s*\d+[A-Za-z]?\s*[:.\-–—]?\s*(.+)$",
                text,
                re.I,
            )
            if not m:
                continue

            caption = m.group(1).strip()

            # Paddle may merge the caption and algorithm steps into one block.
            caption = re.split(r"\s+\d+\s*:\s+", caption, maxsplit=1)[0].strip()

            if caption:
                caption = canonicalize_caption_math(caption)

                if caption:
                    out.append(caption)

    return out


def extract_equation(raw_json_path):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    out = []

    for b in page.get("parsing_res_list", []):
        if b.get("block_label") != "display_formula":
            continue

        text = " ".join(str(b.get("block_content", "")).split()).strip()
        if not text:
            continue

        # Paddle commonly wraps display formulas in $$ ... $$.
        if text.startswith("$$") and text.endswith("$$"):
            text = text[2:-2].strip()

        # Paddle OCR may letter-space textual math operators, e.g.
        # \mathrm{t a n h}, \mathrm{P r}, \operatorname{M L P}.
        # Collapse spacing only inside explicitly textual LaTeX commands.
        def _collapse_textual_math_command(m):
            command = m.group(1)
            body = m.group(2)
            return "\\" + command + "{" + re.sub(r"\s+", "", body) + "}"

        text = re.sub(
            r"\\(mathrm|operatorname)\{([A-Za-z](?:\s+[A-Za-z])+)\}",
            _collapse_textual_math_command,
            text,
        )

        # Convert LaTeX representation to plain text before official alignment.
        text = LatexNodes2Text().latex_to_text(text)

        # Apply the official eSciBench normalization on the prediction side.
        text = normalize_string(text)

        # Preserve sub/superscript structure produced from the Paddle LaTeX.
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            out.append(text)

    return out


def extract_section(raw_json_path):
    import re

    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    out = []

    for block in page.get("parsing_res_list", []):
        if block.get("block_label") != "paragraph_title":
            continue

        text = str(block.get("block_content", "")).strip()
        if not text:
            continue

        text = normalize_string(text)

        # Remove generic section-number prefixes while preserving
        # the actual heading text.
        #
        # Examples:
        #   "3.1. model"      -> "model"
        #   "2 introduction"  -> "introduction"
        #   "iv. discussion"  -> "discussion"
        #   "a. background"   -> "background"
        text = re.sub(
            r"^\s*(?:"
            r"\d+(?:\.\d+)*\.?"
            r"|[ivxlcdm]+\."
            r"|[a-z]\."
            r")\s+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        if not text:
            continue

        # Number-parenthesis items ending in a colon are typically
        # enumerated prose/subpoints rather than document section headings.
        # Example structural form: "2) transitivity:"
        if re.match(r"^\d+\)\s+.+:\s*$", text):
            continue

        # paragraph_title can also contain structural elements that are
        # not document sections. Keep these filters generic and semantic.
        if re.match(r"^theorem\b", text, flags=re.IGNORECASE):
            continue

        if re.match(r"^returns?\s*:\s*$", text, flags=re.IGNORECASE):
            continue

        if re.match(r"^keywords?\s*:?\s*$", text, flags=re.IGNORECASE):
            continue

        if re.match(r"^contents?\s*:?\s*$", text, flags=re.IGNORECASE):
            continue

        out.append(text)

    return out


def extract_reference(raw_json_path):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    refs = []

    for block in page.get("parsing_res_list", []):
        if block.get("block_label") != "reference_content":
            continue

        text = str(block.get("block_content", "")).strip()
        if not text:
            continue

        text = normalize_string(text)

        if text:
            refs.append(text)

    # Remove only strict duplicate fragments:
    # if one normalized reference block is wholly contained inside
    # a longer reference block on the same page, keep the longer one.
    out = []

    for i, text in enumerate(refs):
        contained = False

        for j, other in enumerate(refs):
            if i == j:
                continue

            if len(text) < len(other) and text in other:
                contained = True
                break

        if not contained:
            out.append(text)

    return out


def extract_header(raw_json_path):
    import re
    from pathlib import Path

    raw_json_path = Path(raw_json_path)

    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    page_index = page.get("page_index")
    page_height = page.get("height")
    blocks = page.get("parsing_res_list", [])
    out = []

    def clean_text(value):
        value = str(value or "").strip()
        if not value:
            return ""
        return normalize_string(value)

    def is_copyright(text):
        return bool(
            re.match(r"^\s*(?:©|copyright\b)", text, flags=re.IGNORECASE)
        )

    def is_top_region(block, height):
        if not height:
            return False

        bbox = block.get(
            "block_bbox",
            block.get("bbox", block.get("coordinate"))
        )

        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return False

        try:
            return float(bbox[3]) <= 0.15 * float(height)
        except (TypeError, ValueError, ZeroDivisionError):
            return False

    # Page 0: preserve Paddle's explicit document/journal headers.
    if page_index == 0:
        for block in blocks:
            if block.get("block_label") != "header":
                continue

            text = clean_text(block.get("block_content", ""))

            if not text or is_copyright(text):
                continue

            if text not in out:
                out.append(text)

        return out

    # Establish trusted running-header strings from Paddle's explicit
    # header detections across the same scholarly document.
    occurrences = {}

    for sibling in raw_json_path.parent.glob("*_res.json"):
        try:
            with open(sibling, "r", encoding="utf-8") as f:
                sibling_page = json.load(f)
        except Exception:
            continue

        sibling_index = sibling_page.get("page_index")

        for block in sibling_page.get("parsing_res_list", []):
            if block.get("block_label") != "header":
                continue

            text = clean_text(block.get("block_content", ""))

            if not text or is_copyright(text):
                continue

            occurrences.setdefault(text, set()).add(sibling_index)

    repeated = {
        text for text, pages in occurrences.items()
        if len(pages) >= 2
    }

    # Normal Paddle header detections.
    for block in blocks:
        if block.get("block_label") != "header":
            continue

        text = clean_text(block.get("block_content", ""))

        if text and text in repeated and text not in out:
            out.append(text)

    # Conservative layout-label rescue:
    # a non-header block may be recovered only when
    #   1. it lies in the top 15% of the page, and
    #   2. its normalized text EXACTLY matches a running-header string
    #      independently confirmed as "header" on >=2 other pages.
    for block in blocks:
        if block.get("block_label") == "header":
            continue

        if not is_top_region(block, page_height):
            continue

        text = clean_text(block.get("block_content", ""))

        if text and text in repeated and text not in out:
            out.append(text)

    return out


def extract_footer(raw_json_path):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    blocks = page.get("parsing_res_list", [])
    out = []

    # Generic scholarly-footnote markers.  These markers identify
    # footnotes but are not part of the semantic footer content.
    marker_re = re.compile(
        r"""
        ^\s*
        (?:
            \$\s*\^?\s*\{\s*(?:\d{1,3}|[*†‡])\s*\}\s*\$
            |
            [*†‡]\s*
            |
            \(\s*\d{1,3}\s*\)\s+
            |
            \d{1,3}[.)]\s+
        )
        """,
        flags=re.VERBOSE,
    )

    def has_leading_marker(value):
        value = str(value or "")
        return bool(marker_re.match(value))

    def clean_footnote_text(value):
        value = str(value or "").strip()
        if not value:
            return ""

        # Remove only a marker occurring at the beginning of the
        # footnote. Numbers/symbols inside the actual text are preserved.
        value = marker_re.sub("", value, count=1).strip()

        return value

    def clean_formula(value):
        value = str(value or "").strip()
        if not value:
            return ""

        value = LatexNodes2Text().latex_to_text(value)
        value = re.sub(r"[_^]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()

        return value

    i = 0

    while i < len(blocks):
        block = blocks[i]

        if block.get("block_label") != "footnote":
            i += 1
            continue

        first = clean_footnote_text(block.get("block_content", ""))

        text_parts = []
        if first:
            text_parts.append(first)

        j = i + 1

        # Reconstruct a logical scholarly footnote across layout blocks.
        #
        # Continuations without a new marker belong to the current
        # footnote.  A later footnote carrying a fresh marker starts a
        # new independent footer and is therefore left for the next
        # outer iteration.
        while j < len(blocks):
            next_block = blocks[j]
            label = next_block.get("block_label")

            if label == "display_formula":
                value = clean_formula(next_block.get("block_content", ""))
                if value:
                    text_parts.append(value)
                j += 1
                continue

            if label == "formula_number":
                # Equation numbers are layout metadata, not semantic
                # footer text.
                j += 1
                continue

            if label == "footnote":
                raw_value = str(
                    next_block.get("block_content", "") or ""
                ).strip()

                if has_leading_marker(raw_value):
                    # New explicitly marked footnote: do not merge.
                    break

                value = clean_footnote_text(raw_value)
                if value:
                    text_parts.append(value)

                j += 1
                continue

            break

        text = " ".join(x for x in text_parts if x)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            i = j if j > i else i + 1
            continue

        # Final normalization uses the official eSciBench normalizer.
        text = normalize_string(text)

        if not text:
            i = j if j > i else i + 1
            continue

        low = text.lower()

        # Keep the existing conservative exclusion for a structure that
        # is clearly an author/contact/affiliation note rather than the
        # Footer target used by this wrapper.
        if (
            "are with" in low
            and "email:" in low
            and "phone:" in low
        ):
            i = j if j > i else i + 1
            continue

        out.append(text)

        i = j if j > i else i + 1

    return out


def extract_table(raw_json_path):
    import re
    import html

    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    out = []

    for block in page.get("parsing_res_list", []):
        if block.get("block_label") != "table":
            continue

        table_html = str(block.get("block_content", "")).strip()
        if not table_html:
            continue

        # A single PaddleOCR table block represents one table.
        # Merge all non-empty cells from that table into one text value.
        cells = re.findall(
            r"<(?:td|th)\b[^>]*>(.*?)</(?:td|th)>",
            table_html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        merged = []

        for cell in cells:
            cell = html.unescape(cell)

            # Convert explicit HTML line breaks to spaces before stripping tags.
            cell = re.sub(r"<br\s*/?>", " ", cell, flags=re.IGNORECASE)

            # Flatten any remaining nested markup inside the cell.
            cell = re.sub(r"<[^>]+>", " ", cell)

            # Normalize layout whitespace before official text normalization.
            cell = re.sub(r"\s+", " ", cell).strip()

            if not cell:
                continue

            cell = normalize_string(cell)

            # Generic table-specific math canonicalization.
            # Preserve semantic content while reducing superficial LaTeX
            # representation differences inside table cells.
            cell = re.sub(r"\$\s*(.*?)\s*\$", r"\1", cell)
            cell = cell.replace(r"\alpha", "α")
            cell = cell.replace(r"\tau", "τ")
            cell = cell.replace(r"\times", "×")

            # Flatten simple LaTeX superscript/subscript braces.
            cell = re.sub(r"\^\{([^{}]+)\}", r"^\1", cell)
            cell = re.sub(r"_\{([^{}]+)\}", r"_\1", cell)

            # Remove spacing around common mathematical operators/ranges.
            cell = re.sub(r"\s*=\s*", "=", cell)
            cell = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "-", cell)

            # Final whitespace cleanup after canonicalization.
            cell = re.sub(r"\s+", " ", cell).strip()

            if cell:
                merged.append(cell)

        # Generic fallback: if PaddleOCR labels the block as a table but
        # no td/th cells are recoverable, flatten the whole table markup.
        if not merged:
            fallback = html.unescape(table_html)
            fallback = re.sub(r"<br\s*/?>", " ", fallback, flags=re.IGNORECASE)
            fallback = re.sub(r"<[^>]+>", " ", fallback)
            fallback = re.sub(r"\s+", " ", fallback).strip()
            fallback = normalize_string(fallback)

            if fallback:
                merged.append(fallback)

        # Tables whose non-empty cells consist only of binary digits
        # are treated as non-table structural content.
        # Keep this generic: it depends only on cell content.
        nonempty = [x for x in merged if x.strip()]

        if nonempty:
            binary_only = all(
                re.fullmatch(r"[01]+", x.replace(" ", ""))
                for x in nonempty
            )

            if binary_only:
                continue

            out.append(" ".join(nonempty))

    return out

def extract_list(raw_json_path):
    """
    Extract individual scholarly list items from PaddleOCR-VL output.

    Generic structural policy:
    - each detected list item is returned independently;
    - numbered enumerations are excluded because numbering alone is
      ambiguous with definitions, properties, proofs and algorithmic steps;
    - strong typographic bullets may occur as separate OCR blocks or as
      several lines inside one OCR block;
    - weak dash-like bullets are accepted only as separate block anchors;
    - separate-block items must form a compact, aligned local sequence;
    - unmarked blocks may be merged only as continuations between confirmed
      anchors;
    - item reconstruction is LaTeX-aware before the official benchmark
      normalize_string() is applied.
    """
    with open(raw_json_path, "r", encoding="utf-8") as f:
        page = json.load(f)

    blocks = page.get("parsing_res_list", [])

    strong_chars = "•●▪◦‣"
    weak_chars = "*-–—"

    strong_line_re = re.compile(
        rf"(?m)^\s*([{re.escape(strong_chars)}])\s+(.+?)"
        rf"(?=^\s*[{re.escape(strong_chars)}]\s+|\Z)",
        re.DOTALL,
    )

    strong_start_re = re.compile(
        rf"^\s*([{re.escape(strong_chars)}])\s+(.+)$",
        re.DOTALL,
    )

    weak_start_re = re.compile(
        rf"^\s*([{re.escape(weak_chars)}])\s+(.+)$",
        re.DOTALL,
    )

    any_anchor_re = re.compile(
        rf"^\s*([{re.escape(strong_chars + weak_chars)}])\s+(.+)$",
        re.DOTALL,
    )

    numbered_re = re.compile(
        r"^\s*(?:\(?\d{1,3}\)?[.)])\s+\S",
        re.DOTALL,
    )

    def bbox(block):
        b = block.get("block_bbox")
        if not isinstance(b, (list, tuple)) or len(b) < 4:
            return None
        try:
            return tuple(float(x) for x in b[:4])
        except Exception:
            return None

    def latex_reconstruct(text):
        """
        Conservative reconstruction for text originating from scientific
        LaTeX.  Preserve mathematical content while reducing OCR/LaTeX
        representation differences before official normalization.
        """
        text = str(text or "").strip()
        if not text:
            return ""

        # PDF/OCR line wrapping is not an item boundary.
        text = re.sub(r"\s*\n\s*", " ", text)

        # Common escaped LaTeX textual characters.
        text = (
            text.replace(r"\&", "&")
                .replace(r"\%", "%")
                .replace(r"\#", "#")
                .replace(r"\_", "_")
                .replace(r"\{", "{")
                .replace(r"\}", "}")
        )

        # Convert LaTeX to readable text when pylatexenc is available.
        # If conversion fails, preserve the original reconstructed string.
        try:
            from pylatexenc.latex2text import LatexNodes2Text
            converted = LatexNodes2Text().latex_to_text(text)
            if converted and converted.strip():
                text = converted
        except Exception:
            pass

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def finalise(text):
        text = latex_reconstruct(text)
        if not text:
            return ""
        return normalize_string(text)

    def aligned(a, b):
        ba = bbox(a)
        bb = bbox(b)
        if ba is None or bb is None:
            return False

        ax1, ay1, ax2, ay2 = ba
        bx1, by1, bx2, by2 = bb

        aw = max(1.0, ax2 - ax1)
        bw = max(1.0, bx2 - bx1)
        ah = max(1.0, ay2 - ay1)

        # Same local column / indentation.
        if abs(ax1 - bx1) > 25:
            return False

        # Avoid crossing strongly different column widths.
        if abs(ax2 - bx2) > max(120.0, 0.35 * max(aw, bw)):
            return False

        # Physical reading direction.
        if by1 < ay1 - 5:
            return False

        gap = by1 - ay2

        # Compact scholarly-list spacing.
        if gap > max(65.0, 2.5 * ah):
            return False

        return True

    def continuation_compatible(anchor, continuation):
        ba = bbox(anchor)
        bc = bbox(continuation)

        if ba is None or bc is None:
            return False

        ax1, ay1, ax2, ay2 = ba
        cx1, cy1, cx2, cy2 = bc

        ah = max(1.0, ay2 - ay1)

        # A continuation should not jump substantially to the left of
        # the item's anchor and must remain in the same local column.
        if cx1 < ax1 - 20:
            return False

        if cx1 > ax2:
            return False

        if cy1 < ay1 - 5:
            return False

        if cy1 - ay2 > max(80.0, 3.0 * ah):
            return False

        return True

    def block_text(block):
        return str(block.get("block_content", "") or "").strip()

    out = []

    # ---------------------------------------------------------------
    # Route A: several STRONG bullet items merged by OCR into one block.
    # Weak '-'/'*'/dash markers are deliberately excluded here because
    # scientific prose and mathematical case distinctions commonly use
    # them inside a single text block.
    # ---------------------------------------------------------------
    for block in blocks:
        if block.get("block_label") != "text":
            continue

        text = block_text(block)
        if not text:
            continue

        matches = list(strong_line_re.finditer(text))

        if len(matches) < 2:
            continue

        markers = [m.group(1) for m in matches]

        # A coherent bullet run uses one typographic bullet family.
        if len(set(markers)) != 1:
            continue

        for match in matches:
            item = finalise(match.group(2))
            if item:
                out.append(item)

    # ---------------------------------------------------------------
    # Route B: list items represented as separate OCR text blocks.
    #
    # Strong and weak bullets are permitted here, but a run requires at
    # least two confirmed anchors with the SAME marker and compatible
    # local geometry.  Continuation blocks are retained only when they
    # lie between two confirmed anchors.
    # ---------------------------------------------------------------
    i = 0

    while i < len(blocks):
        first = blocks[i]

        if first.get("block_label") != "text":
            i += 1
            continue

        first_text = block_text(first)
        m0 = any_anchor_re.match(first_text)

        if not m0:
            i += 1
            continue

        marker = m0.group(1)

        # Never reinterpret numeric scholarly enumerations through this
        # route.
        if numbered_re.match(first_text):
            i += 1
            continue

        anchors = [(i, first, m0.group(2))]
        pending = []

        j = i + 1

        while j < len(blocks):
            cur = blocks[j]

            # A non-text layout object ends the local list run.
            if cur.get("block_label") != "text":
                break

            text = block_text(cur)
            if not text:
                break

            # Explicit numbered structures are a boundary, not a bullet
            # continuation.
            if numbered_re.match(text):
                break

            ma = any_anchor_re.match(text)

            if ma:
                # Different marker => different structure/run.
                if ma.group(1) != marker:
                    break

                prev_anchor = anchors[-1][1]

                if not aligned(prev_anchor, cur):
                    break

                anchors.append((j, cur, ma.group(2)))
                pending = []
                j += 1
                continue

            # Unmarked text is only provisional continuation material.
            # It becomes usable only if another matching bullet anchor
            # subsequently confirms the list.
            if continuation_compatible(anchors[-1][1], cur):
                pending.append((j, cur, text))
                j += 1
                continue

            break

        if len(anchors) >= 2:
            # Reconstruct each confirmed item independently.
            for pos, (idx, anchor_block, body) in enumerate(anchors):
                parts = [body]

                if pos + 1 < len(anchors):
                    next_idx = anchors[pos + 1][0]

                    for k in range(idx + 1, next_idx):
                        cb = blocks[k]

                        if cb.get("block_label") != "text":
                            break

                        ct = block_text(cb)

                        if not ct:
                            continue

                        # Never absorb another explicit marker.
                        if any_anchor_re.match(ct) or numbered_re.match(ct):
                            break

                        if continuation_compatible(anchor_block, cb):
                            parts.append(ct)
                        else:
                            break

                item = finalise(" ".join(parts))

                if item:
                    out.append(item)

            i = anchors[-1][0] + 1
        else:
            i += 1

    # Preserve order while avoiding duplicate predictions generated by
    # overlapping OCR representations.
    deduped = []
    seen = set()

    for item in out:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)

    return deduped


def extract_raw(base_dir, label, pdf):
    import os
    from pathlib import Path

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

    raw_root = os.environ.get("PADDLEOCR_RAW_ROOT")
    if not raw_root:
        raise RuntimeError(
            "PADDLEOCR_RAW_ROOT is not defined. "
            "It must point to the PaddleOCR raw-output directory."
        )

    pdf_name = pdf.pdf_name
    stem = Path(pdf_name).stem
    raw_dir = Path(raw_root) / stem

    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw PaddleOCR directory not found for {pdf_name}: {raw_dir}"
        )

    def page_number(path):
        name = path.stem
        prefix = f"{stem}_"
        if name.startswith(prefix):
            name = name[len(prefix):]
        if name.endswith("_res"):
            name = name[:-4]
        return int(name)

    raw_paths = sorted(
        raw_dir.glob("*_res.json"),
        key=page_number,
    )

    rows = []
    handler = handlers[label]

    for raw_path in raw_paths:
        page = page_number(raw_path)

        values = handler(raw_path)

        for value in values:
            rows.append(
                (
                    pdf_name,
                    page,
                    label,
                    value,
                )
            )

    return True, rows
