import os
import re
import json
from html import unescape
from benchmark.normalisation import normalize_string

MARKER_OUTPUT_DIR = "/data/rali5/Tmp/erudit/yves/escibench_project/outputs/marker_outputs"

def html_to_text(html):
    html = unescape(html or "")
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def strip_section_number(text):
    return re.sub(r"^\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)[\.\)]?\s+", "", text, flags=re.I).strip()

def strip_caption_prefix(text):
    return re.sub(r"^\s*(figure|fig\.?|table)\s*\d+[\.:]?\s*", "", text, flags=re.I).strip()



def looks_like_author_name(text):
    text = text or ""
    t = re.sub(r"&lt;/?sup&gt;|<[^>]+>", " ", text)
    t = re.sub(r"\s+", " ", t).strip(" ,;*0123456789")
    if not t:
        return False

    bad = r"university|institute|department|school|center|centre|laboratory|college|faculty|academy|academia|cnrs|infn|caltech|flatiron|street|avenue|road|via|email|@|abstract|keywords"
    if re.search(bad, t, re.I):
        return False

    if len(t.split()) > 8:
        return False

    toks = [x for x in re.split(r"\s+", t) if re.search(r"[A-Za-zÀ-ÿ]", x)]
    if len(toks) < 2:
        return False

    return True

def normalize_equation_text(text):
    text = text or ""
    text = text.replace("\\\\", " ")
    text = re.sub(r"\\operatorname\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathcal\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1 / \2", text)
    text = re.sub(r"\\sum_\{?([^{}\s]*)\}?\^\{?([^{}\s]*)\}?", r"sum_\1^\2", text)
    text = re.sub(r"\\int_\{?([^{}\s]*)\}?\^\{?([^{}\s]*)\}?", r"int_\1^\2", text)
    text = text.replace("\\leq", "<=").replace("\\geq", ">=")
    text = text.replace("\\cdot", " ").replace("\\times", " ")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\|", "|")
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"\((\d+)\)\s*$", "", text)
    text = re.sub(r"[^0-9a-zA-Z_<>=+\-*/^().,| ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for x in node:
            yield from walk(x)

def marker_json_path(pdf):
    base = os.path.splitext(pdf.pdf_name)[0]
    return os.path.join(MARKER_OUTPUT_DIR, base, base + ".json")

def extract_raw(base_dir, label, pdf):
    path = marker_json_path(pdf)
    results = []

    if not os.path.exists(path):
        return False, results

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = list(walk(data))

    ref_started = False
    abstract_started = False
    abstract_done = False
    first_title_seen = False
    front_texts = []
    pending_list_items = []
    pending_list_page = None

    def flush_list():
        nonlocal pending_list_items, pending_list_page, results
        if pending_list_items:
            merged = " ".join(pending_list_items)
            results.append((pdf.pdf_name, pending_list_page or 0, "list", normalize_string(merged)))
            pending_list_items = []
            pending_list_page = None

    for node in nodes:
        bt = node.get("block_type")
        text = html_to_text(node.get("html", ""))
        if not text:
            continue

        page = 0
        m = re.search(r"/page/(\d+)/", node.get("id", ""))
        if m:
            page = int(m.group(1)) + 1

        if bt == "SectionHeader" and re.search(r"\breferences?\b|\bbibliograph", text, re.I):
            ref_started = True

        if page == 1 and bt in ["Text", "SectionHeader"] and text:
            front_texts.append((bt, text))

        if label == "title" and bt == "SectionHeader" and page == 1:
            return True, [(pdf.pdf_name, page, "title", normalize_string(text))]

        elif label == "section" and bt == "SectionHeader":
            # Do not count the article title as a section.
            if text.strip().upper().startswith("SURPRISENET:"):
                continue
            sec = strip_section_number(text)
            results.append((pdf.pdf_name, page, "section", normalize_string(sec)))

        elif label == "caption" and bt == "Caption":
            cap = strip_caption_prefix(text)
            results.append((pdf.pdf_name, page, "caption", normalize_string(cap)))

        elif label == "equation" and bt == "Equation":
            results.append((pdf.pdf_name, page, "equation", normalize_equation_text(text)))

        elif label == "table" and bt == "Table":
            results.append((pdf.pdf_name, page, "table", normalize_string(text)))

        elif label == "header" and bt == "PageHeader":
            results.append((pdf.pdf_name, page, "header", normalize_string(text)))

        elif label == "list" and bt == "ListItem" and not ref_started:
            if pending_list_page is None:
                pending_list_page = page
            elif page != pending_list_page:
                flush_list()
                pending_list_page = page
            pending_list_items.append(text)

        elif label == "abstract":
            if bt == "SectionHeader" and re.fullmatch(r"\s*abstract\s*", text, flags=re.I):
                abstract_started = True
                continue
            if abstract_started and not abstract_done:
                if bt == "Text" and text:
                    results.append((pdf.pdf_name, page, "abstract", normalize_string(text)))
                    abstract_done = True
                elif bt == "SectionHeader" and not re.fullmatch(r"\s*abstract\s*", text, flags=re.I):
                    abstract_done = True

        elif label == "reference" and bt == "ListItem" and ref_started:
            results.append((pdf.pdf_name, page, "reference", normalize_string(text)))

    if label == "list":
        flush_list()

    if label in ["author", "affiliation", "email"]:
        # Front matter heuristic: use only first-page text blocks before ABSTRACT.
        front = []
        for bt, tx in front_texts:
            if bt == "SectionHeader" and re.fullmatch(r"\s*abstract\s*", tx, flags=re.I):
                break
            front.append((bt, tx))

        text_blocks = [tx for bt, tx in front if bt == "Text"]

        if label == "email":
            for tx in text_blocks:
                for em in re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", tx):
                    results.append((pdf.pdf_name, 1, "email", normalize_string(em)))

        elif label == "affiliation":
            for tx in text_blocks:
                if "@" not in tx and (
                    re.search(r"university|institute|school|department|laboratory|centre|center|college|academia", tx, re.I)
                    or re.search(r"\b[a-zA-Z .]+,\s*[A-Z][a-zA-Z .]+$", tx)
                ):
                    results.append((pdf.pdf_name, 1, "affiliation", normalize_string(tx)))

        elif label == "author":
            for candidate in text_blocks[:4]:
                if "@" in candidate:
                    continue
                # Remove affiliation superscripts but keep names.
                candidate = re.sub(r"&lt;sup&gt;.*?&lt;/sup&gt;|<sup>.*?</sup>", " ", candidate)
                parts = re.split(r"\s{2,}|,|;|\band\b|\&", candidate)
                parts = [p.strip() for p in parts if p.strip()]
                for p in parts:
                    if looks_like_author_name(p):
                        results.append((pdf.pdf_name, 1, "author", normalize_string(p)))
                if results:
                    break

    supported = label in ["title", "section", "caption", "equation", "table", "header", "list", "reference", "abstract", "author", "affiliation", "email"]
    return supported, results
