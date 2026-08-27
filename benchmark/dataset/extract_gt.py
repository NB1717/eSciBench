import os
import json
from benchmark.normalisation import normalize_string

def extract_ground_truth_json(pdf, label, tool):
    results = []
    base = os.path.splitext(pdf.pdf_name)[0]

    # Prefer the JSON next to the PDF inside the benchmark dataset directory.
    gt_path = os.path.join(os.path.dirname(pdf.filepath), base + ".json")

    if not os.path.exists(gt_path):
        return results

    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def add(text, out_label):
        text = normalize_string(text)
        if text:
            results.append({
                "tool": tool,
                "pdf_name": pdf.pdf_name,
                "page": 0,
                "label": out_label,
                "data_gt": text
            })

    if label == "title":
        for x in data.get("title", []):
            add(x, "title")

    elif label == "abstract":
        for x in data.get("abstract", []):
            add(x, "abstract")

    elif label == "caption":
        for x in data.get("caption", []):
            add(x, "caption")

    elif label == "keyword":
        for x in data.get("keyword", []):
            add(x, "keyword")

    elif label == "equation":
        for x in data.get("equation", []):
            add(x, "equation")

    elif label == "header":
        for x in data.get("header", []):
            add(x, "header")

    elif label == "footer":
        for x in data.get("footer", []):
            add(x, "footer")

    elif label == "table":
        for x in data.get("table", []):
            add(x, "table")

    elif label == "list":
        for x in data.get("list", []):
            add(x, "list")

    elif label == "reference":
        for x in data.get("bibliography", []):
            add(x.get("text", ""), "reference")

    elif label == "section":
        for x in data.get("section", []):
            add(x.get("title", ""), "section")

    elif label == "author":
        for a in data.get("author", []):
            add(a.get("author_name", ""), "author")

    elif label == "affiliation":
        for a in data.get("author", []):
            for aff in a.get("affiliation", []):
                add(aff, "affiliation")

    elif label == "email":
        for a in data.get("author", []):
            add(a.get("email", ""), "email")

    elif label == "pub_date":
        add(data.get("date", ""), "pub_date")

    return results
