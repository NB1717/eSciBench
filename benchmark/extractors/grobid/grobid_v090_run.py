import os
import re
import unicodedata
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from grobid_client.grobid_client import GrobidClient
from ..pdf_utils import PDF
from ...normalisation import normalize_string

# docker run -it --rm -p 8070:8070 grobid/grobid:0.8.2

NS = {'tei': 'http://www.tei-c.org/ns/1.0'}

label_to_tag: dict[str,str] = {
    # ==== Metadata (from <teiHeader>) ====
    'title': './/tei:teiHeader//tei:titleStmt/tei:title',
    'abstract': './/tei:teiHeader//tei:profileDesc//tei:abstract//tei:p',
    'author': './/tei:teiHeader//tei:author/tei:persName',
    'email': './/tei:teiHeader//tei:email',
    'affiliation': './/tei:teiHeader//tei:affiliation/tei:note[@type="raw_affiliation"]',
    'publisher': './/tei:teiHeader//tei:publisher',
    'journal': './/tei:teiHeader//tei:monogr//tei:title',
    'doi': './/tei:teiHeader//tei:idno[@type="DOI"]',
    'pub_date': './/tei:teiHeader//tei:publicationStmt/tei:date[@type="published"]',
    'keyword': './/tei:teiHeader//tei:profileDesc//tei:textClass//tei:keywords//tei:term',

    # ==== Body content (from <text><body>) ====
    'section': './/tei:text//tei:body//tei:head',
    'paragraph': './/tei:text//tei:body//tei:p',
    'footer' : './/tei:text//tei:body//tei:note//tei:p',
    'equation' : './/tei:text//tei:body//tei:formula',
    'figure' : './/tei:text//tei:body//tei:figure',
    'table': './/tei:figure[@type="table"]',
    'caption': './/tei:figure',

    # ==== Bibliography content (from <text><back><biblStruct>) ====
    'reference': './/tei:text//tei:biblStruct',
    'citation_title': './/tei:text//tei:biblStruct/tei:analytic/tei:title[@type="main"]',
    'citation_author': './/tei:text//tei:biblStruct/tei:analytic/tei:author/tei:persName',
    'citation_year': './/tei:text//tei:biblStruct/tei:monogr/tei:imprint/tei:date',
    'citation_container_title': './/tei:text//tei:biblStruct/tei:monogr/tei:title',                      # i.e. The whole book if the citation is from a chapter
    'citation_publisher': './/tei:text//tei:biblStruct/tei:monogr/tei:imprint/tei:publisher',
    'citation_pub_place': './/tei:text//tei:biblStruct/tei:monogr/tei:imprint/tei:pubPlace',
    'citation_volume': './/tei:text//tei:biblStruct/tei:monogr/tei:imprint/tei:biblScope[@unit="volume"]',
    'citation_issue': './/tei:text//tei:biblStruct/tei:monogr/tei:imprint/tei:biblScope[@unit="issue"]',
    'citation_fpage': './/tei:text//tei:biblStruct//tei:biblScope[@unit="page"]',
    'citation_lpage': './/tei:text//tei:biblStruct//tei:biblScope[@unit="page"]',
    'reference_label': './/tei:text//tei:ref'
}
"""
Mapping of (almost) all labels to xml tags possible with Grobid.
"""


def extract_grobid(base_dir:str) -> None:
    """
    Run Grobid's full-text processing on PDFs in the given directory.

    This generates `.grobid.tei.xml` files in the same directory, using the 
    Grobid server via the Grobid Python client.

    Args:
        base_dir (str): Directory containing PDF files to be processed.
    """
    config_path = "./benchmark/extractors/grobid/config.json"
    client = GrobidClient(config_path=config_path)
    client.process("processFulltextDocument", base_dir, base_dir, n=5, include_raw_citations=True, include_raw_affiliations=True)


def extraction_if_needed(base_dir:str) -> None:
    """
    Only run Grobid extraction if there are PDF files that have not yet been processed.

    Checks for existing `.grobid.tei.xml` files. If one PDF file doesn't have a corresponding
    `.grobid.tei.xml` file, all the files have to be re-extracted (since Grobid extracts per
    folder, and not per file).

    Args:
        base_dir (str): Directory containing PDF files and expected output XMLs.
    """
    already_done = set(
        f.replace('.grobid.tei.xml', '') for f in os.listdir(base_dir) if f.endswith('.grobid.tei.xml')
    )
    all_pdfs = set(
        os.path.splitext(f)[0] for f in os.listdir(base_dir) if f.lower().endswith('.pdf')
    )
    to_process = all_pdfs - already_done
    if to_process :
        extract_grobid(base_dir)


def normalize_string(text):
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    # Remove LaTeX \text{...}
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)

    # x_{q_1} -> x_q_1
    text = re.sub(r'([A-Za-z0-9])_\{([^{}]+)\}', r'\1_\2', text)

    # \cdots -> ⋯
    text = text.replace(r"\cdots", "⋯")

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_equation(text):
    text = normalize_string(text)

    # Remove equation number at end: (2.10), (A.12), (15), ...
    text = re.sub(
        r"\s*[,.;:]?\s*\([A-Za-z0-9.\-]+\)\s*$",
        "",
        text,
    )

    # Normalize spaces around _ and ^
    text = re.sub(r"\s*_\s*", "_", text)
    text = re.sub(r"\s*\^\s*", "^", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.;:])", r"\1", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def extract_raw(base_dir:str, label:str, pdf:PDF) -> tuple[bool, list[tuple]]:
    """
    Main Grobid extractor labeled for a given PDF and label.

    Handles various content types including metadata, section headers, body content,
    and bibliography references, returning all matching elements.

    All extractable labels are in `label_to_tag`.

    Args:
        base_dir (str): Path to directory containing the PDF files (unused).
        label (str): The target label to extract (e.g., 'author', 'title').
        pdf (PDF): PDF metadata object containing filename and path.

    Returns:
        tuple:
            - bool: Whether extraction was successful.
            - list[tuple]: List of (pdf_name, page_number, label, extracted_text) entries.

    Raises:
        Logs parsing or file errors to stdout if TEI file is missing or corrupted.
    """

    results = []

    filepath = pdf.filepath.replace('.pdf', '.grobid.tei.xml')
    filename = pdf.pdf_name
    page_number = 0

    try :
        tree = ET.parse(filepath)
        root = tree.getroot()

        if label not in label_to_tag:
            print(f"[ERROR] Unsupported label '{label}' for Grobid")
            return False, results
        
        tag = label_to_tag[label]

        # Special treatment for author names
        if label == 'author':
            for author in root.findall('.//tei:teiHeader//tei:author/tei:persName', namespaces=NS):
                # Collect all forenames (first, middle, etc.)
                forenames = []
                for f in author.findall('tei:forename', namespaces=NS):
                    if f.text:
                        text = f.text.strip()
                        forenames.append(text)

                # Get the surname
                surname_elem = author.find('tei:surname', namespaces=NS)
                surname = surname_elem.text.strip() if surname_elem is not None and surname_elem.text else ''

                # Combine full name
                name_parts = forenames + ([surname] if surname else [])
                full_name = normalize_string(' '.join(name_parts))
                if full_name:
                    results.append((filename, page_number, label, full_name))
                    
            return True, results


        elif label == 'abstract':
            paragraphs = []
            for elem in root.findall(tag, namespaces=NS):
                text = normalize_string(re.sub(r'\s+', ' ', ''.join(elem.itertext()).strip()))
                if text:
                    paragraphs.append(text)
            if paragraphs:
                abstract_text = normalize_string(" ".join(paragraphs))
                results.append((filename, page_number, label, abstract_text))
            return True, results
        
        elif label == 'equation':

            previous = None
            
            for formula in root.findall(label_to_tag[label], namespaces=NS):

                # Remove equation labels like (1), (2), ...
                for lbl in formula.findall("tei:label", namespaces=NS):
                    formula.remove(lbl)

                text = ''.join(formula.itertext())
                text = normalize_equation(re.sub(r'\s+', ' ', text.strip()))

                if not text:
                    continue
                
                low = text.lower()

                if re.match(
                    r"^(proof|figure|theorem|lemma|corollary|remark|example)\b",
                    low,
                ):
                    continue

                # Remove trivial false positives
                if text in {
                    ")", "(", "]", "[", "{", "}", ".", ",", ";", ":"
                }:
                    continue

                # Merge only if previous equation ended with an incomplete equation number
                if (
                    previous is not None
                    and text.startswith(")")
                    and re.search(r"\([A-Za-z0-9.]+\s*$", previous[3])
                ):

                    previous[3] += text
                    continue
                row = [filename, page_number, label, text]
                results.append(row)

                previous = row
                
            results = [tuple(r) for r in results]
            return True, results


        # Special case: extract table content (not just captions)
        elif label == 'table':
            for fig in root.findall(label_to_tag[label], namespaces=NS):
                table_elem = fig.find('tei:table', namespaces=NS)
                if table_elem is None:
                    continue

                all_text = []
                for row in table_elem.findall('tei:row', namespaces=NS):
                    for cell in row.findall('tei:cell', namespaces=NS):
                        cell_text = ''.join(cell.itertext()).strip()
                        if cell_text:
                            all_text.append(normalize_string(cell_text))

                if all_text:
                    full_table_text = normalize_string(" ".join(all_text))
                    results.append((filename, page_number, label, full_table_text))

            return True, results
        
        elif label == 'caption':
            for fig in root.findall(label_to_tag[label], namespaces=NS):
                figdesc = fig.find('tei:figDesc', namespaces=NS)
                if figdesc is not None and figdesc.text:
                    # Optional: clean "Figure X:" or "Table Y:" prefix
                    raw = figdesc.text.strip()
                    clean = re.sub(r'^(figure|table)\s*\d+\s*:\s*', '', raw, flags=re.IGNORECASE)
                    normalized = normalize_string(clean)
                    if normalized:
                        results.append((filename, page_number, 'caption', normalized))
            return True, results
        
        # Bibliographic references as APA strings
        elif label == 'reference':
            
            for bibl in root.findall(label_to_tag['reference'], namespaces=NS):
                raw_elem = bibl.find('.//tei:note[@type="raw_reference"]', namespaces=NS)
                if raw_elem is not None and raw_elem.text :
                    raw = raw_elem.text.strip()
                    raw = normalize_string(re.sub(r';\s*$', '', raw))
                    results.append((filename, page_number, label, raw))
                    
            return True, results

        # Special treatment for page numbers in bibliographical references
        elif label in ['citation_fpage', 'citation_lpage']:
            for bibl in root.findall('.//tei:biblStruct//tei:biblScope[@unit="page"]', NS):
                fpage = normalize_string(bibl.attrib.get('from'))
                lpage = normalize_string(bibl.attrib.get('to'))

                if label == 'citation_fpage' and fpage:
                    results.append((filename, page_number, 'citation_fpage', fpage))
                elif label == 'citation_lpage' and lpage:
                    results.append((filename, page_number, 'citation_lpage', lpage))
            return True, results
        
        # Treatment of the regular tags
        else :
            for elem in root.findall(tag, namespaces=NS):
                text = normalize_string(re.sub(r'\s+', ' ', ''.join(elem.itertext()).strip()))
                if label == "section":
                    low = text.lower()

                    if low.startswith("figure"):
                        continue

                    if low.startswith("fig"):
                        continue


                    if low.startswith("table"):
                        continue

                    

                    if re.match(
                        r"^(definition|theorem|lemma|remark|proposition|corollary)\b",
                        low,
                    ):
                        continue

                    # Remove algorithm captions misparsed as section titles
                    if re.match(r"^algorithm\s+\d+\b", low):
                        continue

                    # Remove publication banners
                    if low.startswith("accepted by"):
                        continue

                    # Remove obvious figure abbreviations
                    if re.match(r"^(fig\.?|figure)\b", low):
                        continue

                    # Remove symbol-only garbage
                    if re.fullmatch(r"[□⇒→∂∆λµα-ω+\-*/=().,\[\]{} ]+", text):
                        continue


                    if re.fullmatch(r"[0-9(). αβγδεζηθικλμνξοπρστυφχψω+\-=/,* ]+", text):
                        continue

                    text = re.sub(r'^\s*(?:\d+(?:\.\d+)*|[IVXLCDMivxlcdm]+)[\.\)]?\s+', '', text) # Remove section numbering
                    
                if text :
                    results.append((filename, page_number, label, text))
                    
    except Exception as e :
        print(f"[ERROR] Grobid's tei.xml file generation failed: {e}")
        return False, results
    
    return True, results
