"""
This script is the main script for eSciBench; it is the script to call in 
order to run the benchmark.

As of the last update, eSciBench evaluates the following 12 open source tools:
    Camelot, Cermine, GROBID, Nougat, PdfAct, Pdfplumber, PyMuPDF, Refextract, 
    ScienceBeam, ScienceParse, Tabula and Unstructured.

The extractions tools are evaluated on the extraction of the 15 following 
labels:
    Abstract, author affiliation, author names, figure and table caption, 
    article's publication date, author names, displayed equation, footer, 
    header, keyword, list item, bibliographical reference, section title, 
    table content, and article title.

This script can be called with the following options :
    python run_benchmark.py --tool [tool_names] --label [labels] --data_dir [dataset_path]


Author : Noah Tremblay Taillon
Creation date   : June 3,    2025
Last update     : January 1, 2026
"""


import os
import argparse
import sys
import csv
import time
import pandas as pd
from tqdm import tqdm

# IMPORT EXTRACTORS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmark.extractors.mineru.mineru_run import extract_raw as extract_raw_mineru


# IMPORT DATASET UTILS
from benchmark.dataset.extract_gt import extract_ground_truth_json
from benchmark.evaluation.metrics import compute_metrics
from benchmark.extractors.pdf_utils import load_data_json
from benchmark.evaluation.align import align

# REMOVE WARNINGS FROM PYPDF'S CRYPTOGRAPHY DEPRICATION
import warnings

try:
    from cryptography.utils import CryptographyDeprecationWarning
    warnings.filterwarnings(
        "ignore",
        category=CryptographyDeprecationWarning
    )
except ImportError:
    pass


extractor_map:dict[str, callable] = {
    "mineru": extract_raw_mineru,
}
"""
Dictionnary mapping the extractors (keys) to their coresponding processing 
functions (values).
"""

all_labels = [
    "abstract", "affiliation", "author", 
    "citation_author", "citation_container_title",
    "citation_fpage", "citation_issue", "citation_lpage", "citation_pub_place",
    "citation_publisher", "citation_title", "citation_volume", "citation_year",
    "doi", "email", "equation", "figure", "footer", "fpage", "header", 
    "heading", "id", "issue", "journal", "list", "lpage", "paragraph", 
    "pub_date", "publisher", "reference", "reference_label", "section", 
    "table", "title", "volume"
]
"""
List of the union of extracted labels by all the extractors. Is not used in
the code, but serves as a reference of what can be possible.
"""

eSciBench_labels = ["abstract", "affiliation", "author", "caption", 
                    "pub_date", "email", "equation", "footer", "header",
                    "keyword", "list", "reference", "section", "table", 
                    "title"
]
"""
List of all labels being benchmarked in eSciBench.
"""


def run_benchmark(base_dir, labels, tools):
    """
    Run the benchmark pipeline for the specified PDF extraction tools and 
    labels.

    This function coordinates:
      - Running selected extraction tools over the sample PDFs.
      - Collecting both extracted and ground-truth label data.
      - Aggregating per-label and per-PDF timing statistics.
      - Computing evaluation metrics (e.g., precision, recall, F1).
    
    Args:
        base_dir (str): Path to the base directory containing input sample 
            PDFs.
        labels (list of str): List of label types to evaluate (e.g., "title", 
            "abstract").
        tools (list of str): List of extraction tools to use (e.g., "grobid", 
            "cermine").

    Returns:
        pd.DataFrame: A DataFrame containing performance metrics per tool,
                      including timing and extraction quality.
    
    Raises:
        Prints error messages if labels/tools are empty or if tools are not 
            implemented.
        Catches and logs exceptions thrown during individual extraction runs.
    """

    # List of extracted data by the extraction tools
    results:list[dict] = []

    # List of ground-truth data
    ground_truth:list[dict] = []

    # Dictionnary used to hold the time datas for each extraction tool
    timing_stats: dict[str, tuple[float, float, float]] = {}

    # Sanity checks
    if len(labels) == 0 :
        print(f"[ERROR] No defined label")
    if len(tools) == 0 :
        print(f"[ERROR] No defined tool")

    PDFList = []
    PDFList = load_data_json(base_dir) 
    

    for tool in tools :
        tool_start = time.time()
        pdf_count = 0
        label_count = 0

        # 1. Tools needing to generate an extraction file
        if tool == 'grobid': 
            extraction_grobid_if_needed(base_dir)
        elif tool == 'cermine':
            run_cermine_if_needed(base_dir)
        if tool == "docling": 
            extraction_if_needed_docling(base_dir, PDFList)
        elif tool == 'pdfact':
            run_pdfact_if_needed(PDFList, base_dir, labels)
        elif tool == 'sciencebeam' :
            extract_sciencebeam_if_needed(PDFList, base_dir) 
        elif tool == 'nougat':
            extraction_if_needed_nougat(base_dir, PDFList)
        elif tool == 'scienceparse':
            extraction_if_needed_scienceparse(base_dir, PDFList)
        elif tool == 'unstructured':
            extraction_if_needed_unstructured(base_dir, PDFList)
            
        # 2. Extraction function
        extractor = extractor_map.get(tool)
        
        if not extractor:
            print(f"[ERROR] Tool '{tool}' is not implemented")
            continue

        for pdf in tqdm(PDFList):
            pdf_count += 1                                  # For time metrics
            for label in labels :
                
                if label not in eSciBench_labels : continue
                try :
                    call_start = time.time()                # For time metrics
                    # Extract the information with tool. Flag is a boolean, whether or not this tool can extract this label
                    flag, extraction_tuple = extractor(base_dir, label, pdf)
                    call_end = time.time()                  # For time metrics
                    # If flag == True, then the extraction was succesful for this label
                    # We then need the ground-truth for this label. Otherwise, 
                    # the ground-truth is useless, because that tool doesn't extract that label; 
                    # scores would necessarily be 0.
                    if flag:
                        duration = call_end - call_start    # For time metrics
                        for (pdf_name, page, lbl, extracted_text) in extraction_tuple:
                            results.append({
                                "tool": tool,
                                "pdf_name": pdf_name,
                                "page": page,
                                "label": lbl,
                                "data_ex": extracted_text,
                                # Use “duration” as the time for that label‐extraction step
                                "extraction_time": duration
                            })
                            
                        label_count += 1 # Only count labels that the tool can extract, for time metrics
                        
                        # Extracts the gt for the whole document, for this label
                    gt = extract_ground_truth_json(pdf, label, tool)
                    ground_truth.extend(gt)
                    
                except Exception as e :
                    print(f"[ERROR] Failed to run {tool} for label {label}, file {pdf.pdf_name} : {e}")

        
        #============= Token count per label =============#
        # Count and print word counts per label
        df = pd.DataFrame(ground_truth)
        df['word_count'] = df['data_gt'].apply(lambda x: len(x.split()))

        summary = df.groupby('label').agg(
            num_entries=('data_gt', 'count'),
            total_words=('word_count', 'sum'),
            #unique_pdfs=('pdf_name', 'nunique')
        )
        print(summary)
        #============= Token count per label =============#
        

        # 3. Put all extracted information for one page for one label in the same string
        tool_end = time.time()                              # For time metrics
        total_time_tool = tool_end - tool_start
        # Avoid division by zero:
        avg_time_per_label_tool = total_time_tool / label_count if label_count else 0.0
        avg_time_per_pdf_tool = total_time_tool / pdf_count if pdf_count else 0.0

        # Save them so we can inject into every row for “tool” later:
        timing_stats[tool] = (
            total_time_tool,
            avg_time_per_label_tool,
            avg_time_per_pdf_tool
        )

        # Print a summary for this tool:
        print(
            f"→ [{tool}] total_time: {total_time_tool:.2f}s  "
            f"(avg per label: {avg_time_per_label_tool:.2f}s, "
            f"avg per PDF: {avg_time_per_pdf_tool:.2f}s)"
        )
    
    # Make the results and ground truth into alignable DataFrames
    results_df = pd.DataFrame(results, columns=["tool", "pdf_name", "page", "label", "data_ex"])
    ground_truth_df = pd.DataFrame(ground_truth, columns=["tool", "pdf_name", "page", "label", "data_gt"])

    if results_df.empty:
        print("[WARNING] Extraction is empty. Skipping metrics.")
        return pd.DataFrame(), pd.DataFrame()
    elif ground_truth_df.empty:
        print("[WARNING] Ground truth is empty.")
        averaged_df = pd.DataFrame(ground_truth, columns=["tool", "pdf_name", "page", "label", "data_gt"])
    # Merge and align (fuzzy search) the two DataFrames
    aligned_df = align(results_df, ground_truth_df)

    # Debug prints
    #print()
    #print(aligned_df)

    if aligned_df.empty:
        print("[WARNING] Aligned DataFrame is empty. Skipping metrics computation.")
        merged_df_metrics = pd.DataFrame()
        averaged_df = pd.DataFrame()
    else:
        # 5.2. Reorder columns
        merged_df = aligned_df[[
            "tool", "pdf_name", "page", "label",
            "data_ex", "data_gt",
            "tp_alignment", "fp_alignment", "fn_alignment",
            "f1_alignment", "precision_alignment", "recall_alignment"
        ]]

        merged_df_metrics, averaged_df = compute_metrics(merged_df)

    merged_df_metrics["total_time_tool"] = merged_df_metrics["tool"].map(lambda t: timing_stats.get(t, (0, 0, 0))[0])
    merged_df_metrics["avg_time_per_label_tool"] = merged_df_metrics["tool"].map(lambda t: timing_stats.get(t, (0, 0, 0))[1])
    merged_df_metrics["avg_time_per_pdf_tool"] = merged_df_metrics["tool"].map(lambda t: timing_stats.get(t, (0, 0, 0))[2])



    if not merged_df_metrics.empty:
        merged_df_metrics["total_time_tool"] = merged_df_metrics["tool"].map(
            lambda t: timing_stats.get(t, (0, 0, 0))[0]
        )
        merged_df_metrics["avg_time_per_label_tool"] = merged_df_metrics["tool"].map(
            lambda t: timing_stats.get(t, (0, 0, 0))[1]
        )
        merged_df_metrics["avg_time_per_pdf_tool"] = merged_df_metrics["tool"].map(
            lambda t: timing_stats.get(t, (0, 0, 0))[2]
        )   
    
    return merged_df_metrics, averaged_df


def main():
    """
    Entry point for the PDF extraction benchmark.
    Parse command-line arguments and run the benchmark pipeline.

    This function handles:
      - Argument parsing via argparse.
      - Validating tool names.
      - Invoking the benchmark runner.
      - Saving results to a CSV file.

    Side Effects:
        Writes the evaluation results to './data/results/benchmark_results.csv'.
        Prints progress and error messages to the console.
    """

    # Directory where the sample data is located
    base_dir = "/data/rali5/Tmp/erudit/yves/escibench_project/benchmark_100_clean"

    # Argument parser setup
    parser = argparse.ArgumentParser(
                        description="Run PDF extraction benchmark.")
    parser.add_argument("--tools", nargs="+", default=["all"],
                        help="Extraction tools to run (e.g., grobid cermine); 'all' if not specified")
    parser.add_argument("--labels", nargs="+", 
                        default=eSciBench_labels,
                        help="Labels to evaluate; 'all if not specified")
    parser.add_argument("--data_dir", default=base_dir,
                        help="Path to the dataset directory")
    args = parser.parse_args()

    available_tools = list(extractor_map.keys())
    if "all" in args.tools:
        tools = available_tools
    else:
        invalid_tools = [t for t in args.tools if t not in available_tools]
        if invalid_tools:
            print(f"[ERROR] Invalid tool(s): {', '.join(invalid_tools)}")
            print(f"[INFO] Available tools: {', '.join(available_tools)}")
            return
        tools = args.tools
    
    # Results DataFrame
    result_df, averaged_df = run_benchmark(args.data_dir, args.labels, tools)

    # Saving results to CSV
    os.makedirs("./data/results", exist_ok=True)
    result_df.to_csv(
        "./data/results/benchmark_results - mineru.csv",
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\"
    )

    averaged_df.to_csv(
        "./data/results/avg - mineru.csv",
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\"
    )
    os.makedirs("./data/results/tables", exist_ok=True)
    #generate_all_tables(averaged_df, output_dir="./data/results/tables/")
    print("Benchmark complete. Results saved to data/results/benchmark_results.csv")


if __name__ == "__main__":
    main()
