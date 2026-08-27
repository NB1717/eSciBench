import argparse
import base64
import csv
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
import requests


MODEL_NAME = "Unlimited-OCR"
MODEL_REPOSITORY = "baidu/Unlimited-OCR"
MODEL_REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"

PROMPT = "document parsing."
DPI = 300
IMAGE_MODE = "gundam"
MAX_TOKENS = 30000
TEMPERATURE = 0
REQUEST_TIMEOUT_SECONDS = 1200
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 5

PAGE_TIMING_FIELDS = [
    "pdf_name",
    "page",
    "status",
    "inference_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "characters",
    "attempts",
    "raw_page_path",
    "error",
]

PDF_TIMING_FIELDS = [
    "pdf_name",
    "page_count",
    "successful_pages",
    "failed_pages",
    "conversion_seconds",
    "inference_wall_seconds",
    "total_pdf_seconds",
]


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def append_csv(path, fieldnames, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        if write_header:
            writer.writeheader()

        writer.writerow(record)
        file.flush()
        os.fsync(file.fileno())


def check_server(server_url):
    response = requests.get(
        f"{server_url}/health",
        timeout=30,
    )
    response.raise_for_status()

    models_response = requests.get(
        f"{server_url}/v1/models",
        timeout=30,
    )
    models_response.raise_for_status()

    available_models = [
        model["id"]
        for model in models_response.json().get("data", [])
    ]

    if MODEL_NAME not in available_models:
        raise RuntimeError(
            f"{MODEL_NAME} is not available. "
            f"Server models: {available_models}"
        )


def discover_dataset(data_dir, expected_pdf_count):
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    json_paths = sorted(data_dir.glob("*.json"))

    pdf_stems = {path.stem for path in pdf_paths}
    json_stems = {path.stem for path in json_paths}

    missing_json = sorted(pdf_stems - json_stems)
    missing_pdf = sorted(json_stems - pdf_stems)

    if missing_json:
        raise RuntimeError(
            f"PDF files without matching GT JSON: {missing_json}"
        )

    if missing_pdf:
        raise RuntimeError(
            f"GT JSON files without matching PDF: {missing_pdf}"
        )

    if expected_pdf_count is not None:
        if len(pdf_paths) != expected_pdf_count:
            raise RuntimeError(
                f"Expected {expected_pdf_count} PDFs, "
                f"but found {len(pdf_paths)}"
            )

        if len(json_paths) != expected_pdf_count:
            raise RuntimeError(
                f"Expected {expected_pdf_count} GT JSON files, "
                f"but found {len(json_paths)}"
            )

    return pdf_paths


def load_successful_pages(master_jsonl_path):
    successful_pages = set()

    if not master_jsonl_path.exists():
        return successful_pages

    with master_jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSONL at line {line_number}: {error}"
                ) from error

            if record.get("status") == "success":
                key = (
                    record.get("pdf_name"),
                    int(record.get("page")),
                )
                successful_pages.add(key)

    return successful_pages


def render_pdf_pages(pdf_path, temporary_root):
    pdf_temporary_dir = temporary_root / pdf_path.stem

    if pdf_temporary_dir.exists():
        shutil.rmtree(pdf_temporary_dir)

    pdf_temporary_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    conversion_start = time.perf_counter()

    document = fitz.open(str(pdf_path))

    try:
        matrix = fitz.Matrix(DPI / 72, DPI / 72)

        for page_index, page in enumerate(document):
            image_path = (
                pdf_temporary_dir
                / f"page_{page_index + 1:04d}.png"
            )

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )
            pixmap.save(str(image_path))
            image_paths.append(image_path)
    finally:
        document.close()

    conversion_seconds = time.perf_counter() - conversion_start

    return image_paths, conversion_seconds, pdf_temporary_dir


def encode_image(image_path):
    with image_path.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def build_payload(image_path):
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encode_image(image_path),
                        },
                    },
                ],
            }
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "skip_special_tokens": False,
        "images_config": {
            "image_mode": IMAGE_MODE,
        },
    }


def infer_page(server_url, pdf_path, page_number, image_path):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        request_start = time.perf_counter()

        try:
            response = requests.post(
                f"{server_url}/v1/chat/completions",
                json=build_payload(image_path),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            inference_seconds = time.perf_counter() - request_start

            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:2000]}"
                )

            response_json = response.json()
            choices = response_json.get("choices", [])

            if not choices:
                raise RuntimeError(
                    "Server response contains no choices"
                )

            message = choices[0].get("message", {})
            raw_output = message.get("content")

            if raw_output is None:
                raise RuntimeError(
                    "Server response contains no message content"
                )

            usage = response_json.get("usage") or {}

            return {
                "status": "success",
                "pdf_name": pdf_path.name,
                "pdf_stem": pdf_path.stem,
                "page": page_number,
                "raw_output": raw_output,
                "inference_seconds": inference_seconds,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get(
                    "completion_tokens",
                    0,
                ),
                "total_tokens": usage.get("total_tokens", 0),
                "characters": len(raw_output),
                "attempts": attempt,
                "error": "",
                "response_id": response_json.get("id"),
                "finish_reason": choices[0].get("finish_reason"),
                "created": response_json.get("created"),
                "system_fingerprint": response_json.get(
                    "system_fingerprint"
                ),
            }

        except Exception as error:
            inference_seconds = time.perf_counter() - request_start
            last_error = f"{type(error).__name__}: {error}"

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)

    return {
        "status": "failed",
        "pdf_name": pdf_path.name,
        "pdf_stem": pdf_path.stem,
        "page": page_number,
        "raw_output": "",
        "inference_seconds": inference_seconds,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "characters": 0,
        "attempts": MAX_RETRIES,
        "error": last_error,
        "response_id": None,
        "finish_reason": None,
        "created": None,
        "system_fingerprint": None,
    }


def save_page_result(
    result,
    raw_pages_dir,
    master_jsonl_path,
    page_timing_path,
):
    pdf_page_dir = raw_pages_dir / result["pdf_stem"]
    pdf_page_dir.mkdir(parents=True, exist_ok=True)

    raw_page_path = (
        pdf_page_dir
        / f"page_{result['page']:04d}.md"
    )

    if result["status"] == "success":
        with raw_page_path.open("w", encoding="utf-8") as file:
            file.write(result["raw_output"])

    master_record = {
        "model_name": MODEL_NAME,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "pdf_name": result["pdf_name"],
        "page": result["page"],
        "status": result["status"],
        "raw_output": result["raw_output"],
        "raw_page_path": str(raw_page_path),
        "inference_seconds": round(
            result["inference_seconds"],
            6,
        ),
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
        "characters": result["characters"],
        "attempts": result["attempts"],
        "error": result["error"],
        "response_id": result["response_id"],
        "finish_reason": result["finish_reason"],
        "created": result["created"],
        "system_fingerprint": result["system_fingerprint"],
        "settings": {
            "prompt": PROMPT,
            "dpi": DPI,
            "image_mode": IMAGE_MODE,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
    }

    append_jsonl(master_jsonl_path, master_record)

    timing_record = {
        "pdf_name": result["pdf_name"],
        "page": result["page"],
        "status": result["status"],
        "inference_seconds": round(
            result["inference_seconds"],
            6,
        ),
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
        "characters": result["characters"],
        "attempts": result["attempts"],
        "raw_page_path": str(raw_page_path),
        "error": result["error"],
    }

    append_csv(
        page_timing_path,
        PAGE_TIMING_FIELDS,
        timing_record,
    )


def process_pdf(
    pdf_path,
    server_url,
    temporary_root,
    raw_pages_dir,
    master_jsonl_path,
    page_timing_path,
    pdf_timing_path,
    successful_pages,
    workers,
):
    total_pdf_start = time.perf_counter()

    image_paths, conversion_seconds, pdf_temporary_dir = (
        render_pdf_pages(pdf_path, temporary_root)
    )

    page_count = len(image_paths)
    pages_to_run = []

    for page_number, image_path in enumerate(
        image_paths,
        start=1,
    ):
        key = (pdf_path.name, page_number)
        raw_page_path = (
            raw_pages_dir
            / pdf_path.stem
            / f"page_{page_number:04d}.md"
        )

        if key in successful_pages and raw_page_path.exists():
            print(
                f"[SKIP] pdf={pdf_path.name} "
                f"page={page_number}",
                flush=True,
            )
            continue

        pages_to_run.append((page_number, image_path))

    successful_count = page_count - len(pages_to_run)
    failed_count = 0

    inference_start = time.perf_counter()

    try:
        if pages_to_run:
            with ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                future_map = {
                    executor.submit(
                        infer_page,
                        server_url,
                        pdf_path,
                        page_number,
                        image_path,
                    ): page_number
                    for page_number, image_path in pages_to_run
                }

                for future in as_completed(future_map):
                    result = future.result()

                    save_page_result(
                        result,
                        raw_pages_dir,
                        master_jsonl_path,
                        page_timing_path,
                    )

                    if result["status"] == "success":
                        successful_count += 1
                    else:
                        failed_count += 1

                    print(
                        f"[{result['status'].upper()}] "
                        f"pdf={result['pdf_name']} "
                        f"page={result['page']} "
                        f"seconds="
                        f"{result['inference_seconds']:.3f} "
                        f"tokens="
                        f"{result['completion_tokens']} "
                        f"characters={result['characters']}",
                        flush=True,
                    )
    finally:
        inference_wall_seconds = (
            time.perf_counter() - inference_start
        )

        if pdf_temporary_dir.exists():
            shutil.rmtree(pdf_temporary_dir)

    total_pdf_seconds = (
        time.perf_counter() - total_pdf_start
    )

    pdf_timing_record = {
        "pdf_name": pdf_path.name,
        "page_count": page_count,
        "successful_pages": successful_count,
        "failed_pages": failed_count,
        "conversion_seconds": round(
            conversion_seconds,
            6,
        ),
        "inference_wall_seconds": round(
            inference_wall_seconds,
            6,
        ),
        "total_pdf_seconds": round(
            total_pdf_seconds,
            6,
        ),
    }

    append_csv(
        pdf_timing_path,
        PDF_TIMING_FIELDS,
        pdf_timing_record,
    )

    print(
        f"[PDF COMPLETE] pdf={pdf_path.name} "
        f"pages={page_count} "
        f"success={successful_count} "
        f"failed={failed_count} "
        f"conversion_seconds={conversion_seconds:.3f} "
        f"inference_wall_seconds="
        f"{inference_wall_seconds:.3f} "
        f"total_pdf_seconds={total_pdf_seconds:.3f}",
        flush=True,
    )

    return pdf_timing_record

def build_combined_markdown(
    pdf_paths,
    raw_pages_dir,
    combined_markdown_path,
):
    """
    Build one deterministic Markdown file containing the exact raw
    model output for every page of every PDF.

    The per-page raw output is not modified. HTML comments are used
    only as document and page boundaries.
    """
    combined_markdown_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = combined_markdown_path.with_suffix(
        ".md.tmp"
    )

    total_pages = 0

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as combined_file:
        for pdf_path in pdf_paths:
            document = fitz.open(str(pdf_path))

            try:
                page_count = document.page_count
            finally:
                document.close()

            for page_number in range(1, page_count + 1):
                raw_page_path = (
                    raw_pages_dir
                    / pdf_path.stem
                    / f"page_{page_number:04d}.md"
                )

                if not raw_page_path.is_file():
                    raise FileNotFoundError(
                        "Cannot build combined Markdown; "
                        f"raw page is missing: {raw_page_path}"
                    )

                raw_output = raw_page_path.read_text(
                    encoding="utf-8"
                )

                combined_file.write(
                    f"<!-- PDF: {pdf_path.name} | "
                    f"PAGE: {page_number} -->\n"
                )
                combined_file.write(raw_output)

                if not raw_output.endswith("\n"):
                    combined_file.write("\n")

                combined_file.write("\n")
                total_pages += 1

        combined_file.flush()
        os.fsync(combined_file.fileno())

    temporary_path.replace(combined_markdown_path)

    return total_pages

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run raw Unlimited-OCR inference on the "
            "eSciBench PDF dataset."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Path to the eSciBench dataset directory "
            "containing the PDF files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "benchmark/extractors/unlimited_ocr/raw_outputs"
        ),
    )

    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:10000",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--expected-pdf-count",
        type=int,
        default=101,
    )

    parser.add_argument(
        "--pdf-name",
        default=None,
        help="Run only one exact PDF filename.",
    )

    parser.add_argument(
        "--pdf-limit",
        type=int,
        default=None,
        help="Run only the first N sorted PDFs.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir}"
        )

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    check_server(args.server_url)

    pdf_paths = discover_dataset(
        data_dir,
        args.expected_pdf_count,
    )

    if args.pdf_name is not None:
        pdf_paths = [
            path
            for path in pdf_paths
            if path.name == args.pdf_name
        ]

        if not pdf_paths:
            raise FileNotFoundError(
                f"Requested PDF was not found: "
                f"{args.pdf_name}"
            )

    if args.pdf_limit is not None:
        if args.pdf_limit < 1:
            raise ValueError(
                "--pdf-limit must be at least 1"
            )

        pdf_paths = pdf_paths[: args.pdf_limit]

    raw_pages_dir = output_dir / "pages"
    timing_dir = output_dir / "timing"

    master_jsonl_path = (
        output_dir
        / "unlimited_ocr_raw_101.jsonl"
    )
    page_timing_path = (
        timing_dir
        / "page_timings.csv"
    )
    pdf_timing_path = (
        timing_dir
        / "pdf_timings.csv"
    )

    combined_markdown_path = (
        output_dir
        / "unlimited_ocr_raw_101.md"
    )



    run_summary_path = (
        output_dir
        / "run_summary.json"
    )

    raw_pages_dir.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)

    successful_pages = load_successful_pages(
        master_jsonl_path
    )

    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")

    if slurm_tmpdir:
        temporary_root = (
            Path(slurm_tmpdir)
            / "unlimited_ocr_rendered_pages"
        )
        temporary_root.mkdir(parents=True, exist_ok=True)
        remove_temporary_root = False
    else:
        temporary_root = Path(
            tempfile.mkdtemp(
                prefix="unlimited_ocr_rendered_pages_"
            )
        )
        remove_temporary_root = True

    corpus_start = time.perf_counter()
    pdf_records = []

    print(
        f"[START] pdf_count={len(pdf_paths)} "
        f"workers={args.workers} "
        f"data_dir={data_dir}",
        flush=True,
    )

    try:
        for pdf_index, pdf_path in enumerate(
            pdf_paths,
            start=1,
        ):
            print(
                f"[PDF {pdf_index}/{len(pdf_paths)}] "
                f"{pdf_path.name}",
                flush=True,
            )

            pdf_record = process_pdf(
                pdf_path=pdf_path,
                server_url=args.server_url,
                temporary_root=temporary_root,
                raw_pages_dir=raw_pages_dir,
                master_jsonl_path=master_jsonl_path,
                page_timing_path=page_timing_path,
                pdf_timing_path=pdf_timing_path,
                successful_pages=successful_pages,
                workers=args.workers,
            )

            pdf_records.append(pdf_record)
    finally:
        if remove_temporary_root:
            shutil.rmtree(
                temporary_root,
                ignore_errors=True,
            )

    corpus_seconds = time.perf_counter() - corpus_start

    total_pages = sum(
        record["page_count"]
        for record in pdf_records
    )
    successful_pages_count = sum(
        record["successful_pages"]
        for record in pdf_records
    )
    failed_pages_count = sum(
        record["failed_pages"]
        for record in pdf_records
    )
    combined_markdown_created = False
    combined_markdown_pages = 0

    full_dataset_run = (
        args.pdf_name is None
        and args.pdf_limit is None
        and args.expected_pdf_count == 101
        and len(pdf_paths) == 101
    )

    if full_dataset_run and failed_pages_count == 0:
        combined_markdown_pages = build_combined_markdown(
            pdf_paths=pdf_paths,
            raw_pages_dir=raw_pages_dir,
            combined_markdown_path=combined_markdown_path,
        )

        if combined_markdown_pages != total_pages:
            raise RuntimeError(
                "Combined Markdown page count does not "
                f"match the run: {combined_markdown_pages} "
                f"!= {total_pages}"
            )
        combined_markdown_created = True

    run_summary = {
        "model_name": MODEL_NAME,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "combined_raw_markdown": (
            str(combined_markdown_path)
            if combined_markdown_created
            else None
        ),
        "combined_raw_markdown_created": (
            combined_markdown_created
        ),
        "combined_raw_markdown_pages": (
            combined_markdown_pages
        ),
        "pdf_count": len(pdf_records),
        "page_count": total_pages,
        "successful_pages": successful_pages_count,
        "failed_pages": failed_pages_count,
        "corpus_seconds": round(corpus_seconds, 6),
        "average_seconds_per_pdf": (
            round(corpus_seconds / len(pdf_records), 6)
            if pdf_records
            else 0
        ),
        "average_seconds_per_page": (
            round(corpus_seconds / total_pages, 6)
            if total_pages
            else 0
        ),
        "workers": args.workers,
        "settings": {
            "prompt": PROMPT,
            "dpi": DPI,
            "image_mode": IMAGE_MODE,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "request_timeout_seconds": (
                REQUEST_TIMEOUT_SECONDS
            ),
            "max_retries": MAX_RETRIES,
        },
        "master_raw_output": str(
            master_jsonl_path
        ),
        "page_timing_csv": str(
            page_timing_path
        ),
        "pdf_timing_csv": str(
            pdf_timing_path
        ),
    }

    with run_summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[CORPUS COMPLETE] "
        f"pdfs={len(pdf_records)} "
        f"pages={total_pages} "
        f"success={successful_pages_count} "
        f"failed={failed_pages_count} "
        f"seconds={corpus_seconds:.3f}",
        flush=True,
    )
    print(
        f"[RAW OUTPUT] {master_jsonl_path}",
        flush=True,
    )
    print(
        f"[SUMMARY] {run_summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()