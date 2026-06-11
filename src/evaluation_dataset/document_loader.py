from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.evaluation_dataset.pdf_loader import (
    PdfExtractionQualityError,
    PdfPage,
    load_pdf_pages,
    validate_pdf_text_quality,
)


SUPPORTED_INPUT_EXTENSIONS = {".pdf", ".md", ".markdown"}


@dataclass(frozen=True)
class SourceDocument:
    source_file: str
    source_type: str
    unit_label: str
    text: str
    metadata: dict[str, Any]


def load_markdown_file(markdown_path: Path) -> list[SourceDocument]:
    text = markdown_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    return [
        SourceDocument(
            source_file=markdown_path.name,
            source_type="markdown",
            unit_label="document",
            text=text,
            metadata={
                "source_file": markdown_path.name,
                "source_type": "markdown",
            },
        )
    ]


def load_input_directory(input_dir: Path) -> list[SourceDocument]:
    source_paths = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
    ]
    if not source_paths:
        supported = ", ".join(sorted(SUPPORTED_INPUT_EXTENSIONS))
        raise ValueError(f"No supported input files found in {input_dir}. Supported: {supported}")

    documents: list[SourceDocument] = []
    extraction_errors: list[PdfExtractionQualityError] = []
    for source_path in source_paths:
        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            pages = load_pdf_pages(source_path)
            try:
                validate_pdf_text_quality(pages)
            except PdfExtractionQualityError as error:
                extraction_errors.append(error)
                continue
            documents.extend(_pdf_pages_to_source_documents(pages))
        elif suffix in {".md", ".markdown"}:
            documents.extend(load_markdown_file(source_path))

    if not documents:
        if extraction_errors:
            raise extraction_errors[0]
        raise ValueError(f"Supported input files in {input_dir} did not contain extractable text")
    return documents


def _pdf_pages_to_source_documents(pages: list[PdfPage]) -> list[SourceDocument]:
    return [
        SourceDocument(
            source_file=page.source_file,
            source_type="pdf",
            unit_label=f"p{page.page_number}",
            text=page.text,
            metadata={
                "source_file": page.source_file,
                "source_type": "pdf",
                "page_number": page.page_number,
            },
        )
        for page in pages
    ]
