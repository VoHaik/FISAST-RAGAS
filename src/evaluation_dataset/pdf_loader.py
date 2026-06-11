from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class PdfPage:
    source_file: str
    page_number: int
    text: str


class PdfExtractionQualityError(ValueError):
    pass


def load_pdf_pages(pdf_path: Path) -> list[PdfPage]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = fitz.open(pdf_path)
    try:
        pages: list[PdfPage] = []
        for index, page in enumerate(document):
            text = page.get_text().strip()
            if text:
                pages.append(
                    PdfPage(
                        source_file=pdf_path.name,
                        page_number=index + 1,
                        text=text,
                    )
                )
        return pages
    finally:
        document.close()


def load_pdf_directory(input_pdf_dir: Path) -> list[PdfPage]:
    pdf_paths = sorted(input_pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {input_pdf_dir}")

    pages: list[PdfPage] = []
    for pdf_path in pdf_paths:
        pages.extend(load_pdf_pages(pdf_path))
    return pages


def validate_pdf_text_quality(
    pages: list[PdfPage],
    min_total_words: int = 100,
    min_average_words_per_page: int = 30,
) -> None:
    if not pages:
        raise PdfExtractionQualityError(
            "PDF extraction returned no text. The input may be scanned, image-only, "
            "encrypted, or unreadable by PyMuPDF."
        )

    total_words = sum(_word_count(page.text) for page in pages)
    average_words = total_words / len(pages)
    if total_words >= min_total_words and average_words >= min_average_words_per_page:
        return

    source_files = ", ".join(sorted({page.source_file for page in pages}))
    raise PdfExtractionQualityError(
        "PDF extraction produced too little extractable text for RAGAS generation. "
        f"Files: {source_files}. Pages with text: {len(pages)}. "
        f"Extracted words: {total_words}. Average words per page: {average_words:.1f}. "
        "This usually means the PDF is scanned/image-based or only exposes header/footer text. "
        "Run OCR first or provide a selectable-text PDF before generating the RAG evaluation dataset."
    )


def _word_count(text: str) -> int:
    return len([word for word in text.split() if word.strip()])
