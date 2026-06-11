import pytest

from src.evaluation_dataset.pdf_loader import (
    PdfExtractionQualityError,
    PdfPage,
    load_pdf_pages,
    validate_pdf_text_quality,
)


class FakePage:
    def __init__(self, text: str):
        self.text = text

    def get_text(self) -> str:
        return self.text


class FakeDocument:
    def __iter__(self):
        return iter([FakePage("Page one text"), FakePage("Page two text"), FakePage(" ")])

    def close(self) -> None:
        pass


def test_load_pdf_pages_returns_page_text_and_metadata(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr("src.evaluation_dataset.pdf_loader.fitz.open", lambda _: FakeDocument())

    pages = load_pdf_pages(pdf_path)

    assert pages == [
        PdfPage(source_file="sample.pdf", page_number=1, text="Page one text"),
        PdfPage(source_file="sample.pdf", page_number=2, text="Page two text"),
    ]


def test_validate_pdf_text_quality_rejects_scanned_or_header_only_pdf():
    pages = [
        PdfPage(
            source_file="scan.pdf",
            page_number=1,
            text="5/24/26 Scribd 20/731",
        ),
        PdfPage(
            source_file="scan.pdf",
            page_number=2,
            text="5/24/26 Scribd 21/731",
        ),
    ]

    with pytest.raises(PdfExtractionQualityError, match="too little extractable text"):
        validate_pdf_text_quality(pages, min_total_words=100, min_average_words_per_page=30)


def test_validate_pdf_text_quality_accepts_text_pdf():
    pages = [
        PdfPage(
            source_file="text.pdf",
            page_number=1,
            text=" ".join(["historical"] * 120),
        )
    ]

    validate_pdf_text_quality(pages, min_total_words=100, min_average_words_per_page=30)
