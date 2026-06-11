from src.evaluation_dataset.document_loader import (
    SourceDocument,
    load_input_directory,
    load_markdown_file,
)
from src.evaluation_dataset.pdf_loader import PdfPage


def test_load_markdown_file_returns_source_document(tmp_path):
    markdown_path = tmp_path / "bach_dang.md"
    markdown_path.write_text("# Bach Dang\n\nTran Hung Dao defeated invaders.", encoding="utf-8")

    documents = load_markdown_file(markdown_path)

    assert documents == [
        SourceDocument(
            source_file="bach_dang.md",
            source_type="markdown",
            unit_label="document",
            text="# Bach Dang\n\nTran Hung Dao defeated invaders.",
            metadata={"source_file": "bach_dang.md", "source_type": "markdown"},
        )
    ]


def test_load_input_directory_loads_markdown_files(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nContent A", encoding="utf-8")
    (tmp_path / "b.markdown").write_text("# B\n\nContent B", encoding="utf-8")

    documents = load_input_directory(tmp_path)

    assert [document.source_file for document in documents] == ["a.md", "b.markdown"]
    assert {document.source_type for document in documents} == {"markdown"}


def test_load_input_directory_uses_markdown_when_pdf_has_poor_extractable_text(monkeypatch, tmp_path):
    (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "ocr.md").write_text("# OCR\n\n" + " ".join(["history"] * 120), encoding="utf-8")

    monkeypatch.setattr(
        "src.evaluation_dataset.document_loader.load_pdf_pages",
        lambda _: [
            PdfPage(
                source_file="scan.pdf",
                page_number=1,
                text="Scribd 20/731",
            )
        ],
    )

    documents = load_input_directory(tmp_path)

    assert [document.source_file for document in documents] == ["ocr.md"]
