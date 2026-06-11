from src.evaluation_dataset.chunker import chunk_pdf_pages
from src.evaluation_dataset.document_loader import SourceDocument
from src.evaluation_dataset.pdf_loader import PdfPage


def test_chunk_pdf_pages_preserves_source_metadata():
    pages = [
        PdfPage(
            source_file="history.pdf",
            page_number=3,
            text="Tran Hung Dao led Vietnamese forces against Mongol invasions.",
        )
    ]

    chunks = chunk_pdf_pages(pages, chunk_size=40, chunk_overlap=10)

    assert len(chunks) >= 1
    assert chunks[0].metadata["source_file"] == "history.pdf"
    assert chunks[0].metadata["page_number"] == 3
    assert chunks[0].metadata["chunk_id"] == "history.pdf:p3:c0"
    assert "Tran Hung Dao" in chunks[0].page_content


def test_chunk_source_documents_preserves_markdown_metadata():
    documents = [
        SourceDocument(
            source_file="history.md",
            source_type="markdown",
            unit_label="document",
            text="# History\n\nTran Hung Dao led Vietnamese forces.",
            metadata={"source_file": "history.md", "source_type": "markdown"},
        )
    ]

    chunks = chunk_pdf_pages(documents, chunk_size=40, chunk_overlap=10)

    assert chunks[0].metadata["source_file"] == "history.md"
    assert chunks[0].metadata["source_type"] == "markdown"
    assert chunks[0].metadata["chunk_id"] == "history.md:document:c0"
