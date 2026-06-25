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


class MockEmbeddings:
    model: str = "mock-model"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_chunk_pdf_pages_semantic_chunking_with_mock_embeddings():
    pages = [
        PdfPage(
            source_file="history.pdf",
            page_number=1,
            text="First sentence here. Second sentence starts now. Third sentence is final.",
        )
    ]
    mock_emb = MockEmbeddings()
    chunks = chunk_pdf_pages(
        pages,
        chunk_size=1000,
        chunk_overlap=150,
        embeddings=mock_emb,
        chunking_method="semantic"
    )
    assert len(chunks) >= 1
    assert "First sentence" in chunks[0].page_content


def test_chunk_pdf_pages_semantic_fallback_when_no_embeddings():
    pages = [
        PdfPage(
            source_file="history.pdf",
            page_number=1,
            text="Tran Hung Dao led Vietnamese forces.",
        )
    ]
    # Passing embeddings=None with semantic method should gracefully fall back to recursive
    chunks = chunk_pdf_pages(
        pages,
        chunk_size=40,
        chunk_overlap=10,
        embeddings=None,
        chunking_method="semantic"
    )
    assert len(chunks) >= 1
    assert chunks[0].metadata["source_file"] == "history.pdf"
