from dataclasses import dataclass
from typing import Any

try:
    from langchain_core.documents import Document
except ImportError:
    @dataclass
    class Document:
        page_content: str
        metadata: dict[str, Any]

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

from src.evaluation_dataset.document_loader import SourceDocument
from src.evaluation_dataset.pdf_loader import PdfPage


def chunk_pdf_pages(
    pages: list[PdfPage | SourceDocument],
    chunk_size: int,
    chunk_overlap: int,
    embeddings: Any = None,
    chunking_method: str = "semantic",
) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be 0 or greater and smaller than chunk_size")

    splitter = None
    if chunking_method == "semantic" and embeddings is not None:
        try:
            from langchain.embeddings import CacheBackedEmbeddings
            from langchain.storage import LocalFileStore
            from langchain_experimental.text_splitter import SemanticChunker

            store = LocalFileStore("./.cache/embeddings")
            model_name = getattr(embeddings, "model", "default")
            cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
                embeddings,
                store,
                namespace=model_name
            )
            splitter = SemanticChunker(
                cached_embeddings,
                breakpoint_threshold_type="percentile"
            )
        except ImportError:
            pass

    documents: list[Document] = []
    for page in pages:
        source_document = _as_source_document(page)
        if splitter is not None:
            split_texts = splitter.split_text(source_document.text)
        else:
            split_texts = _split_text(source_document.text, chunk_size, chunk_overlap)

        for chunk_index, chunk_text in enumerate(split_texts):
            metadata = dict(source_document.metadata)
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "chunk_id": (
                        f"{source_document.source_file}:"
                        f"{source_document.unit_label}:c{chunk_index}"
                    ),
                }
            )
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata,
                )
            )
    return documents


def _as_source_document(page: PdfPage | SourceDocument) -> SourceDocument:
    if isinstance(page, SourceDocument):
        return page
    return SourceDocument(
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


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)

    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        next_start = start + chunk_size - chunk_overlap
        if next_start <= start:
            break
        start = next_start
    return chunks
