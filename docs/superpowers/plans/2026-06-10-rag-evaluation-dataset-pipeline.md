# RAG Evaluation Dataset Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python pipeline that reads PDF documents and generates a RAG evaluation dataset with `question`, `contexts`, `ground_truth`, and metadata fields.

**Architecture:** The pipeline is split into focused modules: PDF loading, document preparation, testset generation, schema normalization, export, and CLI orchestration. RAGAS is the default generator because its current testset pipeline builds a knowledge graph and generates single-hop and multi-hop query scenarios; DeepEval can be added later behind the same generator interface when CI-first evaluation or richer synthetic goldens are required.

**Tech Stack:** Python 3.10+, PyMuPDF or LangChain PDF loaders, LangChain text splitters, RAGAS, OpenAI or local OpenAI-compatible LLM, Pandas, Typer CLI, Pytest.

---

## Scope

This plan creates an offline dataset-generation pipeline, not the RAG runtime itself.

The output dataset is used to evaluate a RAG system later with metrics such as faithfulness, answer relevancy, context precision, and context recall.

## Target Output Schema

Each generated row should contain:

```json
{
  "question": "Question generated from the PDF content",
  "contexts": ["Source chunk 1", "Source chunk 2"],
  "ground_truth": "Reference answer generated from source contexts",
  "question_type": "single_hop_specific | multi_hop_specific | multi_hop_abstract",
  "source_file": "document.pdf",
  "source_pages": [1, 2],
  "chunk_ids": ["document.pdf:p1:c0"],
  "generator": "ragas",
  "created_at": "2026-06-10T00:00:00Z"
}
```

## Proposed File Structure

- Create: `requirements-eval.txt`
  - Dependencies isolated from production runtime.
- Create: `src/evaluation_dataset/__init__.py`
  - Package marker.
- Create: `src/evaluation_dataset/config.py`
  - Typed configuration for model, chunking, generation size, and output paths.
- Create: `src/evaluation_dataset/pdf_loader.py`
  - Extract text and page metadata from PDF files.
- Create: `src/evaluation_dataset/chunker.py`
  - Convert page text into LangChain `Document` objects and optionally pre-chunk with stable chunk IDs.
- Create: `src/evaluation_dataset/generator.py`
  - RAGAS testset generation wrapper.
- Create: `src/evaluation_dataset/exporter.py`
  - Validate and export `.csv` / `.jsonl`.
- Create: `src/evaluation_dataset/cli.py`
  - Command-line entrypoint.
- Create: `tests/evaluation_dataset/test_pdf_loader.py`
  - PDF extraction behavior.
- Create: `tests/evaluation_dataset/test_chunker.py`
  - Chunk size, overlap, and metadata behavior.
- Create: `tests/evaluation_dataset/test_exporter.py`
  - Schema validation and output formatting.
- Create: `docs/rag-evaluation-dataset.md`
  - Usage guide and operational notes.

---

## Task 1: Add Evaluation Dependencies

**Files:**
- Create: `requirements-eval.txt`

- [ ] **Step 1: Create dependency file**

```txt
ragas>=0.4.0
datasets>=2.19.0
pandas>=2.2.0
langchain>=0.2.0
langchain-core>=0.2.0
langchain-community>=0.2.0
langchain-openai>=0.1.0
langchain-text-splitters>=0.2.0
openai>=1.0.0
pymupdf>=1.24.0
typer>=0.12.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Install dependencies locally**

Run:

```bash
python -m pip install -r requirements-eval.txt
```

Expected:

```text
Successfully installed ...
```

- [ ] **Step 3: Commit**

```bash
git add requirements-eval.txt
git commit -m "chore: add rag evaluation dataset dependencies"
```

---

## Task 2: Define Pipeline Configuration

**Files:**
- Create: `src/evaluation_dataset/__init__.py`
- Create: `src/evaluation_dataset/config.py`
- Test: no test required for static defaults unless validation is added

- [ ] **Step 1: Create package marker**

```python
# src/evaluation_dataset/__init__.py
```

- [ ] **Step 2: Add config dataclass**

```python
# src/evaluation_dataset/config.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationDatasetConfig:
    input_pdf_dir: Path
    output_path: Path
    output_format: str = "jsonl"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    testset_size: int = 100
    single_hop_specific_ratio: float = 0.5
    multi_hop_specific_ratio: float = 0.25
    multi_hop_abstract_ratio: float = 0.25
    generator_model: str = "gpt-4o"
    embeddings_model: str = "text-embedding-3-small"

    def validate(self) -> None:
        if self.output_format not in {"jsonl", "csv"}:
            raise ValueError("output_format must be either 'jsonl' or 'csv'")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be 0 or greater")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        total = (
            self.single_hop_specific_ratio
            + self.multi_hop_specific_ratio
            + self.multi_hop_abstract_ratio
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError("question type ratios must sum to 1.0")
```

- [ ] **Step 3: Verify import**

Run:

```bash
python -c "from src.evaluation_dataset.config import EvaluationDatasetConfig; print('config-ok')"
```

Expected:

```text
config-ok
```

- [ ] **Step 4: Commit**

```bash
git add src/evaluation_dataset/__init__.py src/evaluation_dataset/config.py
git commit -m "feat: add evaluation dataset config"
```

---

## Task 3: Extract Text From PDFs

**Files:**
- Create: `src/evaluation_dataset/pdf_loader.py`
- Test: `tests/evaluation_dataset/test_pdf_loader.py`

- [ ] **Step 1: Write loader test with monkeypatched document**

```python
# tests/evaluation_dataset/test_pdf_loader.py
from pathlib import Path

from src.evaluation_dataset.pdf_loader import PdfPage, load_pdf_pages


class FakePage:
    def __init__(self, text: str):
        self.text = text

    def get_text(self) -> str:
        return self.text


class FakeDocument:
    def __iter__(self):
        return iter([FakePage("Page one text"), FakePage("Page two text")])

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
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
pytest tests/evaluation_dataset/test_pdf_loader.py -v
```

Expected:

```text
ModuleNotFoundError or ImportError
```

- [ ] **Step 3: Implement PDF loader**

```python
# src/evaluation_dataset/pdf_loader.py
from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True)
class PdfPage:
    source_file: str
    page_number: int
    text: str


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
```

- [ ] **Step 4: Run test and confirm pass**

Run:

```bash
pytest tests/evaluation_dataset/test_pdf_loader.py -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/evaluation_dataset/pdf_loader.py tests/evaluation_dataset/test_pdf_loader.py
git commit -m "feat: extract text from pdf documents"
```

---

## Task 4: Chunk Extracted Text With Metadata

**Files:**
- Create: `src/evaluation_dataset/chunker.py`
- Test: `tests/evaluation_dataset/test_chunker.py`

- [ ] **Step 1: Write chunking test**

```python
# tests/evaluation_dataset/test_chunker.py
from src.evaluation_dataset.chunker import chunk_pdf_pages
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
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
pytest tests/evaluation_dataset/test_chunker.py -v
```

Expected:

```text
ModuleNotFoundError or ImportError
```

- [ ] **Step 3: Implement chunker**

```python
# src/evaluation_dataset/chunker.py
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.evaluation_dataset.pdf_loader import PdfPage


def chunk_pdf_pages(
    pages: list[PdfPage],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for page in pages:
        split_texts = splitter.split_text(page.text)
        for chunk_index, chunk_text in enumerate(split_texts):
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source_file": page.source_file,
                        "page_number": page.page_number,
                        "chunk_index": chunk_index,
                        "chunk_id": f"{page.source_file}:p{page.page_number}:c{chunk_index}",
                    },
                )
            )
    return documents
```

- [ ] **Step 4: Run test and confirm pass**

Run:

```bash
pytest tests/evaluation_dataset/test_chunker.py -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/evaluation_dataset/chunker.py tests/evaluation_dataset/test_chunker.py
git commit -m "feat: chunk pdf text for rag evaluation generation"
```

---

## Task 5: Generate Synthetic Testset With RAGAS

**Files:**
- Create: `src/evaluation_dataset/generator.py`

- [ ] **Step 1: Implement RAGAS generator wrapper**

```python
# src/evaluation_dataset/generator.py
import pandas as pd
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution

from src.evaluation_dataset.config import EvaluationDatasetConfig


def generate_ragas_testset(documents, config: EvaluationDatasetConfig) -> pd.DataFrame:
    config.validate()

    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=config.generator_model, temperature=0.2))
    embeddings = OpenAIEmbeddings()

    generator = TestsetGenerator(llm=generator_llm, embedding_model=embeddings)
    query_distribution = default_query_distribution(generator_llm)

    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=config.testset_size,
        query_distribution=query_distribution,
    )

    return testset.to_pandas()
```

- [ ] **Step 2: Run import check**

Run:

```bash
python -c "from src.evaluation_dataset.generator import generate_ragas_testset; print('generator-ok')"
```

Expected:

```text
generator-ok
```

- [ ] **Step 3: Commit**

```bash
git add src/evaluation_dataset/generator.py
git commit -m "feat: generate rag evaluation testsets with ragas"
```

---

## Task 6: Validate And Export Dataset

**Files:**
- Create: `src/evaluation_dataset/exporter.py`
- Test: `tests/evaluation_dataset/test_exporter.py`

- [ ] **Step 1: Write exporter tests**

```python
# tests/evaluation_dataset/test_exporter.py
import json

import pandas as pd

from src.evaluation_dataset.exporter import export_dataset, normalize_dataset


def test_normalize_dataset_requires_core_columns():
    frame = pd.DataFrame([{"question": "Q?", "ground_truth": "A"}])

    try:
        normalize_dataset(frame)
    except ValueError as error:
        assert "contexts" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_export_dataset_writes_jsonl(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "question": "Who led the resistance?",
                "contexts": ["Tran Hung Dao led the resistance."],
                "ground_truth": "Tran Hung Dao.",
            }
        ]
    )
    output_path = tmp_path / "dataset.jsonl"

    export_dataset(frame, output_path, "jsonl")

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["question"] == "Who led the resistance?"
    assert rows[0]["ground_truth"] == "Tran Hung Dao."
```

- [ ] **Step 2: Run test and confirm failure**

Run:

```bash
pytest tests/evaluation_dataset/test_exporter.py -v
```

Expected:

```text
ModuleNotFoundError or ImportError
```

- [ ] **Step 3: Implement exporter**

```python
# src/evaluation_dataset/exporter.py
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"question", "contexts", "ground_truth"}
COLUMN_ALIASES = {
    "user_input": "question",
    "reference_contexts": "contexts",
    "reference": "ground_truth",
}


def normalize_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.rename(columns=_available_aliases(frame)).copy()

    missing = REQUIRED_COLUMNS.difference(normalized.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Generated dataset is missing required columns: {missing_columns}")

    for column in REQUIRED_COLUMNS:
        normalized[column] = normalized[column].apply(_require_non_empty)
    return normalized


def export_dataset(frame: pd.DataFrame, output_path: Path, output_format: str) -> None:
    normalized = normalize_dataset(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "jsonl":
        normalized.to_json(output_path, orient="records", lines=True, force_ascii=False)
        return

    if output_format == "csv":
        normalized.to_csv(output_path, index=False, encoding="utf-8")
        return

    raise ValueError("output_format must be either 'jsonl' or 'csv'")


def _available_aliases(frame: pd.DataFrame) -> dict[str, str]:
    return {
        source: target
        for source, target in COLUMN_ALIASES.items()
        if source in frame.columns and target not in frame.columns
    }


def _require_non_empty(value):
    if value is None:
        raise ValueError("Dataset contains null values in required columns")
    if isinstance(value, str) and not value.strip():
        raise ValueError("Dataset contains blank strings in required columns")
    if isinstance(value, list) and not value:
        raise ValueError("Dataset contains empty context lists")
    return value
```

- [ ] **Step 4: Run test and confirm pass**

Run:

```bash
pytest tests/evaluation_dataset/test_exporter.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/evaluation_dataset/exporter.py tests/evaluation_dataset/test_exporter.py
git commit -m "feat: validate and export rag evaluation datasets"
```

---

## Task 7: Add CLI Orchestration

**Files:**
- Create: `src/evaluation_dataset/cli.py`

- [ ] **Step 1: Implement CLI**

```python
# src/evaluation_dataset/cli.py
from pathlib import Path

import typer
from dotenv import load_dotenv

from src.evaluation_dataset.chunker import chunk_pdf_pages
from src.evaluation_dataset.config import EvaluationDatasetConfig
from src.evaluation_dataset.exporter import export_dataset
from src.evaluation_dataset.generator import generate_ragas_testset
from src.evaluation_dataset.pdf_loader import load_pdf_directory


app = typer.Typer()


@app.command()
def generate(
    input_pdf_dir: Path = typer.Option(..., exists=True, file_okay=False),
    output_path: Path = typer.Option(...),
    output_format: str = typer.Option("jsonl"),
    chunk_size: int = typer.Option(1000),
    chunk_overlap: int = typer.Option(150),
    testset_size: int = typer.Option(100),
    generator_model: str = typer.Option("gpt-4o-mini"),
    embeddings_model: str = typer.Option("text-embedding-3-small"),
) -> None:
    load_dotenv()
    config = EvaluationDatasetConfig(
        input_pdf_dir=input_pdf_dir,
        output_path=output_path,
        output_format=output_format,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        testset_size=testset_size,
        generator_model=generator_model,
        embeddings_model=embeddings_model,
    )
    config.validate()

    pages = load_pdf_directory(config.input_pdf_dir)
    documents = chunk_pdf_pages(
        pages,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    frame = generate_ragas_testset(documents, config)
    export_dataset(frame, config.output_path, config.output_format)

    typer.echo(f"Generated {len(frame)} rows at {config.output_path}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Run CLI help**

Run:

```bash
python -m src.evaluation_dataset.cli --help
```

Expected:

```text
Usage:
```

- [ ] **Step 3: Run smoke generation with real PDFs**

Prerequisite:

```bash
set OPENAI_API_KEY=your_api_key
```

Run:

```bash
python -m src.evaluation_dataset.cli generate --input-pdf-dir data/pdfs --output-path data/eval/rag_eval_dataset.jsonl --output-format jsonl --testset-size 10
```

Expected:

```text
Generated 10 rows at data/eval/rag_eval_dataset.jsonl
```

- [ ] **Step 4: Commit**

```bash
git add src/evaluation_dataset/cli.py
git commit -m "feat: add rag evaluation dataset cli"
```

---

## Task 8: Add Usage Documentation

**Files:**
- Create: `docs/rag-evaluation-dataset.md`

- [ ] **Step 1: Write documentation**

```markdown
# RAG Evaluation Dataset Pipeline

## Purpose

This pipeline creates an evaluation dataset from PDF source documents. The dataset is used to measure RAG quality with questions, source contexts, and ground-truth answers.

## Input

Place PDF files in:

```text
data/pdfs/
```

## Output

The recommended output is:

```text
data/eval/rag_eval_dataset.jsonl
```

Each row contains:

- `question`
- `contexts`
- `ground_truth`
- optional metadata such as question type, source file, page number, and chunk ID

## Generate Dataset

```bash
python -m pip install -r requirements-eval.txt
set OPENAI_API_KEY=your_api_key
python -m src.evaluation_dataset.cli generate --input-pdf-dir data/pdfs --output-path data/eval/rag_eval_dataset.jsonl --output-format jsonl --testset-size 100
```

## Recommended Generation Settings

Use these defaults first:

- `chunk_size`: 1000
- `chunk_overlap`: 150
- `testset_size`: 100
- `single_hop_specific_ratio`: 0.5
- `multi_hop_specific_ratio`: 0.25
- `multi_hop_abstract_ratio`: 0.25

Increase `testset_size` only after manually reviewing the first 20 generated rows.

## Quality Review Checklist

Before using the dataset for evaluation:

- Remove duplicate or near-duplicate questions.
- Remove questions whose answer is not supported by the provided contexts.
- Remove vague questions that require external knowledge.
- Keep a balanced mix of simple, reasoning, and multi-context questions.
- Keep source metadata so failed RAG examples can be traced back to the PDF.

## When To Use DeepEval Instead

Use RAGAS first when the main requirement is RAG-specific single-hop and multi-hop dataset generation over source documents. Use DeepEval first if the project already standardizes on DeepEval goldens, CI gates, filtration, or custom evaluation metrics.
```

- [ ] **Step 2: Commit**

```bash
git add docs/rag-evaluation-dataset.md
git commit -m "docs: document rag evaluation dataset pipeline"
```

---

## Task 9: Final Verification

**Files:**
- Read: all files created in previous tasks

- [ ] **Step 1: Run unit tests**

```bash
pytest tests/evaluation_dataset -v
```

Expected:

```text
4 passed
```

- [ ] **Step 2: Run import checks**

```bash
python -c "from src.evaluation_dataset.config import EvaluationDatasetConfig; from src.evaluation_dataset.cli import app; print('evaluation-dataset-import-ok')"
```

Expected:

```text
evaluation-dataset-import-ok
```

- [ ] **Step 3: Run one real smoke generation**

```bash
python -m src.evaluation_dataset.cli generate --input-pdf-dir data/pdfs --output-path data/eval/rag_eval_dataset.sample.jsonl --output-format jsonl --testset-size 5
```

Expected:

```text
Generated 5 rows at data/eval/rag_eval_dataset.sample.jsonl
```

- [ ] **Step 4: Inspect generated sample**

```bash
python -c "import pandas as pd; df = pd.read_json('data/eval/rag_eval_dataset.sample.jsonl', lines=True); print(df[['question', 'ground_truth']].head().to_string())"
```

Expected:

```text
question
ground_truth
```

- [ ] **Step 5: Commit final verification notes if needed**

```bash
git status --short
```

Expected:

```text
No uncommitted changes after all planned commits
```

---

## Operational Notes

Use RAGAS first if the target is RAG-specific query generation over a knowledge graph, especially for single-hop and multi-hop retrieval scenarios. Use DeepEval first if the team already wants `EvaluationDataset` / `Golden` objects, evolution controls, built-in filtration, and CI evaluation gates. Both can generate synthetic data from documents; RAGAS is not automatically "best" in every project.

Start with 20 generated rows and review manually before scaling to 100 or more. Synthetic datasets can contain unsupported answers, ambiguous questions, or questions requiring external knowledge, so human review is part of the pipeline.

For Vietnamese PDFs, test the generated questions in Vietnamese and preserve UTF-8 output. If the PDF extraction quality is poor, add OCR later as a separate preprocessing task.

## Self-Review

Spec coverage:

- PDF text extraction is covered in Task 3.
- Chunking with `RecursiveCharacterTextSplitter` is covered in Task 4.
- Current RAGAS `TestsetGenerator` configuration is covered in Task 5 using `LangchainLLMWrapper`, `OpenAIEmbeddings`, `testset_size`, and `query_distribution`.
- CSV/JSONL export with `question`, `contexts`, and `ground_truth` is covered in Task 6.
- CLI execution and smoke validation are covered in Tasks 7 and 9.

Placeholder scan:

- No `TBD`, `TODO`, or unspecified implementation steps remain.

Type consistency:

- `EvaluationDatasetConfig`, `PdfPage`, `chunk_pdf_pages`, `generate_ragas_testset`, and `export_dataset` names are consistent across tasks.
