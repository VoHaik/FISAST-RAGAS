# RAG Evaluation Dataset Implementation Summary

## Objective

Implemented an offline Python CLI pipeline that generates a RAG evaluation dataset from PDF files using RAGAS.

The dataset is normalized to the project schema:

```json
{
  "question": "Generated user question",
  "contexts": ["Reference source context"],
  "ground_truth": "Reference answer"
}
```

## Technical Decision

RAGAS is used as the first implementation because this pipeline is specific to RAG evaluation dataset generation. It supports RAG-oriented single-hop and multi-hop query generation over source documents.

DeepEval is not included in this first version. It can be added later if the project needs DeepEval `Golden` objects, CI quality gates, filtration, or custom evaluation metrics.

## Implemented Files

### Dependencies

- `requirements-eval.txt`
  - Contains RAGAS, LangChain, OpenAI, PyMuPDF, Pandas, Typer, dotenv, and Pytest dependencies.

### Source Code

- `src/evaluation_dataset/config.py`
  - Defines `EvaluationDatasetConfig`.
  - Validates output format, chunk settings, testset size, and query distribution ratios.

- `src/evaluation_dataset/pdf_loader.py`
  - Reads PDF files with PyMuPDF.
  - Extracts non-empty page text.
  - Preserves `source_file` and `page_number`.

- `src/evaluation_dataset/document_loader.py`
  - Loads mixed input folders containing `.pdf`, `.md`, and `.markdown`.
  - Converts PDF pages and Markdown files into common `SourceDocument` objects.
  - Skips scanned/header-only PDFs when valid Markdown OCR output is available in the same folder.

- `src/evaluation_dataset/chunker.py`
  - Splits PDF page text into chunks.
  - Splits Markdown text into chunks.
  - Adds metadata including `source_file`, `source_type`, `page_number` when available, `chunk_index`, and `chunk_id`.
  - Uses `RecursiveCharacterTextSplitter` when available.

- `src/evaluation_dataset/generator.py`
  - Wraps RAGAS testset generation.
  - Uses provider-agnostic RAGAS LLM and embedding wrappers.
  - Generates single-hop specific, multi-hop specific, and multi-hop abstract queries.

- `src/evaluation_dataset/model_provider.py`
  - Builds RAGAS-compatible model wrappers.
  - Supports `openai`, `openai-compatible`, and `ollama`.
  - Keeps OpenAI and Ollama imports out of `generator.py`.
  - Allows CLI options and `.env` values to switch model ecosystems.

- `src/evaluation_dataset/exporter.py`
  - Normalizes generated dataset columns.
  - Maps RAGAS columns:
    - `user_input` to `question`
    - `reference_contexts` to `contexts`
    - `reference` to `ground_truth`
  - Exports `.jsonl` or `.csv`.

- `src/evaluation_dataset/cli.py`
  - Provides the CLI command:

```powershell
python -m src.evaluation_dataset.cli generate
```

### Tests

- `tests/evaluation_dataset/test_config.py`
- `tests/evaluation_dataset/test_pdf_loader.py`
- `tests/evaluation_dataset/test_chunker.py`
- `tests/evaluation_dataset/test_exporter.py`

The tests cover config validation, PDF loading behavior, chunk metadata, RAGAS schema normalization, and JSONL export.

### Documentation

- `docs/rag-evaluation-dataset.md`
  - Main pipeline documentation.
- `docs/rag-evaluation-dataset-usage-guide.md`
  - Step-by-step usage guide.
- `docs/rag-evaluation-dataset-implementation-summary.md`
  - This implementation summary.

## Runtime Flow

1. Load environment variables from `.env`.
2. Read PDF or Markdown files from the input directory.
3. Extract source text and metadata.
4. Split extracted text into chunks.
5. Build model provider wrappers from CLI or `.env` config.
6. Generate RAGAS synthetic testset.
7. Normalize generated columns to project schema.
8. Export dataset to `.jsonl` or `.csv`.

## Dynamic Provider Architecture

Model construction is handled by:

```text
src/evaluation_dataset/model_provider.py
```

`generator.py` no longer imports OpenAI or Ollama directly. It receives RAGAS-compatible wrappers from `build_ragas_models(config)`.

Supported providers:

- `openai`
- `openai-compatible`
- `ollama`

Configuration priority:

1. CLI options
2. `.env`
3. source defaults in `EvaluationDatasetConfig`

## Verification Status

Verification was attempted with:

```powershell
python -m pytest tests/evaluation_dataset -v
python -m src.evaluation_dataset.cli --help
python -c "from src.evaluation_dataset.config import EvaluationDatasetConfig; from src.evaluation_dataset.cli import app; print('evaluation-dataset-import-ok')"
```

The commands could not run because `python` is not available on the current shell `PATH`.

Observed error:

```text
python : The term 'python' is not recognized as the name of a cmdlet, function, script file, or operable program.
```

Install Python or add it to `PATH`, then rerun the verification commands.
