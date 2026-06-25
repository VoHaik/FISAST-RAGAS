# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python CLI pipeline for generating RAG evaluation datasets from PDF and Markdown knowledge-base files.

- `src/evaluation_dataset/`: application code. Key modules include `cli.py`, `config.py`, `document_loader.py`, `pdf_loader.py`, `chunker.py`, `generator.py`, `model_provider.py`, and `exporter.py`.
- `tests/evaluation_dataset/`: pytest coverage for config validation, loading, chunking, model provider behavior, and export logic.
- `docs/`: project documentation, implementation notes, usage guides, and sample knowledge-base assets under `docs/KB/`.
- `requirements-eval.txt`: runtime and test dependencies.
- `.env`: local configuration for model credentials and provider settings. Do not commit secrets.

Avoid committing generated caches such as `__pycache__/`, `.pytest_cache/`, or local virtual environments.

## Build, Test, and Development Commands

Install dependencies from the repository root:

```powershell
python -m pip install -r requirements-eval.txt
```

If `python` is not available on PATH in this shell, use the local virtual environment:

```powershell
& 'C:\Users\KHAI\AppData\Local\Programs\Python\Python312\python.exe' -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-eval.txt
```

Run the full test suite:

```powershell
python -m pytest tests
```

Verified command in this workspace:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/evaluation_dataset -q
```

Inspect CLI options:

```powershell
python -m src.evaluation_dataset.cli --help
```

Generate a small sample dataset:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir docs/KB `
  --output-path docs/KB/BachDang/rag_eval_dataset.sample.jsonl `
  --output-format jsonl `
  --testset-size 5
```

Use `--input-dir` for mixed `.pdf`, `.md`, and `.markdown` folders. Keep `--input-pdf-dir` only for legacy PDF-only commands.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax and keep modules focused on one responsibility. Follow existing style: 4-space indentation, type hints for public functions and configuration fields, `snake_case` for modules, functions, variables, and test names, and `PascalCase` for classes such as `EvaluationDatasetConfig`.

Prefer `pathlib.Path` for filesystem paths and raise clear `ValueError` or domain-specific exceptions for validation failures. Keep CLI options explicit and aligned with `EvaluationDatasetConfig`.

## Testing Guidelines

Tests use `pytest`. Add or update tests in `tests/evaluation_dataset/` for any behavior change. Name files `test_<module>.py` and tests `test_<behavior>()`. Use `monkeypatch` for environment variables and temporary paths for filesystem output.

Run `python -m pytest tests` before submitting changes. For CLI or model-provider changes, include at least one test that validates configuration precedence or error handling.

## RAGAS Provider Configuration

The pipeline supports these providers through `EvaluationDatasetConfig`:

- `openai`
- `openai-compatible`
- `ollama`

For the Fake LLM Playwright Proxy, use OpenAI-compatible settings in `.env`:

```env
RAGAS_PROVIDER=openai-compatible
RAGAS_LLM_BASE_URL=http://localhost:8000/v1
RAGAS_LLM_MODEL=fake-gpt-5
RAGAS_EMBEDDINGS_BASE_URL=http://localhost:8000/v1
RAGAS_EMBEDDINGS_MODEL=fake-embeddings
OPENAI_API_KEY=not-needed
RAGAS_TIMEOUT=600
RAGAS_TEMPERATURE=0.0
RAGAS_LLM_FORMAT=
RAGAS_MAX_WORKERS=1
RAGAS_RUN_TIMEOUT=600
RAGAS_MAX_RETRIES=2
```

Keep `RAGAS_MAX_WORKERS=1` for the Fake LLM Playwright Proxy because it drives one browser session and queues concurrent prompts. RAGAS default parallelism can leave many HTTP connections waiting, which looks like the source process is hanging.

Before running a fake-agent smoke test, verify the proxy is running:

```powershell
curl.exe http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

Then run:

```powershell
& '.\.venv\Scripts\python.exe' -m src.evaluation_dataset.cli generate `
  --input-dir docs/KB/BachDang `
  --output-path docs/KB/BachDang/rag_eval_dataset.fake-llm.sample.jsonl `
  --output-format jsonl `
  --testset-size 1 `
  --ragas-max-workers 1
```

`fake-embeddings` is only for pipeline smoke tests. Use real embeddings for quality evaluation.

For Ollama, keep JSON mode enabled to reduce RAGAS parser failures:

```env
RAGAS_PROVIDER=ollama
RAGAS_LLM_FORMAT=json
RAGAS_TEMPERATURE=0.0
RAGAS_NUM_PREDICT=2048
```

## Input Document Notes

Markdown OCR output is supported and preferred for scanned PDFs. If a folder contains a bad scanned PDF and a good `.md` OCR output, the loader uses the Markdown and skips the bad PDF.

If PyMuPDF only extracts header/footer text from a PDF, the CLI exits early with a clear extraction-quality error. Convert the source to Markdown or OCR the PDF before running RAGAS.

## Commit & Pull Request Guidelines

This directory does not currently expose Git history, so no local commit convention can be inferred. Use short, imperative commit messages such as `Add markdown document loader tests` or `Fix Ollama config defaults`.

Pull requests should include a concise summary, tests run, related issue or task links, and notes about any generated dataset artifacts. Include screenshots only when documentation or rendered output changes.

## Security & Configuration Tips

Store API keys in `.env` or the shell environment, never in source or docs. Document new environment variables in `docs/rag-evaluation-dataset-usage-guide.md`. Keep generated datasets small in PRs unless the task specifically requires full outputs.
