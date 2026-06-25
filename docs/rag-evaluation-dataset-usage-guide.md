# RAG Evaluation Dataset Usage Guide

## What This Runs

This is an offline CLI pipeline. It does not start a web server.

It reads source documents, splits them into text chunks, asks a configured LLM provider to generate RAGAS test rows, and exports a dataset.

Supported input files:

```text
*.pdf
*.md
*.markdown
```

Typical input folders:

```text
docs/KB/BachDang/ # included Markdown/PDF sample folder
data/pdfs/        # your own local PDFs or Markdown files
```

Typical output:

```text
data/eval/rag_eval_dataset.jsonl
```

Use Markdown when a PDF is scanned or image-based. The pipeline can only use text that PyMuPDF can extract from a PDF.

`--input-dir` reads only files directly inside the selected folder. It does not recursively scan subfolders.

## Quick Start

From the project root, install dependencies and check the CLI:

```powershell
python -m pip install -r requirements-eval.txt
python -m src.evaluation_dataset.cli --help
```

Configure a model provider before generation. For OpenAI, set:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Or use an Ollama/self-hosted `.env` as shown in the provider examples below.

Then generate a 5-row sample:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir docs/KB/BachDang `
  --output-path data/eval/rag_eval_dataset.sample.jsonl `
  --output-format jsonl `
  --testset-size 5
```

Expected output:

```text
Generated 5 rows at data/eval/rag_eval_dataset.sample.jsonl
```

If this works, increase `--testset-size` for the full run.

## Change The Main Parameters

Most users only need to change these flags:

| What you want to change | CLI flag | Example |
| --- | --- | --- |
| Input folder | `--input-dir` | `--input-dir docs/KB/BachDang` |
| Output file | `--output-path` | `--output-path data/eval/bach_dang.jsonl` |
| Output format | `--output-format` | `--output-format csv` |
| Number of generated rows | `--testset-size` | `--testset-size 20` |
| Text chunk size | `--chunk-size` | `--chunk-size 1500` |
| Repeated text between chunks | `--chunk-overlap` | `--chunk-overlap 250` |
| Model provider | `--provider` | `--provider ollama` |
| Generator model | `--generator-model` | `--generator-model gpt-4o` |
| Embedding model | `--embeddings-model` | `--embeddings-model text-embedding-3-small` |

`--chunk-size` is the main "text size" control. It is measured in characters, not words or tokens.

- Use a smaller value such as `700` when chunks are too broad or questions become vague.
- Use the default `1000` for general use.
- Use a larger value such as `1500` or `2000` when answers need more surrounding context.
- Keep `--chunk-overlap` smaller than `--chunk-size`. A common overlap is 10-20% of the chunk size.

Example with a custom input folder and larger text chunks:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir docs/KB/BachDang `
  --output-path data/eval/bach_dang_large_chunks.jsonl `
  --testset-size 20 `
  --chunk-size 1500 `
  --chunk-overlap 250
```

Configuration priority is:

1. CLI options in the command
2. `.env` values
3. source defaults in `src/evaluation_dataset/config.py`

## 1. Install Python

Use Python 3.10 or newer.

Check installation:

```powershell
python --version
```

If PowerShell cannot find `python`, install Python and enable the option to add Python to `PATH`.

## 2. Install Dependencies

From the project root:

```powershell
python -m pip install -r requirements-eval.txt
```

If `python` is not recognized in this shell, use the full Python path once to create a local virtual environment:

```powershell
& 'C:\Path\To\Python312\python.exe' -m venv .venv
```

Replace `C:\Path\To\Python312\python.exe` with the Python executable installed on your machine. If you do not know the path, reinstall Python 3.10+ and enable "Add Python to PATH".

Then install dependencies through the virtual environment:

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-eval.txt
```

Use this interpreter for all later commands:

```powershell
& '.\.venv\Scripts\python.exe' -m src.evaluation_dataset.cli --help
```

## 3. Add Model API Key

For OpenAI, set the key in the current PowerShell session:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Or create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

## 4. Add Input Files

Use a folder that directly contains `.pdf`, `.md`, or `.markdown` files, such as:

```text
docs/KB/BachDang/
docs/KB/ChiLang/
```

Or create your own input folder:

```powershell
New-Item -ItemType Directory -Path data/pdfs -Force
```

Place source files in that folder:

```text
data/pdfs/
data/pdfs/my_source.pdf
data/pdfs/my_ocr_text.md
```

Markdown is also supported. If you use a chatbot or OCR tool to convert scanned PDFs into Markdown, place the `.md` files in the same input folder:

```text
data/pdfs/I_tran_bach_dang.md
```

Use Markdown when the original PDF is scanned/image-based or only exposes header/footer text.

The command uses whichever folder you pass to `--input-dir`. For example:

```powershell
--input-dir docs/KB/BachDang
--input-dir data/pdfs
```

If your files are split across many subfolders, run the command once per folder or copy the files into one input folder first.

## 5. Run A Small Sample First

Generate 5 rows first:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir data/pdfs `
  --output-path data/eval/rag_eval_dataset.sample.jsonl `
  --output-format jsonl `
  --testset-size 5
```

Expected output:

```text
Generated 5 rows at data/eval/rag_eval_dataset.sample.jsonl
```

## 6. Generate The Full Dataset

After manually checking the sample:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir data/pdfs `
  --output-path data/eval/rag_eval_dataset.jsonl `
  --output-format jsonl `
  --testset-size 100
```

## 7. Configuration Location

There are two places where configuration is defined.

### Source Defaults

Default values are defined in:

```text
src/evaluation_dataset/config.py
```

The main config object is:

```python
EvaluationDatasetConfig
```

It contains defaults for:

- input and output paths
- output format
- chunk size
- chunk overlap
- testset size
- query distribution ratios
- generator model
- embeddings model

### CLI Overrides

Runtime values are passed through:

```text
src/evaluation_dataset/cli.py
```

The CLI command creates an `EvaluationDatasetConfig` object from command-line options. Any option passed in the command overrides the default value.

Example:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir data/pdfs `
  --output-path data/eval/custom_dataset.jsonl `
  --testset-size 50 `
  --chunk-size 1200 `
  --chunk-overlap 200 `
  --generator-model gpt-4o
```

For normal usage, prefer changing parameters through CLI options instead of editing source code.

## 8. Configuration Parameters

### Dynamic Model Provider Configuration

The pipeline supports these providers:

- `openai`
- `openai-compatible`
- `ollama`

Configuration priority:

1. CLI options
2. `.env`
3. source defaults in `EvaluationDatasetConfig`

Use CLI options for one-off runs. Use `.env` when you want a stable local setup.

#### Ollama / Self-Hosted Example

Use this when your model is served by Ollama directly or through an ngrok URL.

```env
RAGAS_PROVIDER=ollama
RAGAS_LLM_BASE_URL=https://macaroni-hunting-mutation.ngrok-free.dev
RAGAS_LLM_MODEL=qwen2.5:14b
RAGAS_EMBEDDINGS_BASE_URL=https://macaroni-hunting-mutation.ngrok-free.dev
RAGAS_EMBEDDINGS_MODEL=nomic-embed-text
RAGAS_TIMEOUT=600
RAGAS_NUM_PREDICT=2048
RAGAS_TEMPERATURE=0.0
RAGAS_LLM_FORMAT=json 
```

Run:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir docs/KB `
  --output-path data/eval/rag_eval_dataset.jsonl `
  --testset-size 20
```

Ollama does not require `OPENAI_API_KEY` unless your proxy, gateway, or ngrok layer enforces authentication.

#### OpenAI Example

```env
RAGAS_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
RAGAS_LLM_MODEL=gpt-4o
RAGAS_EMBEDDINGS_MODEL=text-embedding-3-small
```

#### OpenAI-Compatible Gateway Example

Use this for local gateways that expose an OpenAI-compatible `/v1` API.

```env
RAGAS_PROVIDER=openai-compatible
RAGAS_LLM_BASE_URL=http://localhost:8000/v1
RAGAS_LLM_MODEL=qwen2.5:14b
RAGAS_EMBEDDINGS_BASE_URL=http://localhost:8000/v1
RAGAS_EMBEDDINGS_MODEL=nomic-embed-text
OPENAI_API_KEY=not-needed-or-server-token
```

Some OpenAI-compatible servers require a token. Others accept any dummy value.

#### Fake LLM Playwright Proxy Example

Use this when calling the local Fake LLM Playwright Proxy described in `docs/integration/ragas-and-api-client-guide.md`.

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

Keep `RAGAS_MAX_WORKERS=1` for this proxy. The proxy drives one browser/ChatGPT session and effectively handles one generation at a time. Higher RAGAS concurrency can leave many HTTP connections open while they wait behind the active browser request, which looks like the RAGAS source is hanging and the proxy stopped receiving new requests.

Before running RAGAS, verify the proxy is up:

```powershell
curl.exe http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

Then run a small smoke test:

```powershell
& '.\.venv\Scripts\python.exe' -m src.evaluation_dataset.cli generate `
  --input-dir docs/KB/BachDang `
  --output-path docs/KB/BachDang/rag_eval_dataset.fake-llm.sample.jsonl `
  --output-format jsonl `
  --testset-size 1 `
  --ragas-max-workers 1
```

`fake-embeddings` is only suitable for pipeline smoke tests. Use real embeddings for quality evaluation.

You can tune chunking:

```powershell
--chunk-size 1000
--chunk-overlap 150
```

You can tune query distribution:

```powershell
--single-hop-specific-ratio 0.5
--multi-hop-specific-ratio 0.25
--multi-hop-abstract-ratio 0.25
```

The three ratio values must sum to `1.0`.

You can choose models:

```powershell
--generator-model gpt-4o
--embeddings-model text-embedding-3-small
```

### Parameter Reference

| Parameter | Default | Purpose | When to change |
| --- | --- | --- | --- |
| `--input-dir` | required | Folder containing `.pdf`, `.md`, or `.markdown` files | Change per dataset source |
| `--input-pdf-dir` | optional | Legacy PDF-only input folder option | Prefer `--input-dir` for new usage |
| `--output-path` | required | Output `.jsonl` or `.csv` path | Change per generated dataset |
| `--output-format` | `jsonl` | Export format | Use `csv` for spreadsheet review |
| `--language` | `vietnamese` | Target language for generated test cases | Change to `english` or other languages |
| `--adapt-prompts` | `false` | Enable automatic RAGAS prompt adaptation to the target language | Set to `true` to adapt prompts (requires strong LLM) |
| `--chunk-size` | `1000` | Max characters per chunk | Increase if answers need more context |
| `--chunk-overlap` | `150` | Shared characters between chunks | Increase if context is split awkwardly |
| `--testset-size` | `100` | Number of generated rows | Start with `5` or `20`, then scale |
| `--single-hop-specific-ratio` | `0.5` | Direct questions from one context | Increase for factual retrieval tests |
| `--multi-hop-specific-ratio` | `0.25` | Questions requiring multiple contexts | Increase for retrieval reasoning tests |
| `--multi-hop-abstract-ratio` | `0.25` | Broader multi-context questions | Increase for synthesis tests |
| `--provider` | `.env` or `openai` | Model provider ecosystem | Use `ollama` for self-hosted Ollama |
| `--llm-base-url` | `.env` or empty | LLM endpoint URL | Required for Ollama/ngrok and compatible gateways |
| `--embeddings-base-url` | `.env` or LLM URL | Embedding endpoint URL | Set when embeddings use a different endpoint |
| `--generator-model` | `gpt-4o` | LLM used to generate dataset | Use stronger model for better quality |
| `--embeddings-model` | `text-embedding-3-small` | Embeddings used by RAGAS | Change to match model/provider strategy |
| `--temperature` | `0.2` | Generation randomness | Lower for more deterministic output |
| `--timeout` | `600` | Request timeout in seconds | Increase for slower local models |
| `--num-predict` | `512`, `2048` for Ollama from `.env` defaults | Ollama generation token budget | Increase for longer generated JSON outputs |
| `--llm-format` | `json` for Ollama | Ollama structured output mode | Keep `json` to reduce RAGAS parser failures |
| `--ragas-max-workers` | `16` (or `1`/`2` via env) | Number of parallel worker threads for RAGAS generation | Set to `1` or `2` for local/remote Ollama or rate-limited endpoints to prevent gateway timeouts |
| `--ragas-run-timeout` | `600` | Timeout in seconds for a single Ragas generation run | Increase for slower local LLM responses |
| `--ragas-max-retries` | `2` | Max number of retries for Ragas API requests | Increase if network/rate-limit issues are common |

The three query ratio values must sum to `1.0`.

## 9. Common Configuration Recipes

### Fast Smoke Test

Use this before spending money on a larger generation run:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir data/pdfs `
  --output-path data/eval/smoke.jsonl `
  --testset-size 5
```

### Larger Evaluation Dataset

Use this after manually reviewing the smoke output:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir data/pdfs `
  --output-path data/eval/rag_eval_dataset.jsonl `
  --testset-size 100
```

### More Multi-Hop Questions

Use this when you want to stress retrieval across multiple chunks:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir data/pdfs `
  --output-path data/eval/rag_eval_multihop.jsonl `
  --single-hop-specific-ratio 0.3 `
  --multi-hop-specific-ratio 0.4 `
  --multi-hop-abstract-ratio 0.3
```

### Larger Context Chunks

Use this when generated answers lack enough local context:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir data/pdfs `
  --output-path data/eval/rag_eval_large_chunks.jsonl `
  --chunk-size 1500 `
  --chunk-overlap 250
```

## 10. Environment Configuration

Secrets should be placed in `.env`, not hardcoded in source code.

Project root `.env`:

```env
RAGAS_PROVIDER=ollama
RAGAS_LLM_BASE_URL=https://macaroni-hunting-mutation.ngrok-free.dev
RAGAS_LLM_MODEL=qwen2.5:14b
RAGAS_EMBEDDINGS_BASE_URL=https://macaroni-hunting-mutation.ngrok-free.dev
RAGAS_EMBEDDINGS_MODEL=nomic-embed-text
RAGAS_NUM_PREDICT=2048
RAGAS_TEMPERATURE=0.0
RAGAS_LLM_FORMAT=json
RAGAS_LANGUAGE=vietnamese
RAGAS_ADAPT_PROMPTS=false
```

The CLI calls `load_dotenv()` before creating the RAGAS generator, so provider settings are available through `os.getenv`.

## 11. Output Schema

The exported file contains normalized columns:

```json
{
  "question": "Generated question",
  "contexts": ["Context used to create the answer"],
  "ground_truth": "Reference answer"
}
```

Additional metadata from RAGAS may also be present depending on the generated testset.

## 12. Validate The Setup

Run tests:

```powershell
python -m pytest tests/evaluation_dataset -v
```

Or with the local virtual environment:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/evaluation_dataset -v
```

Run CLI help:

```powershell
python -m src.evaluation_dataset.cli --help
```

Run import check:

```powershell
python -c "from src.evaluation_dataset.config import EvaluationDatasetConfig; from src.evaluation_dataset.cli import app; print('evaluation-dataset-import-ok')"
```

## 13. Manual Quality Review

Before using the generated dataset as a benchmark:

- Remove duplicate questions.
- Remove questions that cannot be answered from `contexts`.
- Remove vague questions that require outside knowledge.
- Keep a mix of single-hop and multi-hop questions.
- Regenerate with a smaller `chunk_size` only if contexts are too broad.
- Regenerate with a larger `chunk_size` only if answers need more surrounding information.

## 14. Troubleshooting PDF Text Extraction

### Error: PDF extraction produced too little extractable text

Example:

```text
PDF extraction produced too little extractable text for RAGAS generation.
Extracted words: 138. Average words per page: 6.0.
```

This means PyMuPDF could open the PDF, but it could not extract real document text. Common causes:

- The PDF is scanned or image-based.
- The PDF came from an embedded viewer such as Scribd.
- The PDF only exposes headers, footers, page numbers, or watermarks as selectable text.
- The real content is inside page images.

RAGAS needs real text documents. If extraction only returns a few words per page, RAGAS will fail with:

```text
Documents appears to be too short (ie 100 tokens or less). Please provide longer documents.
```

Fix options:

- Use an OCR-processed PDF.
- Export the source document to `.md` and run the pipeline with `--input-dir`.
- Run OCR with a tool such as Tesseract or OCRmyPDF before generating the dataset.
- Replace the PDF with a selectable-text version.

If the folder contains both a bad scanned PDF and a good Markdown OCR output, the pipeline will use the Markdown file and skip the bad PDF.

Example:

```text
docs/KB/BachDang/I_tran_bach_dang.pdf
docs/KB/BachDang/I_tran_bach_dang.md
```

Run:

```powershell
& '.\.venv\Scripts\python.exe' -m src.evaluation_dataset.cli generate `
  --input-dir docs/KB/BachDang `
  --output-path docs/KB/BachDang/rag_eval_dataset.sample.jsonl `
  --testset-size 5
```

Quick check for OCR tools:

```powershell
Get-Command tesseract,ocrmypdf -ErrorAction SilentlyContinue
```

If this returns nothing, OCR tooling is not installed on the machine.

## 15. Troubleshooting RAGAS JSON Parser Errors

### Error: Invalid `\uXXXX` escape or `RagasOutputParserException`

Example:

```text
JSONDecodeError: Invalid \uXXXX escape
RagasOutputParserException: The output parser failed to parse the output including retries.
```

This means the local LLM returned malformed JSON while RAGAS was running an extractor such as `ThemesExtractor`.

For Ollama, keep JSON mode enabled:

```env
RAGAS_PROVIDER=ollama
RAGAS_LLM_FORMAT=json
RAGAS_TEMPERATURE=0.0
RAGAS_NUM_PREDICT=2048
```

The code defaults Ollama to:

- `llm_format`: `json`
- `temperature`: `0.0`
- `num_predict`: `2048`

If this still happens:

- Increase `RAGAS_NUM_PREDICT` to `4096`.
- Use a stronger instruction-following model for generation.
- Keep Vietnamese text as raw UTF-8 in Markdown; do not pre-escape Unicode.
- Reduce `--testset-size` and retry in smaller batches.
