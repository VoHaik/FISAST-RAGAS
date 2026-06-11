# RAG Evaluation Dataset Pipeline

## Purpose

This pipeline creates an evaluation dataset from PDF source documents. The dataset is used to measure RAG quality with generated questions, source contexts, and reference answers.

## Why RAGAS

Use RAGAS first for this pipeline because the target is RAG-specific dataset generation over source documents. The generated testset is designed around retrieval scenarios such as single-hop and multi-hop questions.

DeepEval is not required for the first version. Add it later if the project needs DeepEval `Golden` objects, CI quality gates, filtration, or custom evaluation metrics.

## Input

Place PDF or Markdown files in:

```text
data/pdfs/
```

Supported input formats:

- `.pdf`
- `.md`
- `.markdown`

Use Markdown when OCR/chatbot extraction is better than PDF text extraction.

## Output

The recommended output is:

```text
data/eval/rag_eval_dataset.jsonl
```

Each row is normalized to include:

- `question`
- `contexts`
- `ground_truth`

RAGAS source columns such as `user_input`, `reference_contexts`, and `reference` are mapped to the project schema during export.

## Install

```powershell
python -m pip install -r requirements-eval.txt
```

## Configure Model Provider

The pipeline supports:

- `openai`
- `openai-compatible`
- `ollama`

For Ollama/self-hosted models, create a `.env` file at the project root:

```env
RAGAS_PROVIDER=ollama
RAGAS_LLM_BASE_URL=https://macaroni-hunting-mutation.ngrok-free.dev
RAGAS_LLM_MODEL=qwen2.5:14b
RAGAS_EMBEDDINGS_BASE_URL=https://macaroni-hunting-mutation.ngrok-free.dev
RAGAS_EMBEDDINGS_MODEL=nomic-embed-text
RAGAS_TIMEOUT=600
RAGAS_NUM_PREDICT=512
```

Ollama does not need `OPENAI_API_KEY` unless your proxy/gateway requires authentication.

For OpenAI:

```env
RAGAS_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
RAGAS_LLM_MODEL=gpt-4o
RAGAS_EMBEDDINGS_MODEL=text-embedding-3-small
```

## Generate Dataset

This is an offline CLI job, not a web app. No server needs to be started.

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir data/pdfs `
  --output-path data/eval/rag_eval_dataset.jsonl `
  --output-format jsonl `
  --testset-size 20
```

## Recommended First Run

Start with a small dataset:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-dir data/pdfs `
  --output-path data/eval/rag_eval_dataset.sample.jsonl `
  --output-format jsonl `
  --testset-size 5
```

Manually inspect the sample before generating 100+ rows.

## Generation Settings

Defaults:

- `chunk_size`: `1000`
- `chunk_overlap`: `150`
- `testset_size`: `100`
- `single_hop_specific_ratio`: `0.5`
- `multi_hop_specific_ratio`: `0.25`
- `multi_hop_abstract_ratio`: `0.25`
- `provider`: `openai`
- `generator_model`: `gpt-4o`
- `embeddings_model`: `text-embedding-3-small`
- `timeout`: `600`
- `num_predict`: `512`

## Quality Review Checklist

Before using the dataset for RAG evaluation:

- Remove duplicate or near-duplicate questions.
- Remove questions whose answer is not supported by the provided contexts.
- Remove vague questions that require external knowledge.
- Keep a balanced mix of single-hop and multi-hop questions.
- Keep source metadata where available so failed examples can be traced back to the PDF.

## Validation

Run unit tests:

```powershell
python -m pytest tests/evaluation_dataset -v
```

Run CLI help:

```powershell
python -m src.evaluation_dataset.cli --help
```
