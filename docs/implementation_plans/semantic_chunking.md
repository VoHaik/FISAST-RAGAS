# Implementation Plan - Integrating Semantic Chunking with Persistent Disk Cache

This plan details how to integrate **Semantic Chunking** as the **default** chunking method in the RAG Evaluation Dataset generation pipeline, incorporating a local **Persistent Disk Cache** for embeddings to prevent redundant LLM/Ollama API requests.

## User Review Required

> [!IMPORTANT]
> **Persistent Disk Cache**: The pipeline will save generated embedding vectors into a local `.cache/embeddings/` directory. If you run the pipeline again on the same text, it will load vectors from the disk instantly, reducing network queries and preventing Ollama gateway timeouts.
>
> **Dynamic Fallback**: If `langchain-experimental` is missing or the embedding model is not initialized, it automatically falls back to `RecursiveCharacterTextSplitter` to avoid pipeline failure.

## Proposed Changes

We will introduce the `chunking_method` option which will default to `semantic` but can be manually set to `recursive`.

---

### [Dependency Layer]

#### [MODIFY] [requirements-eval.txt](file:///c:/Users/KHAI/Documents/semester%208/FISAT-2026/requirements-eval.txt)
* Append `langchain-experimental>=0.0.1` to the dependencies.

---

### [Core Config & CLI CLI]

#### [MODIFY] [config.py](file:///c:/Users/KHAI/Documents/semester%208/FISAT-2026/src/evaluation_dataset/config.py)
* Add `chunking_method: str = "semantic"` to `EvaluationDatasetConfig`.
* Load `RAGAS_CHUNKING_METHOD` in `from_env()`.
* Add validation in `validate()` to ensure `chunking_method` is either `recursive` or `semantic`.

#### [MODIFY] [cli.py](file:///c:/Users/KHAI/Documents/semester%208/FISAT-2026/src/evaluation_dataset/cli.py)
* Add `--chunking-method` as a command-line parameter in the `generate` command, defaulting to `"semantic"`.
* Instantiate the Ragas models using `build_ragas_models(config)` before chunking.
* Pass the raw embedding model (`models.embeddings.embeddings`) and `config.chunking_method` to the `chunk_pdf_pages` function.
* Pass the pre-built `models` instance to `generate_ragas_testset` to avoid rebuilding models.

---

### [Document Splitting Layer]

#### [MODIFY] [chunker.py](file:///c:/Users/KHAI/Documents/semester%208/FISAT-2026/src/evaluation_dataset/chunker.py)
* Update `chunk_pdf_pages` to accept `embeddings: Any = None` and `chunking_method: str = "semantic"`.
* Implement `LocalFileStore` and `CacheBackedEmbeddings` initialization inside `chunk_pdf_pages`.
* Wrap the passed `embeddings` model with the cache store.
* Initialize `SemanticChunker` with the cached embedding instance if `chunking_method == "semantic"`.
* Split text using the semantic splitter if available; otherwise, fallback to `_split_text` (using the recursive character splitter).

#### [MODIFY] [generator.py](file:///c:/Users/KHAI/Documents/semester%208/FISAT-2026/src/evaluation_dataset/generator.py)
* Update `generate_ragas_testset` to accept an optional pre-built `models: RagasModels | None = None` parameter. If provided, reuse these models rather than instantiating them again.

---

### [Documentation]

#### [MODIFY] [rag-evaluation-dataset-usage-guide.md](file:///c:/Users/KHAI/Documents/semester%208/FISAT-2026/docs/rag-evaluation-dataset-usage-guide.md)
* Update parameters reference and examples to document `--chunking-method` (defaulting to `semantic`) and emphasize Markdown input folder configurations.

---

## Verification Plan

### Automated Tests
* Add test cases in `tests/evaluation_dataset/test_chunker.py` to:
  * Verify `chunk_pdf_pages` works with `chunking_method="semantic"` when a mock embedding model is passed.
  * Verify graceful fallback to `recursive` character chunking if embeddings are not supplied.
* Run the pytest suite:
  ```powershell
  & '.\venv\Scripts\python.exe' -m pytest tests/evaluation_dataset
  ```

### Manual Verification
* Run a smoke test dataset generation using semantic chunking with a size of 1:
  ```powershell
  & '.\venv\Scripts\python.exe' -m src.evaluation_dataset.cli generate `
    --input-dir docs/KB/BachDang `
    --output-path docs/KB/BachDang/rag_eval_dataset.semantic.jsonl `
    --testset-size 1 `
    --chunking-method semantic `
    --ragas-max-workers 1
  ```
