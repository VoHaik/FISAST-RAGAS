d# Dynamic RAGAS Model Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the RAG evaluation dataset generator provider-agnostic so users can switch freely between OpenAI, Ollama/self-hosted models, and OpenAI-compatible ecosystems through CLI options or `.env`.

**Architecture:** Introduce a provider factory module that builds RAGAS-compatible LLM and embedding wrappers from a normalized config. Keep `generator.py` focused on RAGAS testset generation, move model/provider construction out of it, and expose provider settings through CLI plus `.env` defaults.

**Tech Stack:** Python 3.10+, RAGAS, LangChain, `langchain-openai`, `langchain-ollama`, `python-dotenv`, Typer, Pytest.

---

## Design Decision

Do not make API keys mandatory globally.

Provider rules:

- `openai`: requires `OPENAI_API_KEY`.
- `openai-compatible`: may require an API key depending on the server. Some local gateways accept a dummy key.
- `ollama`: does not require an API key unless the reverse proxy/ngrok layer adds auth.

The implementation should read config from CLI first, then environment variables, then code defaults.

## Target Usage

### Ollama / Self-Hosted

`.env`:

```env
RAGAS_PROVIDER=ollama
RAGAS_LLM_BASE_URL=https://macaroni-hunting-mutation.ngrok-free.dev
RAGAS_LLM_MODEL=qwen2.5:14b
RAGAS_EMBEDDINGS_BASE_URL=https://macaroni-hunting-mutation.ngrok-free.dev
RAGAS_EMBEDDINGS_MODEL=nomic-embed-text
RAGAS_TIMEOUT=600
RAGAS_NUM_PREDICT=512
```

Run:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-pdf-dir docs/KB `
  --output-path data/eval/rag_eval_dataset.jsonl `
  --testset-size 20
```

### OpenAI

`.env`:

```env
RAGAS_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
RAGAS_LLM_MODEL=gpt-4o
RAGAS_EMBEDDINGS_MODEL=text-embedding-3-small
```

Run:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-pdf-dir docs/KB `
  --output-path data/eval/rag_eval_dataset.jsonl `
  --testset-size 20
```

### OpenAI-Compatible Gateway

`.env`:

```env
RAGAS_PROVIDER=openai-compatible
RAGAS_LLM_BASE_URL=http://localhost:8000/v1
RAGAS_LLM_MODEL=qwen2.5:14b
RAGAS_EMBEDDINGS_BASE_URL=http://localhost:8000/v1
RAGAS_EMBEDDINGS_MODEL=nomic-embed-text
OPENAI_API_KEY=not-needed-or-server-token
```

Run:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-pdf-dir docs/KB `
  --output-path data/eval/rag_eval_dataset.jsonl `
  --testset-size 20
```

---

## File Structure

- Modify: `requirements-eval.txt`
  - Add `langchain-ollama`.
- Modify: `src/evaluation_dataset/config.py`
  - Add provider, base URLs, temperature, timeout, and generation options.
  - Add `from_env` helper to load defaults from environment.
- Create: `src/evaluation_dataset/model_provider.py`
  - Build RAGAS-compatible LLM and embedding wrappers for each provider.
- Modify: `src/evaluation_dataset/generator.py`
  - Use `build_ragas_models(config)` instead of constructing OpenAI directly.
- Modify: `src/evaluation_dataset/cli.py`
  - Add provider-related CLI options.
  - Load `.env` before building config.
- Create: `tests/evaluation_dataset/test_model_provider.py`
  - Test provider selection without making network calls.
- Modify: `tests/evaluation_dataset/test_config.py`
  - Test environment fallback and validation.
- Modify: `docs/rag-evaluation-dataset-usage-guide.md`
  - Document dynamic provider config.
- Modify: `docs/rag-evaluation-dataset-implementation-summary.md`
  - Document provider-agnostic architecture.

---

## Task 1: Add Ollama Dependency

**Files:**
- Modify: `requirements-eval.txt`

- [ ] **Step 1: Add `langchain-ollama`**

Update dependency file:

```txt
ragas>=0.4.0
datasets>=2.19.0
pandas>=2.2.0
langchain>=0.2.0
langchain-core>=0.2.0
langchain-community>=0.2.0
langchain-openai>=0.1.0
langchain-ollama>=0.1.0
langchain-text-splitters>=0.2.0
openai>=1.0.0
pymupdf>=1.24.0
typer>=0.12.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Verify install command**

Run:

```powershell
python -m pip install -r requirements-eval.txt
```

Expected:

```text
Successfully installed ...
```

---

## Task 2: Extend Config For Dynamic Providers

**Files:**
- Modify: `src/evaluation_dataset/config.py`
- Modify: `tests/evaluation_dataset/test_config.py`

- [ ] **Step 1: Write failing config tests**

Add to `tests/evaluation_dataset/test_config.py`:

```python
from pathlib import Path

import pytest

from src.evaluation_dataset.config import EvaluationDatasetConfig


def test_config_from_env_reads_ollama_provider(monkeypatch):
    monkeypatch.setenv("RAGAS_PROVIDER", "ollama")
    monkeypatch.setenv("RAGAS_LLM_BASE_URL", "https://example.ngrok-free.dev")
    monkeypatch.setenv("RAGAS_LLM_MODEL", "qwen2.5:14b")
    monkeypatch.setenv("RAGAS_EMBEDDINGS_MODEL", "nomic-embed-text")

    config = EvaluationDatasetConfig.from_env(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("data/eval/out.jsonl"),
    )

    assert config.provider == "ollama"
    assert config.llm_base_url == "https://example.ngrok-free.dev"
    assert config.generator_model == "qwen2.5:14b"
    assert config.embeddings_model == "nomic-embed-text"


def test_validate_rejects_unknown_provider():
    config = EvaluationDatasetConfig(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("data/eval/out.jsonl"),
        provider="unknown",
    )

    with pytest.raises(ValueError, match="provider"):
        config.validate()
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
python -m pytest tests/evaluation_dataset/test_config.py -v
```

Expected:

```text
FAILED ... EvaluationDatasetConfig has no attribute from_env
```

- [ ] **Step 3: Implement provider config**

Replace `src/evaluation_dataset/config.py` with:

```python
import os
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_PROVIDERS = {"openai", "openai-compatible", "ollama"}


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
    provider: str = "openai"
    generator_model: str = "gpt-4o"
    embeddings_model: str = "text-embedding-3-small"
    llm_base_url: str | None = None
    embeddings_base_url: str | None = None
    temperature: float = 0.2
    timeout: int = 600
    num_predict: int = 512

    @classmethod
    def from_env(
        cls,
        input_pdf_dir: Path,
        output_path: Path,
        output_format: str = "jsonl",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        testset_size: int = 100,
        single_hop_specific_ratio: float = 0.5,
        multi_hop_specific_ratio: float = 0.25,
        multi_hop_abstract_ratio: float = 0.25,
        provider: str | None = None,
        generator_model: str | None = None,
        embeddings_model: str | None = None,
        llm_base_url: str | None = None,
        embeddings_base_url: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
        num_predict: int | None = None,
    ) -> "EvaluationDatasetConfig":
        selected_provider = provider or os.getenv("RAGAS_PROVIDER", "openai")
        return cls(
            input_pdf_dir=input_pdf_dir,
            output_path=output_path,
            output_format=output_format,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            testset_size=testset_size,
            single_hop_specific_ratio=single_hop_specific_ratio,
            multi_hop_specific_ratio=multi_hop_specific_ratio,
            multi_hop_abstract_ratio=multi_hop_abstract_ratio,
            provider=selected_provider,
            generator_model=generator_model
            or os.getenv("RAGAS_LLM_MODEL")
            or _default_llm_model(selected_provider),
            embeddings_model=embeddings_model
            or os.getenv("RAGAS_EMBEDDINGS_MODEL")
            or _default_embeddings_model(selected_provider),
            llm_base_url=llm_base_url or os.getenv("RAGAS_LLM_BASE_URL"),
            embeddings_base_url=embeddings_base_url or os.getenv("RAGAS_EMBEDDINGS_BASE_URL"),
            temperature=temperature
            if temperature is not None
            else float(os.getenv("RAGAS_TEMPERATURE", "0.2")),
            timeout=timeout if timeout is not None else int(os.getenv("RAGAS_TIMEOUT", "600")),
            num_predict=num_predict
            if num_predict is not None
            else int(os.getenv("RAGAS_NUM_PREDICT", "512")),
        )

    def validate(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"provider must be one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
            )
        if self.output_format not in {"jsonl", "csv"}:
            raise ValueError("output_format must be either 'jsonl' or 'csv'")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be 0 or greater")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.testset_size <= 0:
            raise ValueError("testset_size must be greater than 0")
        if self.temperature < 0:
            raise ValueError("temperature must be 0 or greater")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if self.num_predict <= 0:
            raise ValueError("num_predict must be greater than 0")

        total = (
            self.single_hop_specific_ratio
            + self.multi_hop_specific_ratio
            + self.multi_hop_abstract_ratio
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError("question type ratios must sum to 1.0")


def _default_llm_model(provider: str) -> str:
    if provider == "ollama":
        return "qwen2.5:14b"
    return "gpt-4o"


def _default_embeddings_model(provider: str) -> str:
    if provider == "ollama":
        return "nomic-embed-text"
    return "text-embedding-3-small"
```

- [ ] **Step 4: Run tests and confirm pass**

Run:

```powershell
python -m pytest tests/evaluation_dataset/test_config.py -v
```

Expected:

```text
4 passed
```

---

## Task 3: Create Provider Factory

**Files:**
- Create: `src/evaluation_dataset/model_provider.py`
- Create: `tests/evaluation_dataset/test_model_provider.py`

- [ ] **Step 1: Write failing provider tests without network calls**

Create `tests/evaluation_dataset/test_model_provider.py`:

```python
from pathlib import Path

from src.evaluation_dataset.config import EvaluationDatasetConfig
from src.evaluation_dataset.model_provider import build_model_settings


def test_build_model_settings_for_ollama():
    config = EvaluationDatasetConfig(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("data/eval/out.jsonl"),
        provider="ollama",
        generator_model="qwen2.5:14b",
        embeddings_model="nomic-embed-text",
        llm_base_url="https://example.ngrok-free.dev",
        embeddings_base_url="https://example.ngrok-free.dev",
        timeout=600,
        num_predict=512,
    )

    settings = build_model_settings(config)

    assert settings.provider == "ollama"
    assert settings.llm_model == "qwen2.5:14b"
    assert settings.embeddings_model == "nomic-embed-text"
    assert settings.llm_base_url == "https://example.ngrok-free.dev"


def test_build_model_settings_for_openai_compatible():
    config = EvaluationDatasetConfig(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("data/eval/out.jsonl"),
        provider="openai-compatible",
        generator_model="qwen2.5:14b",
        embeddings_model="nomic-embed-text",
        llm_base_url="http://localhost:8000/v1",
        embeddings_base_url="http://localhost:8000/v1",
    )

    settings = build_model_settings(config)

    assert settings.provider == "openai-compatible"
    assert settings.llm_base_url == "http://localhost:8000/v1"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```powershell
python -m pytest tests/evaluation_dataset/test_model_provider.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'src.evaluation_dataset.model_provider'
```

- [ ] **Step 3: Implement provider settings and factory**

Create `src/evaluation_dataset/model_provider.py`:

```python
from dataclasses import dataclass
from typing import Any

from src.evaluation_dataset.config import EvaluationDatasetConfig


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    llm_model: str
    embeddings_model: str
    llm_base_url: str | None
    embeddings_base_url: str | None
    temperature: float
    timeout: int
    num_predict: int


@dataclass(frozen=True)
class RagasModels:
    llm: Any
    embeddings: Any


def build_model_settings(config: EvaluationDatasetConfig) -> ModelSettings:
    config.validate()
    return ModelSettings(
        provider=config.provider,
        llm_model=config.generator_model,
        embeddings_model=config.embeddings_model,
        llm_base_url=config.llm_base_url,
        embeddings_base_url=config.embeddings_base_url or config.llm_base_url,
        temperature=config.temperature,
        timeout=config.timeout,
        num_predict=config.num_predict,
    )


def build_ragas_models(config: EvaluationDatasetConfig) -> RagasModels:
    settings = build_model_settings(config)
    if settings.provider == "openai":
        return _build_openai_models(settings)
    if settings.provider == "openai-compatible":
        return _build_openai_compatible_models(settings)
    if settings.provider == "ollama":
        return _build_ollama_models(settings)
    raise ValueError(f"Unsupported provider: {settings.provider}")


def _build_openai_models(settings: ModelSettings) -> RagasModels:
    imports = _load_openai_imports()
    llm = imports["LangchainLLMWrapper"](
        imports["ChatOpenAI"](
            model=settings.llm_model,
            temperature=settings.temperature,
            timeout=settings.timeout,
        )
    )
    embeddings = imports["LangchainEmbeddingsWrapper"](
        imports["OpenAIEmbeddings"](model=settings.embeddings_model)
    )
    return RagasModels(llm=llm, embeddings=embeddings)


def _build_openai_compatible_models(settings: ModelSettings) -> RagasModels:
    imports = _load_openai_imports()
    llm = imports["LangchainLLMWrapper"](
        imports["ChatOpenAI"](
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=settings.temperature,
            timeout=settings.timeout,
        )
    )
    embeddings = imports["LangchainEmbeddingsWrapper"](
        imports["OpenAIEmbeddings"](
            model=settings.embeddings_model,
            base_url=settings.embeddings_base_url,
        )
    )
    return RagasModels(llm=llm, embeddings=embeddings)


def _build_ollama_models(settings: ModelSettings) -> RagasModels:
    imports = _load_ollama_imports()
    llm = imports["LangchainLLMWrapper"](
        imports["OllamaLLM"](
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.temperature,
            timeout=settings.timeout,
            num_predict=settings.num_predict,
        )
    )
    embeddings = imports["LangchainEmbeddingsWrapper"](
        imports["OllamaEmbeddings"](
            base_url=settings.embeddings_base_url,
            model=settings.embeddings_model,
        )
    )
    return RagasModels(llm=llm, embeddings=embeddings)


def _load_openai_imports() -> dict[str, Any]:
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as error:
        raise RuntimeError(
            "OpenAI provider dependencies are missing. "
            "Install them with: python -m pip install -r requirements-eval.txt"
        ) from error
    return {
        "ChatOpenAI": ChatOpenAI,
        "OpenAIEmbeddings": OpenAIEmbeddings,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
    }


def _load_ollama_imports() -> dict[str, Any]:
    try:
        from langchain_ollama import OllamaEmbeddings, OllamaLLM
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as error:
        raise RuntimeError(
            "Ollama provider dependencies are missing. "
            "Install them with: python -m pip install -r requirements-eval.txt"
        ) from error
    return {
        "OllamaEmbeddings": OllamaEmbeddings,
        "OllamaLLM": OllamaLLM,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
    }
```

- [ ] **Step 4: Run provider settings tests**

Run:

```powershell
python -m pytest tests/evaluation_dataset/test_model_provider.py -v
```

Expected:

```text
2 passed
```

---

## Task 4: Update Generator To Use Provider Factory

**Files:**
- Modify: `src/evaluation_dataset/generator.py`

- [ ] **Step 1: Replace direct OpenAI construction**

Update `src/evaluation_dataset/generator.py`:

```python
from typing import Any

import pandas as pd

from src.evaluation_dataset.config import EvaluationDatasetConfig
from src.evaluation_dataset.model_provider import build_ragas_models


def generate_ragas_testset(documents: list[Any], config: EvaluationDatasetConfig) -> pd.DataFrame:
    config.validate()

    imports = _load_ragas_testset_imports()
    models = build_ragas_models(config)

    generator = imports["TestsetGenerator"](
        llm=models.llm,
        embedding_model=models.embeddings,
    )
    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=config.testset_size,
        query_distribution=[
            (
                imports["SingleHopSpecificQuerySynthesizer"](llm=models.llm),
                config.single_hop_specific_ratio,
            ),
            (
                imports["MultiHopSpecificQuerySynthesizer"](llm=models.llm),
                config.multi_hop_specific_ratio,
            ),
            (
                imports["MultiHopAbstractQuerySynthesizer"](llm=models.llm),
                config.multi_hop_abstract_ratio,
            ),
        ],
    )
    return testset.to_pandas()


def _load_ragas_testset_imports() -> dict[str, Any]:
    try:
        from ragas.testset import TestsetGenerator
        from ragas.testset.synthesizers import (
            MultiHopAbstractQuerySynthesizer,
            MultiHopSpecificQuerySynthesizer,
            SingleHopSpecificQuerySynthesizer,
        )
    except ImportError as error:
        raise RuntimeError(
            "RAGAS testset dependencies are missing. "
            "Install them with: python -m pip install -r requirements-eval.txt"
        ) from error

    return {
        "TestsetGenerator": TestsetGenerator,
        "SingleHopSpecificQuerySynthesizer": SingleHopSpecificQuerySynthesizer,
        "MultiHopSpecificQuerySynthesizer": MultiHopSpecificQuerySynthesizer,
        "MultiHopAbstractQuerySynthesizer": MultiHopAbstractQuerySynthesizer,
    }
```

- [ ] **Step 2: Run import check**

Run:

```powershell
python -c "from src.evaluation_dataset.generator import generate_ragas_testset; print('generator-ok')"
```

Expected:

```text
generator-ok
```

---

## Task 5: Expose Provider Options In CLI

**Files:**
- Modify: `src/evaluation_dataset/cli.py`

- [ ] **Step 1: Update CLI options**

Replace config construction in `src/evaluation_dataset/cli.py` with:

```python
    config = EvaluationDatasetConfig.from_env(
        input_pdf_dir=input_pdf_dir,
        output_path=output_path,
        output_format=output_format,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        testset_size=testset_size,
        single_hop_specific_ratio=single_hop_specific_ratio,
        multi_hop_specific_ratio=multi_hop_specific_ratio,
        multi_hop_abstract_ratio=multi_hop_abstract_ratio,
        provider=provider,
        generator_model=generator_model,
        embeddings_model=embeddings_model,
        llm_base_url=llm_base_url,
        embeddings_base_url=embeddings_base_url,
        temperature=temperature,
        timeout=timeout,
        num_predict=num_predict,
    )
```

Add command parameters:

```python
    provider: str | None = typer.Option(None),
    llm_base_url: str | None = typer.Option(None),
    embeddings_base_url: str | None = typer.Option(None),
    generator_model: str | None = typer.Option(None),
    embeddings_model: str | None = typer.Option(None),
    temperature: float | None = typer.Option(None),
    timeout: int | None = typer.Option(None),
    num_predict: int | None = typer.Option(None),
```

- [ ] **Step 2: Run CLI help**

Run:

```powershell
python -m src.evaluation_dataset.cli --help
```

Expected includes:

```text
--provider
--llm-base-url
--embeddings-base-url
--generator-model
--embeddings-model
```

---

## Task 6: Update Documentation

**Files:**
- Modify: `docs/rag-evaluation-dataset-usage-guide.md`
- Modify: `docs/rag-evaluation-dataset-implementation-summary.md`

- [ ] **Step 1: Add provider section to usage guide**

Add this section to `docs/rag-evaluation-dataset-usage-guide.md`:

```markdown
## Dynamic Model Provider Configuration

The pipeline supports these providers:

- `openai`
- `openai-compatible`
- `ollama`

Configuration priority:

1. CLI options
2. `.env`
3. source defaults in `EvaluationDatasetConfig`

### Ollama Example

```env
RAGAS_PROVIDER=ollama
RAGAS_LLM_BASE_URL=https://macaroni-hunting-mutation.ngrok-free.dev
RAGAS_LLM_MODEL=qwen2.5:14b
RAGAS_EMBEDDINGS_BASE_URL=https://macaroni-hunting-mutation.ngrok-free.dev
RAGAS_EMBEDDINGS_MODEL=nomic-embed-text
RAGAS_TIMEOUT=600
RAGAS_NUM_PREDICT=512
```

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-pdf-dir docs/KB `
  --output-path data/eval/rag_eval_dataset.jsonl `
  --testset-size 20
```

Ollama does not require `OPENAI_API_KEY` unless your proxy or gateway enforces authentication.
```

- [ ] **Step 2: Update implementation summary**

Add:

```markdown
## Dynamic Provider Architecture

Model construction is handled by `src/evaluation_dataset/model_provider.py`.

`generator.py` no longer imports OpenAI or Ollama directly. It receives RAGAS-compatible wrappers from `build_ragas_models(config)`.
```

---

## Task 7: Final Verification

**Files:**
- All modified files

- [ ] **Step 1: Run unit tests**

```powershell
python -m pytest tests/evaluation_dataset -v
```

Expected:

```text
All tests pass
```

- [ ] **Step 2: Run import checks**

```powershell
python -c "from src.evaluation_dataset.config import EvaluationDatasetConfig; from src.evaluation_dataset.model_provider import build_model_settings; from src.evaluation_dataset.cli import app; print('dynamic-provider-import-ok')"
```

Expected:

```text
dynamic-provider-import-ok
```

- [ ] **Step 3: Run CLI help**

```powershell
python -m src.evaluation_dataset.cli --help
```

Expected:

```text
--provider
--llm-base-url
--embeddings-base-url
```

- [ ] **Step 4: Run Ollama smoke generation**

Prerequisite:

```env
RAGAS_PROVIDER=ollama
RAGAS_LLM_BASE_URL=https://macaroni-hunting-mutation.ngrok-free.dev
RAGAS_LLM_MODEL=qwen2.5:14b
RAGAS_EMBEDDINGS_BASE_URL=https://macaroni-hunting-mutation.ngrok-free.dev
RAGAS_EMBEDDINGS_MODEL=nomic-embed-text
```

Run:

```powershell
python -m src.evaluation_dataset.cli generate `
  --input-pdf-dir docs/KB `
  --output-path data/eval/rag_eval_ollama_sample.jsonl `
  --testset-size 3
```

Expected:

```text
Generated 3 rows at data/eval/rag_eval_ollama_sample.jsonl
```

---

## Self-Review

Spec coverage:

- Dynamic model provider selection is covered by Tasks 2, 3, 4, and 5.
- Ollama self-hosted support is covered by Task 3 and usage examples.
- OpenAI and OpenAI-compatible ecosystems are covered by Task 3.
- `.env` usage is covered by Task 2 and Task 6.
- CLI override behavior is covered by Task 5.

Placeholder scan:

- No placeholder instructions remain.

Type consistency:

- `EvaluationDatasetConfig`, `ModelSettings`, `RagasModels`, `build_model_settings`, and `build_ragas_models` are used consistently.
