from pathlib import Path

import pytest

from src.evaluation_dataset.config import EvaluationDatasetConfig


def test_validate_rejects_invalid_ratio_total():
    config = EvaluationDatasetConfig(
        input_pdf_dir=Path("data/pdfs"),
        output_path=Path("data/eval/dataset.jsonl"),
        single_hop_specific_ratio=0.5,
        multi_hop_specific_ratio=0.5,
        multi_hop_abstract_ratio=0.5,
    )

    with pytest.raises(ValueError, match="ratios must sum"):
        config.validate()


def test_validate_rejects_overlap_greater_than_chunk_size():
    config = EvaluationDatasetConfig(
        input_pdf_dir=Path("data/pdfs"),
        output_path=Path("data/eval/dataset.jsonl"),
        chunk_size=100,
        chunk_overlap=100,
    )

    with pytest.raises(ValueError, match="chunk_overlap"):
        config.validate()


def test_config_from_env_reads_ollama_provider(monkeypatch):
    monkeypatch.setenv("RAGAS_PROVIDER", "ollama")
    monkeypatch.setenv("RAGAS_LLM_BASE_URL", "https://example.ngrok-free.dev")
    monkeypatch.setenv("RAGAS_LLM_MODEL", "qwen2.5:14b")
    monkeypatch.setenv("RAGAS_EMBEDDINGS_MODEL", "nomic-embed-text")
    monkeypatch.setenv("RAGAS_TIMEOUT", "600")
    monkeypatch.setenv("RAGAS_NUM_PREDICT", "512")

    config = EvaluationDatasetConfig.from_env(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("data/eval/out.jsonl"),
    )

    assert config.provider == "ollama"
    assert config.llm_base_url == "https://example.ngrok-free.dev"
    assert config.generator_model == "qwen2.5:14b"
    assert config.embeddings_model == "nomic-embed-text"
    assert config.timeout == 600
    assert config.num_predict == 512
    assert config.llm_format == "json"


def test_config_from_env_prefers_cli_values_over_environment(monkeypatch):
    monkeypatch.setenv("RAGAS_PROVIDER", "ollama")
    monkeypatch.setenv("RAGAS_LLM_MODEL", "qwen2.5:14b")

    config = EvaluationDatasetConfig.from_env(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("data/eval/out.jsonl"),
        provider="openai",
        generator_model="gpt-4o",
    )

    assert config.provider == "openai"
    assert config.generator_model == "gpt-4o"


def test_config_from_env_uses_ollama_json_defaults(monkeypatch):
    monkeypatch.setenv("RAGAS_PROVIDER", "ollama")

    config = EvaluationDatasetConfig.from_env(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("data/eval/out.jsonl"),
    )

    assert config.llm_format == "json"
    assert config.temperature == 0.0
    assert config.num_predict == 2048


def test_config_from_env_reads_ragas_run_config(monkeypatch):
    monkeypatch.setenv("RAGAS_MAX_WORKERS", "1")
    monkeypatch.setenv("RAGAS_RUN_TIMEOUT", "900")
    monkeypatch.setenv("RAGAS_MAX_RETRIES", "3")

    config = EvaluationDatasetConfig.from_env(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("data/eval/out.jsonl"),
    )

    assert config.ragas_max_workers == 1
    assert config.ragas_run_timeout == 900
    assert config.ragas_max_retries == 3


def test_validate_rejects_unknown_provider():
    config = EvaluationDatasetConfig(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("data/eval/out.jsonl"),
        provider="unknown",
    )

    with pytest.raises(ValueError, match="provider"):
        config.validate()
