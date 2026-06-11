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
    assert settings.embeddings_base_url == "https://example.ngrok-free.dev"
    assert settings.timeout == 600
    assert settings.num_predict == 512
    assert settings.llm_format == "json"


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
    assert settings.embeddings_base_url == "http://localhost:8000/v1"


def test_model_settings_uses_llm_base_url_for_embeddings_when_missing():
    config = EvaluationDatasetConfig(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("data/eval/out.jsonl"),
        provider="ollama",
        llm_base_url="https://example.ngrok-free.dev",
    )

    settings = build_model_settings(config)

    assert settings.embeddings_base_url == "https://example.ngrok-free.dev"
