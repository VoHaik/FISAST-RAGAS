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
    llm_format: str
    system_prompt: str | None = None


@dataclass(frozen=True)
class RagasModels:
    llm: Any
    embeddings: Any


def build_model_settings(config: EvaluationDatasetConfig) -> ModelSettings:
    config.validate()
    system_prompt = None
    if config.language and config.language.lower() == "vietnamese":
        system_prompt = "Always write the question ('query') and answer ('answer') in Vietnamese."

    return ModelSettings(
        provider=config.provider,
        llm_model=config.generator_model,
        embeddings_model=config.embeddings_model,
        llm_base_url=config.llm_base_url,
        embeddings_base_url=config.embeddings_base_url or config.llm_base_url,
        temperature=config.temperature,
        timeout=config.timeout,
        num_predict=config.num_predict,
        llm_format=config.llm_format or ("json" if config.provider == "ollama" else ""),
        system_prompt=system_prompt,
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
    ollama_kwargs = {
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "temperature": settings.temperature,
        "timeout": settings.timeout,
        "num_predict": settings.num_predict,
        "format": settings.llm_format,
    }
    if settings.system_prompt:
        ollama_kwargs["system"] = settings.system_prompt

    llm = imports["LangchainLLMWrapper"](
        imports["OllamaLLM"](**ollama_kwargs)
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
