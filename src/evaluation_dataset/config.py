import os
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_PROVIDERS = {"openai", "openai-compatible", "ollama"}


@dataclass(frozen=True)
class EvaluationDatasetConfig:
    input_pdf_dir: Path
    output_path: Path
    output_format: str = "jsonl"
    language: str = "vietnamese"
    adapt_prompts: bool = False
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
    llm_format: str = ""
    ragas_max_workers: int = 16
    ragas_run_timeout: int = 600
    ragas_max_retries: int = 2

    @classmethod
    def from_env(
        cls,
        input_pdf_dir: Path,
        output_path: Path,
        output_format: str = "jsonl",
        language: str | None = None,
        adapt_prompts: bool | None = None,
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
        llm_format: str | None = None,
        ragas_max_workers: int | None = None,
        ragas_run_timeout: int | None = None,
        ragas_max_retries: int | None = None,
    ) -> "EvaluationDatasetConfig":
        selected_provider = provider or os.getenv("RAGAS_PROVIDER", "openai")
        return cls(
            input_pdf_dir=input_pdf_dir,
            output_path=output_path,
            output_format=output_format,
            language=language or os.getenv("RAGAS_LANGUAGE", "vietnamese"),
            adapt_prompts=adapt_prompts if adapt_prompts is not None else os.getenv("RAGAS_ADAPT_PROMPTS", "false").lower() == "true",
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
            else float(os.getenv("RAGAS_TEMPERATURE", _default_temperature(selected_provider))),
            timeout=timeout if timeout is not None else int(os.getenv("RAGAS_TIMEOUT", "600")),
            num_predict=num_predict
            if num_predict is not None
            else int(os.getenv("RAGAS_NUM_PREDICT", _default_num_predict(selected_provider))),
            llm_format=llm_format
            if llm_format is not None
            else os.getenv("RAGAS_LLM_FORMAT", _default_llm_format(selected_provider)),
            ragas_max_workers=ragas_max_workers
            if ragas_max_workers is not None
            else int(os.getenv("RAGAS_MAX_WORKERS", "16")),
            ragas_run_timeout=ragas_run_timeout
            if ragas_run_timeout is not None
            else int(os.getenv("RAGAS_RUN_TIMEOUT", os.getenv("RAGAS_TIMEOUT", "600"))),
            ragas_max_retries=ragas_max_retries
            if ragas_max_retries is not None
            else int(os.getenv("RAGAS_MAX_RETRIES", "2")),
        )

    def validate(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"provider must be one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
            )
        if self.output_format not in {"jsonl", "csv"}:
            raise ValueError("output_format must be either 'jsonl' or 'csv'")
        if not self.language:
            raise ValueError("language must be a non-empty string")
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
        if self.llm_format not in {"", "json"}:
            raise ValueError("llm_format must be either '' or 'json'")
        if self.ragas_max_workers <= 0:
            raise ValueError("ragas_max_workers must be greater than 0")
        if self.ragas_run_timeout <= 0:
            raise ValueError("ragas_run_timeout must be greater than 0")
        if self.ragas_max_retries < 0:
            raise ValueError("ragas_max_retries must be 0 or greater")

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


def _default_temperature(provider: str) -> str:
    if provider == "ollama":
        return "0.0"
    return "0.2"


def _default_num_predict(provider: str) -> str:
    if provider == "ollama":
        return "2048"
    return "512"


def _default_llm_format(provider: str) -> str:
    if provider == "ollama":
        return "json"
    return ""
