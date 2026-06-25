from pathlib import Path
from types import SimpleNamespace

from src.evaluation_dataset.config import EvaluationDatasetConfig
from src.evaluation_dataset import generator


def test_generate_ragas_testset_passes_run_config(monkeypatch):
    captured = {}

    class FakeRunConfig:
        def __init__(self, timeout, max_retries, max_workers):
            self.timeout = timeout
            self.max_retries = max_retries
            self.max_workers = max_workers

    class FakeSynthesizer:
        def __init__(self, llm):
            self.llm = llm

    class FakeTestset:
        def to_pandas(self):
            return "frame"

    class FakeTestsetGenerator:
        def __init__(self, llm, embedding_model):
            self.llm = llm
            self.embedding_model = embedding_model

        def generate_with_chunks(self, chunks, **kwargs):
            captured["documents"] = chunks
            captured.update(kwargs)
            return FakeTestset()

    monkeypatch.setattr(
        generator,
        "_load_ragas_testset_imports",
        lambda: {
            "TestsetGenerator": FakeTestsetGenerator,
            "SingleHopSpecificQuerySynthesizer": FakeSynthesizer,
            "MultiHopSpecificQuerySynthesizer": FakeSynthesizer,
            "MultiHopAbstractQuerySynthesizer": FakeSynthesizer,
            "RunConfig": FakeRunConfig,
        },
    )
    monkeypatch.setattr(
        generator,
        "build_ragas_models",
        lambda config: SimpleNamespace(llm="llm", embeddings="embeddings"),
    )

    config = EvaluationDatasetConfig(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("out.jsonl"),
        ragas_max_workers=1,
        ragas_run_timeout=900,
        ragas_max_retries=3,
    )

    frame = generator.generate_ragas_testset(["doc"], config)

    assert frame == "frame"
    assert captured["documents"] == ["doc"]
    assert captured["run_config"].max_workers == 1
    assert captured["run_config"].timeout == 900
    assert captured["run_config"].max_retries == 3


def test_generate_ragas_testset_calls_adapt(monkeypatch):
    adapted_languages = []

    # Mock PromptMixin.adapt_prompts to avoid real LLM calls during test
    async def mock_adapt_prompts(self, language, llm, adapt_instruction=False):
        return {}
    monkeypatch.setattr("ragas.prompt.mixin.PromptMixin.adapt_prompts", mock_adapt_prompts)

    class FakeSynthesizerWithAdapt:
        def __init__(self, llm):
            self.llm = llm

        def adapt(self, language, llm, cache_dir=None):
            adapted_languages.append(language)

    class FakeRunConfig:
        def __init__(self, timeout, max_retries, max_workers):
            pass

    class FakeTestset:
        def to_pandas(self):
            return "frame"

    class FakeTestsetGenerator:
        def __init__(self, llm, embedding_model):
            pass

        def generate_with_chunks(self, chunks, **kwargs):
            return FakeTestset()

    monkeypatch.setattr(
        generator,
        "_load_ragas_testset_imports",
        lambda: {
            "TestsetGenerator": FakeTestsetGenerator,
            "SingleHopSpecificQuerySynthesizer": FakeSynthesizerWithAdapt,
            "MultiHopSpecificQuerySynthesizer": FakeSynthesizerWithAdapt,
            "MultiHopAbstractQuerySynthesizer": FakeSynthesizerWithAdapt,
            "RunConfig": FakeRunConfig,
        },
    )
    monkeypatch.setattr(
        generator,
        "build_ragas_models",
        lambda config: SimpleNamespace(llm="llm", embeddings="embeddings"),
    )

    config = EvaluationDatasetConfig(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("out.jsonl"),
        language="vietnamese",
        adapt_prompts=True,
    )

    generator.generate_ragas_testset(["doc"], config)
    assert adapted_languages == ["vietnamese", "vietnamese", "vietnamese"]

    # If adapt_prompts is False, it should skip adaptation even if language is vietnamese
    adapted_languages.clear()
    config_no_adapt = EvaluationDatasetConfig(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("out.jsonl"),
        language="vietnamese",
        adapt_prompts=False,
    )
    generator.generate_ragas_testset(["doc"], config_no_adapt)
    assert len(adapted_languages) == 0

    # If language is "english", it should skip adaptation
    adapted_languages.clear()
    config_english = EvaluationDatasetConfig(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("out.jsonl"),
        language="english",
        adapt_prompts=True,
    )
    generator.generate_ragas_testset(["doc"], config_english)
    assert len(adapted_languages) == 0


def test_generate_ragas_testset_injects_language_instruction(monkeypatch):
    updated_prompts = []

    class FakePrompt:
        def __init__(self, instruction):
            self.instruction = instruction

    class FakeSynthesizerWithPrompts:
        def __init__(self, llm):
            self.llm = llm
            self.prompts = {
                "query_answer_generation_prompt": FakePrompt("Base instruction.")
            }

        def get_prompts(self):
            return self.prompts

        def set_prompts(self, **kwargs):
            self.prompts.update(kwargs)
            updated_prompts.append(self.prompts["query_answer_generation_prompt"].instruction)

    class FakeRunConfig:
        def __init__(self, timeout, max_retries, max_workers):
            pass

    class FakeTestset:
        def to_pandas(self):
            return "frame"

    class FakeTestsetGenerator:
        def __init__(self, llm, embedding_model):
            pass

        def generate_with_chunks(self, chunks, **kwargs):
            return FakeTestset()

    monkeypatch.setattr(
        generator,
        "_load_ragas_testset_imports",
        lambda: {
            "TestsetGenerator": FakeTestsetGenerator,
            "SingleHopSpecificQuerySynthesizer": FakeSynthesizerWithPrompts,
            "MultiHopSpecificQuerySynthesizer": FakeSynthesizerWithPrompts,
            "MultiHopAbstractQuerySynthesizer": FakeSynthesizerWithPrompts,
            "RunConfig": FakeRunConfig,
        },
    )
    monkeypatch.setattr(
        generator,
        "build_ragas_models",
        lambda config: SimpleNamespace(llm="llm", embeddings="embeddings"),
    )

    config = EvaluationDatasetConfig(
        input_pdf_dir=Path("docs/KB"),
        output_path=Path("out.jsonl"),
        language="vietnamese",
    )

    generator.generate_ragas_testset(["doc"], config)
    assert len(updated_prompts) == 3
    for inst in updated_prompts:
        assert "Always write the query ('query') and answer ('answer') in proper, fully-signed Vietnamese (tiếng Việt có dấu đầy đủ, đúng chính tả, không viết không dấu hoặc thiếu dấu)." in inst


