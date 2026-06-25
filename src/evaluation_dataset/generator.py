from typing import Any

import pandas as pd

from src.evaluation_dataset.config import EvaluationDatasetConfig
from src.evaluation_dataset.model_provider import build_ragas_models


def generate_ragas_testset(documents: list[Any], config: EvaluationDatasetConfig) -> pd.DataFrame:
    config.validate()

    imports = _load_ragas_testset_imports()
    models = build_ragas_models(config)
    run_config = imports["RunConfig"](
        timeout=config.ragas_run_timeout,
        max_retries=config.ragas_max_retries,
        max_workers=config.ragas_max_workers,
    )

    generator = imports["TestsetGenerator"](
        llm=models.llm,
        embedding_model=models.embeddings,
    )
    sh_synthesizer = imports["SingleHopSpecificQuerySynthesizer"](llm=models.llm)
    mh_spec_synthesizer = imports["MultiHopSpecificQuerySynthesizer"](llm=models.llm)
    mh_abs_synthesizer = imports["MultiHopAbstractQuerySynthesizer"](llm=models.llm)

    if config.adapt_prompts and config.language and config.language.lower() != "english":
        from ragas.utils import async_to_sync
        for synth in (sh_synthesizer, mh_spec_synthesizer, mh_abs_synthesizer):
            if hasattr(synth, "adapt_prompts") and hasattr(synth, "set_prompts"):
                adapted = async_to_sync(synth.adapt_prompts)(config.language, models.llm)
                synth.set_prompts(**adapted)
            elif hasattr(synth, "adapt"):
                synth.adapt(config.language, models.llm)

    if config.language and config.language.lower() != "english":
        lang_str = config.language.capitalize()
        if config.language.lower() == "vietnamese":
            lang_str = "proper, fully-signed Vietnamese (tiếng Việt có dấu đầy đủ, đúng chính tả, không viết không dấu hoặc thiếu dấu)"
        for synth in (sh_synthesizer, mh_spec_synthesizer, mh_abs_synthesizer):
            if hasattr(synth, "get_prompts") and hasattr(synth, "set_prompts"):
                prompts = synth.get_prompts()
                if "query_answer_generation_prompt" in prompts:
                    prompt = prompts["query_answer_generation_prompt"]
                    prompt.instruction = (
                        prompt.instruction
                        + f" Always write the query ('query') and answer ('answer') in {lang_str}."
                        + " Crucial: Ensure that the generated query can be fully and completely answered using ONLY the facts explicitly provided in the context. Do not generate queries about minor details, words mentioned in passing, or external knowledge."
                    )
                    synth.set_prompts(**{"query_answer_generation_prompt": prompt})

    def _get_all_prompt_mixins(transform_list: list[Any]) -> list[Any]:
        mixins = []
        for t in transform_list:
            if hasattr(t, "adapt_prompts") and hasattr(t, "set_prompts"):
                mixins.append(t)
            if hasattr(t, "transformations"):
                mixins.extend(_get_all_prompt_mixins(t.transformations))
            if hasattr(t, "transforms"):
                mixins.extend(_get_all_prompt_mixins(t.transforms))
            if hasattr(t, "transform"):
                mixins.extend(_get_all_prompt_mixins([t.transform]))
        return mixins

    transforms = None
    if config.adapt_prompts:
        from ragas.testset.transforms.default import default_transforms_for_prechunked
        transforms = default_transforms_for_prechunked(
            llm=models.llm,
            embedding_model=models.embeddings,
        )
        if config.language and config.language.lower() != "english":
            from ragas.utils import async_to_sync
            mixins = _get_all_prompt_mixins(transforms)
            for mixin in mixins:
                adapted = async_to_sync(mixin.adapt_prompts)(config.language, models.llm)
                mixin.set_prompts(**adapted)

    testset = generator.generate_with_chunks(
        chunks=documents,
        testset_size=config.testset_size,
        transforms=transforms,
        query_distribution=[
            (sh_synthesizer, config.single_hop_specific_ratio),
            (mh_spec_synthesizer, config.multi_hop_specific_ratio),
            (mh_abs_synthesizer, config.multi_hop_abstract_ratio),
        ],
        run_config=run_config,
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
        from ragas.run_config import RunConfig
    except ImportError as error:
        raise RuntimeError(
            "RAGAS generation dependencies are missing. "
            "Install them with: python -m pip install -r requirements-eval.txt"
        ) from error

    return {
        "TestsetGenerator": TestsetGenerator,
        "SingleHopSpecificQuerySynthesizer": SingleHopSpecificQuerySynthesizer,
        "MultiHopSpecificQuerySynthesizer": MultiHopSpecificQuerySynthesizer,
        "MultiHopAbstractQuerySynthesizer": MultiHopAbstractQuerySynthesizer,
        "RunConfig": RunConfig,
    }
