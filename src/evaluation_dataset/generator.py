from typing import Any

import pandas as pd

from src.evaluation_dataset.config import EvaluationDatasetConfig
from src.evaluation_dataset.model_provider import build_ragas_models


def _run_transforms_stage(
    documents: list[Any],
    config: EvaluationDatasetConfig,
    run_config: Any,
) -> Any:
    # Build models freshly inside this scope. When this function exits,
    # all models, LLM/embeddings clients, and connection pools will be garbage collected.
    from ragas.testset.transforms.default import default_transforms_for_prechunked
    from ragas.testset.transforms import apply_transforms
    from ragas.testset.graph import Node, NodeType, KnowledgeGraph

    models = build_ragas_models(config)

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

    # Build the nodes list exactly like Ragas
    nodes = []
    for chunk in documents:
        if isinstance(chunk, str):
            page_content = chunk
            metadata = {}
        else:
            page_content = chunk.page_content
            metadata = chunk.metadata

        if page_content is not None and page_content.strip() != "":
            node = Node(
                type=NodeType.CHUNK,
                properties={
                    "page_content": page_content,
                    "document_metadata": metadata,
                },
            )
            nodes.append(node)

    kg = KnowledgeGraph(nodes=nodes)

    transforms = default_transforms_for_prechunked(
        llm=models.llm,
        embedding_model=models.embeddings,
    )

    if config.adapt_prompts and config.language and config.language.lower() != "english":
        from ragas.utils import async_to_sync
        mixins = _get_all_prompt_mixins(transforms)
        for mixin in mixins:
            adapted = async_to_sync(mixin.adapt_prompts)(config.language, models.llm)
            mixin.set_prompts(**adapted)

    apply_transforms(kg, transforms, run_config=run_config)
    return kg


def generate_ragas_testset(
    documents: list[Any],
    config: EvaluationDatasetConfig,
    models: Any = None,
) -> pd.DataFrame:
    config.validate()

    imports = _load_ragas_testset_imports()
    run_config = imports["RunConfig"](
        timeout=config.ragas_run_timeout,
        max_retries=config.ragas_max_retries,
        max_workers=config.ragas_max_workers,
    )

    # 1. Run the transform stage. All HTTP clients/loops from this stage
    # are completely destroyed and collected once the helper returns.
    kg = _run_transforms_stage(documents, config, run_config)

    # 2. Build a fresh models instance for the generation stage.
    # This prevents reusing any connection pools that were bound to the closed transform loop.
    models = build_ragas_models(config)

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

    # Check relationships for clustering potential
    has_overlap_rel = any(rel.type == "entities_overlap" for rel in kg.relationships)
    has_similarity_rel = any(rel.get_property("summary_similarity") is not None for rel in kg.relationships)

    query_distribution = [
        (sh_synthesizer, config.single_hop_specific_ratio),
        (mh_spec_synthesizer, config.multi_hop_specific_ratio),
        (mh_abs_synthesizer, config.multi_hop_abstract_ratio),
    ]

    if not has_overlap_rel or not has_similarity_rel:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "Not enough semantic relationships ('entities_overlap' or 'summary_similarity') found in the knowledge graph. "
            "Adjusting query distribution to 100% single-hop queries to avoid generation failure."
        )
        query_distribution = [
            (sh_synthesizer, 1.0),
        ]

    generator.knowledge_graph = kg
    testset = generator.generate(
        testset_size=config.testset_size,
        query_distribution=query_distribution,
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
