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
            "RAGAS generation dependencies are missing. "
            "Install them with: python -m pip install -r requirements-eval.txt"
        ) from error

    return {
        "TestsetGenerator": TestsetGenerator,
        "SingleHopSpecificQuerySynthesizer": SingleHopSpecificQuerySynthesizer,
        "MultiHopSpecificQuerySynthesizer": MultiHopSpecificQuerySynthesizer,
        "MultiHopAbstractQuerySynthesizer": MultiHopAbstractQuerySynthesizer,
    }
