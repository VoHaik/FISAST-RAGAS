import json

import pandas as pd
import pytest

from src.evaluation_dataset.exporter import export_dataset, normalize_dataset


def test_normalize_dataset_requires_core_columns():
    frame = pd.DataFrame([{"question": "Q?", "ground_truth": "A"}])

    with pytest.raises(ValueError, match="contexts"):
        normalize_dataset(frame)


def test_normalize_dataset_maps_ragas_columns_to_project_schema():
    frame = pd.DataFrame(
        [
            {
                "user_input": "Who led the resistance?",
                "reference_contexts": ["Tran Hung Dao led the resistance."],
                "reference": "Tran Hung Dao.",
            }
        ]
    )

    normalized = normalize_dataset(frame)

    assert normalized.loc[0, "question"] == "Who led the resistance?"
    assert normalized.loc[0, "contexts"] == ["Tran Hung Dao led the resistance."]
    assert normalized.loc[0, "ground_truth"] == "Tran Hung Dao."


def test_export_dataset_writes_jsonl(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "question": "Who led the resistance?",
                "contexts": ["Tran Hung Dao led the resistance."],
                "ground_truth": "Tran Hung Dao.",
            }
        ]
    )
    output_path = tmp_path / "dataset.jsonl"

    export_dataset(frame, output_path, "jsonl")

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["question"] == "Who led the resistance?"
    assert rows[0]["ground_truth"] == "Tran Hung Dao."


def test_normalize_dataset_replaces_null_with_string_none():
    frame = pd.DataFrame(
        [
            {
                "question": "Q?",
                "contexts": ["C"],
                "ground_truth": "A",
                "persona_name": None,
                "query_style": None,
            }
        ]
    )
    normalized = normalize_dataset(frame)
    assert normalized.loc[0, "persona_name"] == "None"
    assert normalized.loc[0, "query_style"] == "None"
