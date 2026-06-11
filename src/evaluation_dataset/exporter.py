from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {"question", "contexts", "ground_truth"}
COLUMN_ALIASES = {
    "user_input": "question",
    "reference_contexts": "contexts",
    "reference": "ground_truth",
}


def normalize_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.rename(columns=_available_aliases(frame)).copy()

    missing = REQUIRED_COLUMNS.difference(normalized.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Generated dataset is missing required columns: {missing_columns}")

    for column in REQUIRED_COLUMNS:
        normalized[column] = normalized[column].apply(_require_non_empty)
    return normalized


def export_dataset(frame: pd.DataFrame, output_path: Path, output_format: str) -> None:
    normalized = normalize_dataset(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "jsonl":
        normalized.to_json(output_path, orient="records", lines=True, force_ascii=False)
        return

    if output_format == "csv":
        normalized.to_csv(output_path, index=False, encoding="utf-8")
        return

    raise ValueError("output_format must be either 'jsonl' or 'csv'")


def _available_aliases(frame: pd.DataFrame) -> dict[str, str]:
    return {
        source: target
        for source, target in COLUMN_ALIASES.items()
        if source in frame.columns and target not in frame.columns
    }


def _require_non_empty(value: Any) -> Any:
    if value is None:
        raise ValueError("Dataset contains null values in required columns")
    if isinstance(value, str) and not value.strip():
        raise ValueError("Dataset contains blank strings in required columns")
    if isinstance(value, list) and not value:
        raise ValueError("Dataset contains empty context lists")
    return value
