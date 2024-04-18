from pathlib import Path
from typing import List

import typer

from classification_data import DataSelector


def build_extension_validator(
    valid_extensions: List[str],
    nullable: bool = False,
):
    def validate_extension(value: Path) -> Path:
        if nullable and value is None:
            return value        
        extension = value.suffix.lower()
        if extension not in valid_extensions:
            raise typer.BadParameter(
                f"Invalid file extension: '{extension}', valid extensions: '{', '.join(valid_extensions)}'.",
            )

        return value

    return validate_extension


def validate_metrics_filters_key(value: str) -> str:
    if value not in DataSelector.NOTEBOOK_METRICS_FILTERS.keys():
        raise typer.BadParameter(f"valid options: {', '.join(DataSelector.NOTEBOOK_METRICS_FILTERS.keys())}")
    return value


def validate_scores_filters_key(value: str) -> str:
    if value not in DataSelector.NOTEBOOK_SCORES_FILTERS.keys():
        raise typer.BadParameter(f"valid options: {', '.join(DataSelector.NOTEBOOK_SCORES_FILTERS.keys())}")
    return value
