import logging
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import typer
from typing_extensions import Annotated

import config
from classification_data import DataSelector
from classifiers import (
    CatBoostClassifierCustom,
    DecisionTreeClassifierCustom,
    RandomTreeClassifierCustom,
    XGBoostClassifierCustom,
)
from logger import init_logger
from notebook_metrics import aggregate_notebook_metrics
from process_cell_metrics import run_code_metrics_extraction, run_markdown_metrics_extraction

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True)


class FileType(str, Enum):
    markdown = "markdown"
    code = "code"

class ModelType(str, Enum):
    cat_boost = "cat_boost"
    xgb_boost = "xgb_boost"
    decision_tree = "decision_tree"
    random_forest = "random_forest"

    def get_classifier_class(self):
        return {
            ModelType.cat_boost: CatBoostClassifierCustom,
            ModelType.xgb_boost: XGBoostClassifierCustom,
            ModelType.decision_tree: DecisionTreeClassifierCustom,
            ModelType.random_forest: RandomTreeClassifierCustom,
        }.get(self.value)


@app.command()
def extract_metrics(
    input_file_path: Annotated[
        Path,
        typer.Argument(help="File to process."),
    ] = Path(config.CODE_DF_FILE_PATH),
    output_file_path: Annotated[
        Path,
        typer.Argument(help="Desired destination path of extracted metrics."),
    ] = Path(config.CODE_METRICS_DF_FILE_PATH),
    chunk_size: Annotated[
        int,
        typer.Argument(help="Size of chunks for processing the csv."),
    ] = config.CHUNK_SIZE,
    limit_chunk_count: Annotated[
        int,
        typer.Argument(help="Number of chunks to process (leave as is for no limit)."),
    ] = config.LIMIT_CHUNK_COUNT,
    file_type: Annotated[
        FileType,
        typer.Option(case_sensitive=False),
    ] = FileType.code,
):
    """
    Extract metrics of notebook blocks gathered in a csv file.

    Use --file-type to decide the type of extraction
    """
    if file_type == FileType.code:
        run_code_metrics_extraction(
            code_df_file_path=str(input_file_path.resolve()),
            code_metrics_df_file_path=str(output_file_path.resolve()),
            chunk_size=chunk_size,
            limit_chunk_count=limit_chunk_count,
        )
    elif file_type == FileType.markdown:
        run_markdown_metrics_extraction(
            markdown_df_file_path=str(input_file_path.resolve()),
            markdown_metrics_df_file_path=str(output_file_path.resolve()),
            chunk_size=chunk_size,
            limit_chunk_count=limit_chunk_count,
        )


@app.command()
def aggregate_metrics(
    code_metrics_df_file_path: Annotated[
        Path,
        typer.Argument(),
    ] = Path(config.CODE_METRICS_DF_FILE_PATH),
    markdown_metrics_df_file_path: Annotated[
        Path,
        typer.Argument(),
    ] = Path(config.MARKDOWN_METRICS_DF_FILE_PATH),
    notebook_metrics_df_file_path: Annotated[
        Path,
        typer.Argument(),
    ] = Path(config.NOTEBOOK_METRICS_DF_FILE_PATH),
    user_pt_metrics_df_file_path: Annotated[
        Optional[Path],
        typer.Argument(),
    ] = None,
):
    """
    Aggregate code metrics and markdown metrics to get the notebook metrics dataframe.

    user_pt_metrics_df_file_path is optional
    """
    aggregate_notebook_metrics(
        code_metrics_df_file_path=str(code_metrics_df_file_path.resolve()),
        markdown_metrics_df_file_path=str(markdown_metrics_df_file_path.resolve()),
        notebook_metrics_df_file_path=str(notebook_metrics_df_file_path.resolve()),
        user_pt_metrics_df_file_path=(
            str(user_pt_metrics_df_file_path.resolve()) if user_pt_metrics_df_file_path else None
        ),
    )


def validate_metrics_filters_key(value: str) -> str:
    if value not in DataSelector.NOTEBOOK_METRICS_FILTERS.keys():
        raise typer.BadParameter(f"valid options: {', '.join(DataSelector.NOTEBOOK_METRICS_FILTERS.keys())}")
    return value


def validate_scores_filters_key(value: str) -> str:
    if value not in DataSelector.NOTEBOOK_SCORES_FILTERS.keys():
        raise typer.BadParameter(f"valid options: {', '.join(DataSelector.NOTEBOOK_SCORES_FILTERS.keys())}")
    return value


@app.command()
def train_model(
    model: Annotated[
        ModelType,
        typer.Option("--model", "-m"),  # model
    ] = ModelType.cat_boost,
    notebook_metrics_df_file_path: Annotated[
        Path,
        typer.Option("--notebook-metrics-df-file-path", "-f"),  # features
    ] = Path(config.NOTEBOOK_METRICS_DF_FILE_PATH),
    notebook_scores_df_file_path: Annotated[
        Path,
        typer.Option("--notebook-scores-df-file-path", "-s"),  # scores
    ] = Path(config.NOTEBOOK_SCORES_DF_FILE_PATH),
    model_file_path: Annotated[
        Path,
        typer.Option("--folder-path-to-store-model", "-d"),  # destination folder
    ] = Path(config.DEFAULT_MODEL_FILE_PATH),
    selected_score: Annotated[
        str,
        typer.Option("--selected-score", "-ss"),  # selected score
    ] = "combined_score",
    notebook_metrics_filters_key: Annotated[
        str,
        typer.Option(
            "--metrics-filters-key",
            "-ffk",  # filter features key
            callback=validate_metrics_filters_key,
            help=f"Predefined key to filter notebook metrics based "
            f"(valid options: {', '.join(DataSelector.NOTEBOOK_METRICS_FILTERS.keys())})",
        ),
    ] = "default",
    notebook_scores_filters_key: Annotated[
        str,
        typer.Option(
            "--scores-filters-key",
            "-fsk",  # filter scores key
            callback=validate_scores_filters_key,
            help=f"Predefined key to filter notebook metrics based "
            f"(valid options: {', '.join(DataSelector.NOTEBOOK_SCORES_FILTERS.keys())})",
        ),
    ] = "default",
    notebook_metrics_filters: Annotated[
        Optional[str],
        typer.Option("--metrics-filters", "-ff"),  # filter features
    ] = None,
    notebook_scores_filters: Annotated[
        Optional[str],
        typer.Option("--scores-filters", "-fs"),  # filter scores
    ] = None,
):
    model_class = model.get_classifier_class()
    classifier = model_class.get_default_instance()

    notebook_metrics_filters = DataSelector.NOTEBOOK_METRICS_FILTERS[notebook_metrics_filters_key]
    notebook_scores_filters = DataSelector.NOTEBOOK_SCORES_FILTERS[notebook_scores_filters_key]

    X_train, X_test, y_train, y_test = DataSelector(
        notebook_metrics_df_file_path=str(notebook_metrics_df_file_path.resolve()),
        notebook_scores_df_file_path=str(notebook_scores_df_file_path.resolve()),
    ).get_train_test_split(
        notebook_metrics_filters=notebook_metrics_filters,
        notebook_scores_filters=notebook_scores_filters,
        sort_by=selected_score,
    )
    classifier.train(X_train=X_train, y_train=y_train)
    classifier.test(X_test=X_test, y_test=y_test)
    classifier.save_model(str(model_file_path.resolve()))


if __name__ == "__main__":
    init_logger()
    app()
    # TODO: Create FastAPI Service
