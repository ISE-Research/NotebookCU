import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

import config
from classification_data import DataSelector
from classifiers import (BaseClassifier, CatBoostClassifierCustom,
                         DecisionTreeClassifierCustom,
                         RandomTreeClassifierCustom, XGBoostClassifierCustom)
from extract_metrics import extract_notebook_metrics_from_ipynb_file
from logger import init_logger
from notebook_metrics import aggregate_notebook_metrics
from process_cell_metrics import (run_code_metrics_extraction,
                                  run_markdown_metrics_extraction)
from validators import (build_extension_validator,
                        validate_metrics_filters_key,
                        validate_scores_filters_key)

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

    def get_classifier_class(self) -> BaseClassifier:
        return {
            ModelType.cat_boost: CatBoostClassifierCustom,
            ModelType.xgb_boost: XGBoostClassifierCustom,
            ModelType.decision_tree: DecisionTreeClassifierCustom,
            ModelType.random_forest: RandomTreeClassifierCustom,
        }.get(self)


@app.command()
def extract_dataframe_metrics(
    input_file_path: Annotated[
        Path,
        typer.Argument(
            help="File to process.",
            exists=True,
            dir_okay=False,
            callback=build_extension_validator([".csv"]),
        ),
    ] = Path(config.CODE_DF_FILE_PATH),
    output_file_path: Annotated[
        Path,
        typer.Argument(
            help="Desired destination path of extracted metrics.",
            callback=build_extension_validator([".csv"]),
        ),
    ] = Path(config.CODE_METRICS_DF_FILE_PATH),
    chunk_size: Annotated[
        int,
        typer.Argument(
            help="Size of chunks for processing the csv.",
            min=100,
        ),
    ] = config.CHUNK_SIZE,
    limit_chunk_count: Annotated[
        int,
        typer.Argument(
            help="Number of chunks to process (leave as is for no limit).",
            min=-1,
        ),
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
        typer.Argument(
            help="File path for code metrics dataframe.",
            exists=True,
            dir_okay=False,
            callback=build_extension_validator([".csv"]),
        ),
    ] = Path(config.CODE_METRICS_DF_FILE_PATH),
    markdown_metrics_df_file_path: Annotated[
        Path,
        typer.Argument(
            help="File path for markdown metrics dataframe.",
            exists=True,
            dir_okay=False,
            callback=build_extension_validator([".csv"]),
        ),
    ] = Path(config.MARKDOWN_METRICS_DF_FILE_PATH),
    notebook_metrics_df_file_path: Annotated[
        Path,
        typer.Argument(
            help="File path for aggregated metrics dataframe.",
            callback=build_extension_validator([".csv"]),
        ),
    ] = Path(config.NOTEBOOK_METRICS_DF_FILE_PATH),
    user_pt_metrics_df_file_path: Annotated[
        Optional[Path],
        typer.Argument(
            help="File path for PT score.",
            exists=True,
            dir_okay=False,
            callback=build_extension_validator([".csv"]),
        ),
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


@app.command()
def train_model(
    model: Annotated[
        ModelType,
        typer.Argument(
            help=f"Chosen model to be trained.",
            case_sensitive=False,
        ),  # model
    ] = ModelType.cat_boost,
    notebook_metrics_df_file_path: Annotated[
        Path,
        typer.Argument(
            help="Chosen metrics file to be used for training the model.",
            exists=True,
            dir_okay=False,
            callback=build_extension_validator([".csv"]),
        ),  # features
    ] = Path(config.NOTEBOOK_METRICS_DF_FILE_PATH),
    notebook_scores_df_file_path: Annotated[
        Path,
        typer.Argument(
            help="Chosen scores file to be used for training the model.",
            exists=True,
            dir_okay=False,
            callback=build_extension_validator([".csv"]),
        ),  # scores
    ] = Path(config.NOTEBOOK_SCORES_DF_FILE_PATH),
    model_file_path: Annotated[
        Path,
        typer.Argument(
            help="Chosen file path to store the created model.",
            callback=build_extension_validator([".csv"]),
        ),  # destination path
    ] = Path(config.DEFAULT_MODEL_FILE_PATH),
    selected_score: Annotated[
        str,
        typer.Option("--selected-score", "-ss"),  # selected score
    ] = "combined_score",
    split_factor: Annotated[
        float,
        typer.Option(
            "--split-factor",
            "-sf",
            min=0,
            max=1,
        ),
    ] = 0.7,
    selection_ratio: Annotated[
        float,
        typer.Option(
            "--selection-ratio",
            "-sr",
            min=0,
            max=1,
        ),
    ] = 0.25,
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
        typer.Option("--metrics-filters", "-ff", help="Will override filter key."),  # filter features
    ] = None,
    notebook_scores_filters: Annotated[
        Optional[str],
        typer.Option("--scores-filters", "-fs", help="Will override filter key."),  # filter scores
    ] = None,
):
    """
    Train a classifier model to decide if a jupyter notebook file is of high quality or not.
    """
    model_class = model.get_classifier_class()
    classifier = model_class.get_default_instance()

    if notebook_scores_filters is None:
        notebook_metrics_filters = DataSelector.NOTEBOOK_METRICS_FILTERS[notebook_metrics_filters_key]
    if notebook_scores_filters is None:
        notebook_scores_filters = DataSelector.NOTEBOOK_SCORES_FILTERS[notebook_scores_filters_key]

    x_train, x_test, y_train, y_test = DataSelector(
        notebook_metrics_df_file_path=str(notebook_metrics_df_file_path.resolve()),
        notebook_scores_df_file_path=str(notebook_scores_df_file_path.resolve()),
    ).get_train_test_split(
        notebook_metrics_filters=notebook_metrics_filters,
        notebook_scores_filters=notebook_scores_filters,
        sort_by=selected_score,
        split_factor=split_factor,
        selection_ratio=selection_ratio,
    )
    classifier.train(X_train=x_train, y_train=y_train)
    classifier.test(X_test=x_test, y_test=y_test)
    classifier.save_model(str(model_file_path.resolve()))


@app.command()
def extract_notebook_metrics(
    input_file_path: Annotated[
        Path,
        typer.Argument(
            help="File to process.",
            exists=True,
            dir_okay=False,
            callback=build_extension_validator([".ipynb"]),
        ),
    ],
    output_file_path: Annotated[
        Path,
        typer.Argument(
            help="Desired destination path of extracted metrics.",
            callback=build_extension_validator([".json", ".csv"]),
        ),
    ],
    base_code_df_file_path: Annotated[
        Path,
        typer.Option(
            "--base-code-df",
            "-bcd",
            help="Base code dataframe file to be used for metrics.",
            exists=True,
            dir_okay=False,
            callback=build_extension_validator([".csv"]),
        ),
    ] = Path(config.CODE_DF_FILE_PATH),
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", "-cs", help="Size of chunks for processing the base code df csv."),
    ] = config.CHUNK_SIZE,
):
    """
    Extract metrics of a jupyter notebook file.
    """
    extracted_notebook_metrics_df = extract_notebook_metrics_from_ipynb_file(
        file_path=str(input_file_path.resolve()),
        base_code_df_file_path=str(base_code_df_file_path.resolve()),
        chunk_size=chunk_size,
    )
    print(extracted_notebook_metrics_df.to_string())
    if output_file_path.suffix == ".csv":
        extracted_notebook_metrics_df.to_csv(str(output_file_path.resolve()))
    elif output_file_path.suffix == ".json":
        with open(str(output_file_path.resolve()), "w") as file:
            file.write(extracted_notebook_metrics_df.iloc[[0]].to_json(orient="records", indent=4))
    else:
        raise typer.BadParameter("Invalid file extension. Only csv and json files supported")


@app.command()
def predict(
    input_file_path: Annotated[
        Path,
        typer.Argument(
            help="File to process.",
            exists=True,
            dir_okay=False,
            callback=build_extension_validator([".csv"]),
        ),
    ],
    model: Annotated[
        ModelType,
        typer.Argument(
            help=f"Chosen model to be trained.",
            case_sensitive=False,
        ),  # model
    ] = ModelType.cat_boost,
    selected_model_path: Annotated[
        Path,
        typer.Argument(
            help="Selected classifier model file path.",
            exists=True,
            dir_okay=False,
        ),
    ] = Path(config.DEFAULT_MODEL_FILE_PATH),
    base_code_df_file_path: Annotated[
        Path,
        typer.Option(
            "--base-code-df",
            "-bcd",
            help="Base code dataframe file to be used for metrics.",
            exists=True,
            dir_okay=False,
            callback=build_extension_validator([".csv"]),
        ),
    ] = Path(config.CODE_DF_FILE_PATH),
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", "-cs", help="Size of chunks for processing the base code df csv."),
    ] = config.CHUNK_SIZE,
):
    """
    Predict if a jupyter notebook file is high quality based on selected metrics.
    """
    # TODO: check validity of file type and content before extracting the metrics
    extracted_notebook_metrics_df = extract_notebook_metrics_from_ipynb_file(
        file_path=str(input_file_path.resolve()),
        base_code_df_file_path=str(base_code_df_file_path.resolve()),
        chunk_size=chunk_size,
    )
    print(extracted_notebook_metrics_df.to_string())

    model_class = model.get_classifier_class()
    classifier: BaseClassifier = model_class()
    classifier.load_model(str(selected_model_path.resolve()))

    # TODO: standardize column names
    extracted_notebook_metrics_df.drop(["kernel_id"], axis=1, inplace=True)
    extracted_notebook_metrics_df.rename(
        columns={
            "ALLC": "ALLCL",
        },
        inplace=True,
    )
    # TODO: remove PT or take as input
    extracted_notebook_metrics_df["PT"] = 10

    result = classifier.predict(x=extracted_notebook_metrics_df)
    print(f"result: {result}")


if __name__ == "__main__":
    init_logger()
    app()
    # TODO: create FastAPI service
