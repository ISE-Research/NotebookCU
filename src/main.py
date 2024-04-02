import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

import config
from logger import init_logger
from notebook_metrics import aggregate_notebook_metrics
from process_cell_metrics import (run_code_metrics_extraction,
                                  run_markdown_metrics_extraction)

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True)


class FileType(str, Enum):
    markdown = "markdown"
    code = "code"


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
    notebook_metrics_df_file_path:Annotated[
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
        user_pt_metrics_df_file_path=str(user_pt_metrics_df_file_path.resolve()) if user_pt_metrics_df_file_path else None
    )

if __name__ == "__main__":
    init_logger()
    app()
    # TODO: Create FastAPI Service
