import logging

from logger import init_logger
from pathlib import Path
from enum import Enum

import typer
from typing_extensions import Annotated
from process_cell_metrics import run_code_metrics_extraction, run_markdown_metrics_extraction
import config

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True)


class FileType(str, Enum):
    markdown = "markdown"
    code = "code"


@app.command()
def extract_metrics(
        input_file_path: Annotated[Path, typer.Argument()] = config.CODE_DF_FILE_PATH,
        output_file_path: Annotated[Path, typer.Argument()] = config.CODE_METRICS_DF_FILE_PATH,
        chunk_size: Annotated[int, typer.Argument()] = config.CHUNK_SIZE,
        limit_chunk_count: Annotated[int, typer.Argument()] = config.LIMIT_CHUNK_COUNT,
        file_type: Annotated[FileType, typer.Option(case_sensitive=False)] = FileType.code,
):
    if file_type == FileType.code:
        run_code_metrics_extraction(code_df_file_path=str(input_file_path.resolve()),
                                    code_metrics_df_file_path=str(output_file_path.resolve()),
                                    chunk_size=chunk_size,
                                    limit_chunk_count=limit_chunk_count)
    elif file_type == FileType.markdown:
        run_markdown_metrics_extraction(markdown_df_file_path=str(input_file_path.resolve()),
                                        markdown_metrics_df_file_path=str(output_file_path.resolve()),
                                        chunk_size=chunk_size,
                                        limit_chunk_count=limit_chunk_count)
    else:
        raise typer.BadParameter("Invalid file type.")


if __name__ == "__main__":
    init_logger()
    app()
    # TODO: Create FastAPI Service
