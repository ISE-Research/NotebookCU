import json
import logging
from typing import Tuple

import pandas as pd
from tqdm import tqdm

import utils.config as config
from core.cell_metrics import extract_code_metrics, extract_markdown_metrics, get_eap_score_dict
from core.notebook_metrics import get_aggregated_notebook_metrics

logger = logging.getLogger(__name__)

tqdm.pandas()


def extract_cell_data_from_ipynb_file(
    file_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with open(file_path, encoding="utf8") as f:
        data = json.load(f)

    cells_count = 0

    df_codes = pd.DataFrame(
        columns=[
            "kernel_id",
            "cell_index",
            "source",
            "output_type",
            "execution_count",
        ]
    )
    df_markdowns = pd.DataFrame(columns=["kernel_id", "cell_index", "source"])

    for cell in data["cells"]:
        cells_count += 1
        string_src = cell["source"]
        if type(string_src) is list:
            string_src = "".join(string_src)

        if cell["cell_type"] == "code":
            try:
                output_type = cell["outputs"][0]["output_type"]
            except (KeyError, IndexError):
                output_type = None

            df_tmp = pd.DataFrame(
                {
                    "kernel_id": 1,
                    "cell_index": cells_count,
                    "source": string_src,
                    "output_type": output_type,
                    "execution_count": cell["execution_count"],
                },
                index=[0],
            )
            df_codes = pd.concat([df_codes, df_tmp], ignore_index=True, axis=0)

        elif cell["cell_type"] == "markdown":
            df_tmp = pd.DataFrame(
                {
                    "kernel_id": 1,
                    "cell_index": cells_count,
                    "source": string_src,
                },
                index=[0],
            )
            df_markdowns = pd.concat([df_markdowns, df_tmp], ignore_index=True, axis=0)
        else:
            raise Exception("Unknown cell")

    return df_codes, df_markdowns


def extract_cell_metrics_from_ipynb_file(
    file_path: str,
    base_code_df_file_path: str,
    chunk_size: int = config.CHUNK_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_codes, df_markdowns = extract_cell_data_from_ipynb_file(file_path)
    eap_score_dict = get_eap_score_dict(base_code_df_file_path, chunk_size=chunk_size)

    df_codes_metrics: pd.DataFrame = extract_code_metrics(df_codes, eap_score_dict)
    df_markdowns_metrics: pd.DataFrame = extract_markdown_metrics(df_markdowns)

    return df_codes_metrics, df_markdowns_metrics


def extract_notebook_metrics_from_ipynb_file(
    file_path: str,
    base_code_df_file_path: str,
    chunk_size: int = config.CHUNK_SIZE,
) -> pd.DataFrame:
    df_codes_metrics, df_markdowns_metrics = extract_cell_metrics_from_ipynb_file(
        file_path,
        base_code_df_file_path,
        chunk_size,
    )
    # TODO: standardize column names
    return get_aggregated_notebook_metrics(
        df_codes_metrics,
        df_markdowns_metrics,
    )
