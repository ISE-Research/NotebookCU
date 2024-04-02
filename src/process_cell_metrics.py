import logging

import pandas as pd
from tqdm import tqdm

import config
from cell_metrics import (extract_code_metrics, extract_markdown_metrics,
                          get_eap_score_dict)

logger = logging.getLogger(__name__)

tqdm.pandas()


def run_code_metrics_extraction(code_df_file_path: str = config.CODE_DF_FILE_PATH,
                                code_metrics_df_file_path: str = config.CODE_METRICS_DF_FILE_PATH,
                                chunk_size: int = config.CHUNK_SIZE,
                                limit_chunk_count: int = config.LIMIT_CHUNK_COUNT):
    eap_score_dict = get_eap_score_dict(code_df_file_path, chunk_size=chunk_size)
    chunk_reader = pd.read_csv(code_df_file_path, chunksize=chunk_size)
    for i, chunk in enumerate(chunk_reader):
        if 0 < limit_chunk_count < i:
            break
        chunk: pd.DataFrame
        logger.info(f"processing code metrics: chunksize={chunk_size} chunk_index={i}")
        chunk.fillna("", inplace=True)
        processed_chunk = extract_code_metrics(chunk, eap_score_dict)
        logger.info(f"saving code metrics: chunksize={chunk_size} chunk_index={i}")
        processed_chunk.to_csv(
            code_metrics_df_file_path,
            index=False,
            header=(i == 0),
            mode="w" if (i == 0) else "a",
        )


def run_markdown_metrics_extraction(markdown_df_file_path: str = config.MARKDOWN_DF_FILE_PATH,
                                    markdown_metrics_df_file_path: str = config.MARKDOWN_METRICS_DF_FILE_PATH,
                                    chunk_size: int = config.CHUNK_SIZE,
                                    limit_chunk_count: int = config.LIMIT_CHUNK_COUNT):
    chunk_reader = pd.read_csv(markdown_df_file_path, chunksize=chunk_size)
    for i, chunk in enumerate(chunk_reader):
        if 0 < limit_chunk_count < i:
            break
        chunk: pd.DataFrame
        logger.info(f"processing md metrics: chunksize={chunk_size} chunk_index={i}")
        chunk.fillna("", inplace=True)
        processed_chunk = extract_markdown_metrics(chunk)
        logger.info(f"saving md metrics: chunksize={chunk_size} chunk_index={i}")
        processed_chunk.to_csv(
            markdown_metrics_df_file_path,
            index=False,
            header=(i == 0),
            mode="w" if (i == 0) else "a",
        )
