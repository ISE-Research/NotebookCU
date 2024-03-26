import logging

import pandas as pd
from tqdm import tqdm

from cell_metrics import (extract_code_metrics, extract_markdown_metrics,
                          get_eap_score_dict)
from config import (CHUNK_SIZE, CODE_DF_FILE_PATH, CODE_METRICS_DF_FILE_PATH,
                    LIMIT_CHUNK_COUNT, MARKDOWN_DF_FILE_PATH,
                    MARKDOWN_METRICS_DF_FILE_PATH)

logger = logging.getLogger(__name__)

tqdm.pandas()


def run_code_metrics_extraction():
    eap_score_dict = get_eap_score_dict(CODE_DF_FILE_PATH, chunk_size=CHUNK_SIZE)
    chunk_reader = pd.read_csv(CODE_DF_FILE_PATH, chunksize=CHUNK_SIZE)
    for i, chunk in enumerate(chunk_reader):
        if LIMIT_CHUNK_COUNT > 0 and i > LIMIT_CHUNK_COUNT:
            break
        chunk: pd.DataFrame
        logger.info(f"processing code metrics: chunksize={CHUNK_SIZE} chunk_index={i}")
        chunk.fillna("", inplace=True)
        processed_chunk = extract_code_metrics(chunk, eap_score_dict)
        logger.info(f"saving code metrics: chunksize={CHUNK_SIZE} chunk_index={i}")
        processed_chunk.to_csv(
            CODE_METRICS_DF_FILE_PATH,
            index=False,
            header=(i == 0),
            mode="w" if (i == 0) else "a",
        )


def run_md_metrics_extraction():
    chunk_reader = pd.read_csv(MARKDOWN_DF_FILE_PATH, chunksize=CHUNK_SIZE)
    for i, chunk in enumerate(chunk_reader):
        if LIMIT_CHUNK_COUNT > 0 and i > LIMIT_CHUNK_COUNT:
            break
        chunk: pd.DataFrame
        logger.info(f"processing md metrics: chunksize={CHUNK_SIZE} chunk_index={i}")
        chunk.fillna("", inplace=True)
        processed_chunk = extract_markdown_metrics(chunk)
        logger.info(f"saving md metrics: chunksize={CHUNK_SIZE} chunk_index={i}")
        processed_chunk.to_csv(
            MARKDOWN_METRICS_DF_FILE_PATH,
            index=False,
            header=(i == 0),
            mode="w" if (i == 0) else "a",
        )
