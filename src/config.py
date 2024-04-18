import os

from environs import Env

# Create an Env instance
env = Env()

# Read and set default values from .env file
env.read_env()

# Read values from environment variables with defaults
DATAFRAMES_FOLDER_PATH = env.str("DATAFRAMES_FOLDER_PATH", "../dataframes")
MODELS_FOLDER_PATH = env.str("MODELS_FOLDER_PATH", "../models")
DEFAULT_MODEL_FILE_PATH = env.str("DEFAULT_MODEL_FILE_PATH", f"{MODELS_FOLDER_PATH}/model")
CODE_DF_FILE_PATH = os.path.join(DATAFRAMES_FOLDER_PATH, "code.csv")
MARKDOWN_DF_FILE_PATH = os.path.join(DATAFRAMES_FOLDER_PATH, "markdown.csv")
METRICS_FOLDER_PATH = env.str("METRICS_FOLDER_PATH", "../metrics")
NOTEBOOKS_FOLDER_PATH = env.str("NOTEBOOKS_FOLDER_PATH", "../notebooks")
CODE_METRICS_DF_FILE_PATH = os.path.join(
    METRICS_FOLDER_PATH, env.str("CODE_METRICS_DF_FILE_NAME", "code_cell_metrics.csv")
)
MARKDOWN_METRICS_DF_FILE_PATH = os.path.join(
    METRICS_FOLDER_PATH, env.str("MARKDOWN_METRICS_DF_FILE_NAME", "markdown_cell_metrics.csv")
)
USER_PT_METRICS_DF_FILE_PATH = os.path.join(
    DATAFRAMES_FOLDER_PATH, env.str("USER_PT_METRICS_DF_FILE_NAME", "user_pt.csv")
)
NOTEBOOK_METRICS_DF_FILE_PATH = os.path.join(
    METRICS_FOLDER_PATH, env.str("NOTEBOOK_METRICS_DF_FILE_NAME", "notebook_metrics.csv")
)
NOTEBOOK_SCORES_DF_FILE_PATH = os.path.join(
    METRICS_FOLDER_PATH, env.str("NOTEBOOK_SCORES_DF_FILE_NAME", "notebook_scored.csv")
)
DEFAULT_NOTEBOOK_SCORES_SORT_BY = env.str("DEFAULT_NOTEBOOK_SCORES_SORT_BY", "combined_score")
CACHE_PATH = env.str("CACHE_PATH", "../cache")
CHUNK_SIZE = env.int("CHUNK_SIZE", 100000)
LIMIT_CHUNK_COUNT = env.int("LIMIT_CHUNK_COUNT", -1)
MODELS_DICT_FILE_PATH = env.str("MODELS_DICT_FILE_PATH", f"{MODELS_FOLDER_PATH}/meta.json")
DEFAULT_DIGEST_SIZE_FOR_MODEL_KEYS = env.int("DEFAULT_DIGEST_SIZE_FOR_MODEL_KEYS", 8)
