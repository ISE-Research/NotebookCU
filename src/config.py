import os

from environs import Env

# Create an Env instance
env = Env()

# Read and set default values from .env file
env.read_env()

# Read values from environment variables with defaults
DATAFRAMES_FOLDER_PATH = env.str("DATAFRAMES_FOLDER_PATH", "../dataframes")
MODELS_FOLDER_PATH = env.str("MODELS_FOLDER_PATH", "../models")
CODE_DF_FILE_PATH = os.path.join(DATAFRAMES_FOLDER_PATH, "code.csv")
MARKDOWN_DF_FILE_PATH = os.path.join(DATAFRAMES_FOLDER_PATH, "markdown.csv")
METRICS_FOLDER_PATH = env.str("METRICS_FOLDER_PATH", "../metrics")
CODE_METRICS_DF_FILE_PATH = os.path.join(METRICS_FOLDER_PATH, "code_cell_metrics.csv")
MARKDOWN_METRICS_DF_FILE_PATH = os.path.join(METRICS_FOLDER_PATH, "markdown_cell_metrics.csv")
USER_PT_METRICS_DF_FILE_PATH = os.path.join(DATAFRAMES_FOLDER_PATH, "user_pt.csv")
NOTEBOOK_METRICS_DF_FILE_PATH = os.path.join(METRICS_FOLDER_PATH, "notebook_metrics.csv")
CACHE_PATH = env.str("CACHE_PATH", "../cache")
CHUNK_SIZE = env.int("CHUNK_SIZE", 100000)
LIMIT_CHUNK_COUNT = env.int("LIMIT_CHUNK_COUNT", -1)
