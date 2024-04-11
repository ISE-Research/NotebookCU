import logging
from typing import Optional

import pandas as pd
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

tqdm.pandas()


def safe_convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return None


def clean_kernel_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.kernel_id != "train_emb"]
    df.kernel_id = df["kernel_id"].apply(safe_convert_to_int)
    nan_count = df["kernel_id"].isna().sum()
    logger.info(f"NaN kernel_id column count: {nan_count}")
    df.fillna(0, inplace=True)
    return df


def clean_cell_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.kernel_id != "train_emb"]
    df["cell_index"] = pd.to_numeric(df["cell_index"], errors="coerce")
    df["cell_index"] = df["cell_index"].apply(safe_convert_to_int)
    nan_count = df["cell_index"].isna().sum()
    logger.info(f"NaN cell_index column count: {nan_count}")
    df.fillna(0, inplace=True)
    return df


def get_aggregated_code_cell_metrics(
    code_df: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Going to get_aggregated_code_cell_metrics...")
    pm1 = (
        code_df[
            [
                "kernel_id",
                "LOC",
                "BLC",
                "UDF",
                "I",
                "EH",
                "NVD",
                "NEC",
                "S",
                "P",
                "OPRND",
                "OPRATOR",
                "UOPRND",
                "UOPRATOR",
                "ID",
                "LOCom",
                "EAP",
                "CW",
            ]
        ]
        .groupby(by="kernel_id")
        .sum()
        .reset_index()
    )
    pm2 = code_df[["kernel_id", "ALID", "LOC", "ALLC", "KLCID", "CyC"]].groupby(by="kernel_id").mean().reset_index()
    pm2.rename(columns={"LOC": "MeanLCC"}, inplace=True)
    pm3 = (
        code_df[["kernel_id", "MLID", "NBD", "ID"]]
        .groupby(by="kernel_id")
        .max()
        .reset_index()
        .rename(columns={"ID": "AID"})
    )
    pm4 = code_df[["kernel_id", "LOC"]].groupby("kernel_id").count().reset_index().rename(columns={"LOC": "CC"})
    del code_df

    result_df = pd.concat(
        [
            pm1,
            pm2.drop(columns=["kernel_id"]),
            pm3.drop(columns=["kernel_id"]),
            pm4.drop(columns=["kernel_id"]),
        ],
        axis=1,
    )
    result_df = clean_kernel_id(result_df)
    logger.info("Cleaned code_df.")
    result_df = result_df[result_df["LOC"] > 1]
    return result_df


def get_aggregated_markdown_cell_metrics(
    markdown_df: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Going to get_aggregated_markdown_cell_metrics...")
    qm1 = markdown_df[["kernel_id", "H1"]].groupby("kernel_id").count().reset_index().rename(columns={"H1": "MC"})
    qm2 = (
        markdown_df[["kernel_id", "MW", "LMC"]]
        .groupby("kernel_id")
        .mean()
        .reset_index()
        .rename(columns={"MW": "MeanWMC", "LMC": "MeanLMC"})
    )
    qm3 = markdown_df[["kernel_id", "H1", "H2", "H3", "MW", "LMC"]].groupby("kernel_id").sum().reset_index()
    del markdown_df

    result_df = pd.concat(
        [qm1, qm2.drop(columns=["kernel_id"]), qm3.drop(columns=["kernel_id"])],
        axis=1,
    )
    result_df = clean_kernel_id(result_df)
    logger.info("Cleaned md_df")
    return result_df


def get_aggregated_notebook_metrics(
    code_cell_metrics_df: pd.DataFrame,
    markdown_cell_metrics_df: pd.DataFrame,
    user_pt_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    aggregated_code_cell_metrics_df = get_aggregated_code_cell_metrics(code_df=code_cell_metrics_df)
    aggregated_markdown_cell_metrics_df = get_aggregated_markdown_cell_metrics(markdown_df=markdown_cell_metrics_df)
    logger.info("Going to merge...")
    notebook_metrics_df = aggregated_code_cell_metrics_df.merge(aggregated_markdown_cell_metrics_df, how="left").fillna(
        0
    )
    if user_pt_df is not None:
        notebook_metrics_df = notebook_metrics_df.merge(user_pt_df, how="left").fillna(0)
        notebook_metrics_df.kernel_id = notebook_metrics_df["kernel_id"].astype(int)
    return notebook_metrics_df


def aggregate_notebook_metrics(
    code_metrics_df_file_path: str = config.CODE_METRICS_DF_FILE_PATH,
    markdown_metrics_df_file_path: str = config.MARKDOWN_METRICS_DF_FILE_PATH,
    notebook_metrics_df_file_path: str = config.NOTEBOOK_METRICS_DF_FILE_PATH,
    user_pt_metrics_df_file_path: Optional[str] = None,
):
    logger.info("Going to get_aggregated_code_cell_metrics...")

    code_df = pd.read_csv(code_metrics_df_file_path)
    markdown_df = pd.read_csv(markdown_metrics_df_file_path)
    user_pt_df = None
    if user_pt_metrics_df_file_path is not None:
        user_pt_df = (
            pd.read_csv(user_pt_metrics_df_file_path)
            .rename(columns={"Id_x": "kernel_id", "PerformanceTier": "PT"})
            .drop(columns=["Unnamed: 0"])
        )
    logger.info("Loaded files successfully.")

    notebook_metrics_df = get_aggregated_notebook_metrics(
        code_cell_metrics_df=code_df,
        markdown_cell_metrics_df=markdown_df,
        user_pt_df=user_pt_df,
    )

    logger.info("Going to save...")
    notebook_metrics_df.info()
    logger.info(f"Shape:{notebook_metrics_df.shape}")
    notebook_metrics_df.to_csv(notebook_metrics_df_file_path)
    logger.info("Saved Successfully.")
