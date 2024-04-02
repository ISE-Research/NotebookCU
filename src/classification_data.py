import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)


tqdm.pandas()


class DataSelector:
    NOTEBOOK_METRICS_FILTERS = {
        "default": ["LOC < 1000"],
        "ML_based": [
            "TYPE == ML",
            "LOC < 1000",
        ],
    }

    NOTEBOOK_SCORES_FILTERS = {
        "default": ["TotalViews >= 500"],
        "ML_based": [
            "TYPE == ML",
            "TotalViews >= 500",
        ],
    }

    def __init__(self, notebook_metrics_df_file_path: str, notebook_scores_df_file_path: str):
        logger.info("Going to load features and scores...")
        self.notebook_metrics_df = pd.read_csv(notebook_metrics_df_file_path)
        self.notebook_metrics_df.drop(["Unnamed: 0"], axis=1, inplace=True)
        self.notebook_metrics_df = self.notebook_metrics_df.drop_duplicates()
        self.notebook_scores_df = pd.read_csv(notebook_scores_df_file_path)
        logger.info("Loaded features and scores successfully.")

    def get_train_test_split(
        self,
        notebook_metrics_filters: list = [],
        notebook_scores_filters: list = [],
        split_factor: float = 0.7,
        sort_by: str = config.DEFAULT_NOTEBOOK_SCORES_SORT_BY,
    ):
        features_df = self.apply_filters(self.notebook_metrics_df, notebook_metrics_filters)
        scores_df = self.apply_filters(self.notebook_scores_df, notebook_scores_filters)
        return self._split_data(
            features_df=features_df, scores_df=scores_df, split_factor=split_factor, sort_by=sort_by
        )

    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: list) -> pd.DataFrame:
        for filter in filters:
            df = df.query(filter)
        return df

    def _prepare_data(self, features_df: pd.DataFrame, scores_df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
        logger.info(f"features_df:\n{features_df.columns}\n{features_df.shape}\n")
        features_df.info()

        # TODO: standardize column names
        scores_df = scores_df[
            [
                "KernelId",
                "TotalViews",
                "TotalVotes",
                "PerformanceTier_kerneluser",
                "topic_score",
                "score_scaled",
                "vote_scaled",
                "combined_score",
            ]
        ]
        logger.info(f"scores_df:\n{scores_df.columns}\n{scores_df.shape}")
        scores_df.info()

        logger.info("Going to merge features and scores...")

        merged_df = pd.merge(
            scores_df,
            features_df.rename(columns={"project_ID": "KernelId"}),
            on="KernelId",
            how="inner",
        )

        logger.info("Going to sort merged_df...")
        merged_df.sort_values(by=[sort_by], inplace=True)

        # Drop scores and unique id fields
        merged_df.drop(
            [
                "KernelId",
                "PerformanceTier_kerneluser",
                "TotalViews",
                "TotalVotes",
                "topic_score",
                "score_scaled",
                "vote_scaled",
                "combined_score",
            ],
            axis=1,
            inplace=True,
        )

        # TODO: standardize column names
        merged_df.rename(
            columns={
                "LOCOM": "LOCom",
                "ALLC": "ALLCL",
                "NOI": "I",
                "MedLC": "MedLCC",
                "MeanLC": "MeanLCC",
                "BLOC": "BLC",
                "NOEH": "EH",
                "NExR": "NEC",
                "MedLMD": "MedLMC",
                "MeanLMD": "MeanLMC",
                "visulization": "VF",
                "BLOMD": "BLM",
                "MCEL": "MC",
                "DeF": "UDF",
                "WOMD": "MeanWMC",
                "CCEL": "CC",
                "output_stream_len": "OSL",
                "NDD": "NVD",
            },
            inplace=True,
        )

        logger.info(f"merged_df:\n{merged_df.columns}\n{merged_df.shape}")
        merged_df.info()
        return merged_df

    def _split_data(
        self,
        features_df: pd.DataFrame,
        scores_df: pd.DataFrame,
        split_factor: float = 0.7,
        sort_by: str = config.DEFAULT_NOTEBOOK_SCORES_SORT_BY,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, list, list]:
        notebooks_sorted_by_score = self._prepare_data(features_df=features_df, scores_df=scores_df, sort_by=sort_by)

        logger.info(f"get_train_test_split df.shape: {notebooks_sorted_by_score.shape}")
        X = notebooks_sorted_by_score.copy()
        zeros_len = int(split_factor * len(X))
        X["topic_qualities"] = [0 for _ in range(zeros_len)] + [1 for _ in range(len(X) - zeros_len)]
        le = len(X)
        q1 = int(0.25 * (split_factor) * le)
        q2 = int(0.25 * (1 - split_factor) * le)

        X = pd.concat([X.head(q1), X.tail(q2)])
        y = list(X["topic_qualities"])
        X.drop(["topic_qualities"], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        logger.info(f"X_train.shape:{X_train.shape}")
        logger.info(f"X_test.shape:{X_test.shape}")
        logger.info(f"len(y_train):{len(y_train)}")
        logger.info(f"len(y_test):{len(y_test)}")
        return (X_train, X_test, y_train, y_test)
