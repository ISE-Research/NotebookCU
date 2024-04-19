import logging
from typing import Optional, Tuple

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

    def __init__(
        self,
        notebook_metrics_df_file_path: str,
        notebook_scores_df_file_path: str,
        expert_scores_df_file_path: Optional[str],
    ):
        logger.info("Going to load features and scores...")
        self.notebook_metrics_df = pd.read_csv(notebook_metrics_df_file_path, low_memory=False)
        logger.info(
            f"self.notebook_metrics_df:\n"
            f"columns: {self.notebook_metrics_df.columns}\n"
            f"shape: {self.notebook_metrics_df.shape}\n"
        )
        self.notebook_scores_df = pd.read_csv(notebook_scores_df_file_path)
        logger.info(
            f"self.notebook_scores_df:\n"
            f"columns: {self.notebook_scores_df.columns}\n"
            f"shape: {self.notebook_scores_df.shape}\n"
        )
        self.experts_scores_df = None
        if expert_scores_df_file_path is not None:
            self.experts_scores_df = pd.read_csv(expert_scores_df_file_path)
            logger.info(
                f"self.experts_scores_df:\n"
                f"columns: {self.experts_scores_df.columns}\n"
                f"shape: {self.experts_scores_df.shape}\n"
            )
        logger.info("Loaded features and scores successfully.")
        self._clean_notebook_metrics_df()
        self._clean_notebook_scores_df()
        if self.experts_scores_df is not None:
            self._clean_experts_scores_df()

    def get_train_test_split(
        self,
        notebook_metrics_filters: list = [],
        notebook_scores_filters: list = [],
        split_factor: float = 0.7,
        selection_ratio: float = 0.25,
        sort_by: str = config.DEFAULT_NOTEBOOK_SCORES_SORT_BY,
        include_pt: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, list, list]:
        features_df = self.apply_filters(self.notebook_metrics_df, notebook_metrics_filters)
        scores_df = self.apply_filters(self.notebook_scores_df, notebook_scores_filters)
        notebooks_sorted_by_score = self._prepare_data(
            features_df=features_df,
            scores_df=scores_df,
            sort_by=sort_by,
            include_pt=include_pt,
        )
        return self._split_data(
            notebooks_sorted_by_score=notebooks_sorted_by_score,
            split_factor=split_factor,
            selection_ratio=selection_ratio,
        )

    def get_experts_test_split(
        self,
        notebook_metrics_filters: list = [],
        include_pt: bool = True,
    ) -> Tuple[pd.DataFrame, list]:
        if self.experts_scores_df is None:
            raise Exception("set experts_scores_df to get experts test split.")
        features_df = self.apply_filters(self.notebook_metrics_df, notebook_metrics_filters)
        merged_df = pd.merge(self.experts_scores_df, features_df, on="KernelId", how="inner")
        if not include_pt:
            merged_df.drop(columns=["PT"], axis=1, inplace=True)
        ones = merged_df[merged_df["expert_score"] == 1]
        zeros = merged_df[merged_df["expert_score"] == 0]
        min_len = min(len(ones), len(zeros))
        X = pd.concat([zeros.head(min_len), ones.head(min_len)])
        X.drop(["KernelId"], axis=1, inplace=True)
        X.rename(
            columns={
                "ALLC": "ALLCL",
            },
            inplace=True,
        )
        X_test_experts = X.drop(["expert_score"], axis=1)
        y_test_experts = list(X["expert_score"])
        return X_test_experts, y_test_experts

    def _clean_notebook_metrics_df(self) -> None:
        logger.info("Going to clean self.notebook_metrics_df...")
        logger.info(
            f"self.notebook_metrics_df:\n"
            f"columns: {self.notebook_metrics_df.columns}\n"
            f"shape: {self.notebook_metrics_df.shape}\n"
            f"dtypes: {self.notebook_metrics_df.dtypes}\n"
        )
        self.notebook_metrics_df.drop_duplicates(inplace=True)
        numeric_columns = list(self.notebook_metrics_df.columns)
        numeric_columns.remove("kernel_id")
        self.notebook_metrics_df[numeric_columns] = self.notebook_metrics_df[numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        self.notebook_metrics_df.dropna(inplace=True)
        self.notebook_metrics_df.rename(columns={"kernel_id": "KernelId"}, inplace=True)
        logger.info("Cleaned self.notebook_scores_df.")
        logger.info(
            f"self.notebook_metrics_df:\n"
            f"columns: {self.notebook_metrics_df.columns}\n"
            f"shape: {self.notebook_metrics_df.shape}\n"
            f"dtypes: {self.notebook_metrics_df.dtypes}\n"
        )

    def _clean_notebook_scores_df(self) -> None:
        logger.info("Going to clean self.notebook_scores_df...")
        logger.info(
            f"self.notebook_scores_df:\n"
            f"columns: {self.notebook_scores_df.columns}\n"
            f"shape: {self.notebook_scores_df.shape}\n"
            f"dtypes: {self.notebook_scores_df.dtypes}\n"
        )
        self.notebook_scores_df = self.notebook_scores_df[
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
        self.notebook_scores_df.drop_duplicates(inplace=True)
        numeric_columns = ["combined_score", "TotalVotes", "score_scaled", "vote_scaled"]
        self.notebook_scores_df[numeric_columns] = self.notebook_scores_df[numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        self.notebook_scores_df.dropna(subset=numeric_columns, inplace=True)
        logger.info("Cleaned self.notebook_scores_df.")
        logger.info(
            f"self.notebook_scores_df:\n"
            f"columns: {self.notebook_scores_df.columns}\n"
            f"shape: {self.notebook_scores_df.shape}\n"
            f"dtypes: {self.notebook_scores_df.dtypes}\n"
        )

    def _clean_experts_scores_df(self) -> None:
        logger.info("Going to clean self.experts_scores_df...")
        logger.info(
            f"self.experts_scores_df:\n"
            f"columns: {self.experts_scores_df.columns}\n"
            f"shape: {self.experts_scores_df.shape}\n"
            f"dtypes: {self.experts_scores_df.dtypes}\n"
        )
        self.experts_scores_df = self.experts_scores_df[
            [
                "KernelId",
                "expert_score",
            ]
        ]
        self.experts_scores_df.drop_duplicates(inplace=True)
        numeric_columns = ["expert_score"]
        self.experts_scores_df[numeric_columns] = self.experts_scores_df[numeric_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        self.experts_scores_df.dropna(subset=numeric_columns, inplace=True)
        logger.info("Cleaned self.experts_scores_df.")
        logger.info(
            f"self.experts_scores_df:\n"
            f"columns: {self.experts_scores_df.columns}\n"
            f"shape: {self.experts_scores_df.shape}\n"
            f"dtypes: {self.experts_scores_df.dtypes}\n"
        )

    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: list) -> pd.DataFrame:
        df = df.copy()
        for filter in filters:
            df = df.query(filter)
        return df

    def _prepare_data(
        self, features_df: pd.DataFrame, scores_df: pd.DataFrame, sort_by: str, include_pt: bool = True
    ) -> pd.DataFrame:
        logger.info(f"features_df:\n" f"columns: {features_df.columns}\n" f"shape: {features_df.shape}\n")
        logger.info(f"scores_df:\n" f"columns: {scores_df.columns}\n" f"shape: {scores_df.shape}\n")

        logger.info("Going to merge features and scores...")

        merged_df = pd.merge(
            scores_df,
            features_df,
            on="KernelId",
            how="inner",
        )

        logger.info(f"Going to sort merged_df by {sort_by}...")
        merged_df.sort_values(by=[sort_by], inplace=True)

        # Drop scores and unique id fields
        merged_df.drop(
            (
                [
                    "KernelId",
                    "PerformanceTier_kerneluser",
                    "TotalViews",
                    "TotalVotes",
                    "topic_score",
                    "score_scaled",
                    "vote_scaled",
                    "combined_score",
                ]
                + ([] if include_pt else ["PT"])
            ),
            axis=1,
            inplace=True,
        )

        # TODO: standardize column names
        merged_df.rename(
            columns={
                "ALLC": "ALLCL",
            },
            inplace=True,
        )

        logger.info(f"merged_df:\n{merged_df.columns}\n{merged_df.shape}")
        merged_df.info()
        return merged_df

    def _split_data(
        self,
        notebooks_sorted_by_score: pd.DataFrame,
        split_factor: float = 0.7,
        selection_ratio: float = 0.25,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, list, list]:
        logger.info(f"get_train_test_split df.shape: {notebooks_sorted_by_score.shape}")
        X = notebooks_sorted_by_score.copy()
        le = len(X)
        q1 = int(selection_ratio * (split_factor) * le)
        q2 = int(selection_ratio * (1 - split_factor) * le)

        X = pd.concat([X.head(q1), X.tail(q2)])
        y = [0 for _ in range(q1)] + [1 for _ in range(q2)]
        # TODO add ground truth
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        logger.info(f"X_train.shape:{X_train.shape}")
        logger.info(f"X_test.shape:{X_test.shape}")
        logger.info(f"len(y_train):{len(y_train)}")
        logger.info(f"len(y_test):{len(y_test)}")
        return (X_train, X_test, y_train, y_test)
