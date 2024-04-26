import logging
import pickle
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

logger = logging.getLogger(__name__)

tqdm.pandas()


class BaseClassifier(ABC):
    def __init__(self, **kwargs):
        self.model_class = None
        self.model = None
        self.kwargs = kwargs

    def train(self, X_train: pd.DataFrame, y_train: list) -> None:
        self.model = self.model_class(**self.kwargs)
        self.model.fit(X_train, y_train)
        logger.info(f"{self.__class__.__name__} train finished successfully.")

    def test(self, X_test: pd.DataFrame, y_test: list) -> dict:
        y_pred = self.model.predict(X_test)
        roc_auc_score_ = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        logger.info(f"\nroc_auc_score: {roc_auc_score_}")
        logger.info(f"{self.__class__.__name__} test finished successfully.")
        metrics = classification_report(y_test, y_pred, output_dict=True)
        metrics.update({"roc_auc_score": roc_auc_score_})
        return metrics

    def predict(self, x: Union[pd.DataFrame, np.array]):
        if isinstance(x, pd.DataFrame):
            return self.model.predict(x)
        else:
            return self.model.predict(x.reshape(1, -1))  # Reshape for single prediction

    def save_model(self, path: str) -> None:
        if self.model:
            self.model.save_model(path)
            logger.info(f"Model {path} saved successfully.")
        else:
            logger.warning("No model trained yet. Cannot save.")

    def load_model(self, path: str) -> None:
        try:
            self.model = self.model_class()
            self.model.load_model(path)
            logger.info(f"Model {path} loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model {path} file not found.")

    @staticmethod
    @abstractmethod
    def get_default_instance() -> "BaseClassifier":
        pass


class BaseClassifierPickleLoader(BaseClassifier, ABC):

    def save_model(self, path: str) -> None:
        if self.model:
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info("Model saved successfully.")
        else:
            logger.warning("No model trained yet. Cannot save.")

    def load_model(self, path: str) -> "BaseClassifier":
        try:
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model {path} loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model {path} file not found.")


class CatBoostClassifierCustom(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_class = CatBoostClassifier

    @staticmethod
    def get_default_instance() -> "CatBoostClassifierCustom":
        return CatBoostClassifierCustom(iterations=100, learning_rate=0.001)


class XGBoostClassifierCustom(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_class = xgb.XGBClassifier

    @staticmethod
    def get_default_instance() -> "XGBoostClassifierCustom":
        return XGBoostClassifierCustom()


class DecisionTreeClassifierCustom(BaseClassifierPickleLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_class = DecisionTreeClassifier

    @staticmethod
    def get_default_instance() -> "DecisionTreeClassifierCustom":
        return DecisionTreeClassifierCustom(random_state=0, max_depth=2)


class RandomTreeClassifierCustom(BaseClassifierPickleLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_class = RandomizedSearchCV

    @staticmethod
    def get_default_instance() -> "RandomTreeClassifierCustom":
        random_grid = {
            "bootstrap": [True, False],
            "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            "max_features": ["auto", "sqrt"],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5, 10],
        }
        return RandomTreeClassifierCustom(
            estimator=RandomForestClassifier(),
            param_distributions=random_grid,
            n_iter=10,
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1,
        )
