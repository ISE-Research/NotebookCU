from enum import Enum

from classifiers import (BaseClassifier, CatBoostClassifierCustom,
                         DecisionTreeClassifierCustom,
                         RandomTreeClassifierCustom, XGBoostClassifierCustom)


class FileType(str, Enum):
    markdown = "markdown"
    code = "code"


class ModelType(str, Enum):
    cat_boost = "cat_boost"
    xgb_boost = "xgb_boost"
    decision_tree = "decision_tree"
    random_forest = "random_forest"

    def get_classifier_class(self) -> BaseClassifier:
        return {
            ModelType.cat_boost: CatBoostClassifierCustom,
            ModelType.xgb_boost: XGBoostClassifierCustom,
            ModelType.decision_tree: DecisionTreeClassifierCustom,
            ModelType.random_forest: RandomTreeClassifierCustom,
        }.get(self)
