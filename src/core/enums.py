from enum import Enum

from core.classifiers import (
    BaseClassifier,
    CatBoostClassifierCustom,
    DecisionTreeClassifierCustom,
    RandomTreeClassifierCustom,
    XGBoostClassifierCustom,
)


class FileType(str, Enum):
    markdown = "markdown"
    code = "code"


class ClassifierType(str, Enum):
    cat_boost = "cat_boost"
    xg_boost = "xg_boost"
    decision_tree = "decision_tree"
    random_forest = "random_forest"

    def get_classifier_class(self) -> BaseClassifier:
        return {
            ClassifierType.cat_boost: CatBoostClassifierCustom,
            ClassifierType.xg_boost: XGBoostClassifierCustom,
            ClassifierType.decision_tree: DecisionTreeClassifierCustom,
            ClassifierType.random_forest: RandomTreeClassifierCustom,
        }.get(self)
