import json
import logging
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, Optional

import config
from classifiers import BaseClassifier
from enums import ModelType

logger = logging.getLogger(__name__)


class ModelStore:
    CHECKSUM_EXCLUDED_KEYS = ["metrics"]

    def __init__(self, models_dict_file_path: str = config.MODELS_DICT_FILE_PATH):
        self._models_dict_file_path: str = models_dict_file_path
        self._models_dict: Dict[str, Dict[str, Any]]
        self.load_models_dict()

    @property
    def active_models(self) -> Dict[str, Dict[str, Any]]:
        # TODO: unify dictionary filters like this.
        active_models = {
            key: {inner_key: inner_val for inner_key, inner_val in inner_dict.items() if inner_key != "model"}
            for key, inner_dict in self._models_dict.items()
            if "model" in inner_dict
        }
        return active_models

    def load_models_dict(self) -> None:
        self.load_models_dict_file()
        self.load_models_dict_models()

    def load_models_dict_file(self) -> None:
        file_path = Path(self._models_dict_file_path)
        if not file_path.exists():
            data = dict()
        else:
            if not file_path.is_file():
                raise Exception("Selected file path is not a file.")
            with open(str(file_path.resolve()), "r") as file:
                data = json.load(file)
        self._models_dict = data

    def load_models_dict_models(self) -> None:
        for key, val in self._models_dict.items():
            if self.get_hash_checksum(val) != key:
                logger.error(f"Wrong checksum for {key}: {val}.")
                continue
            model_file_path = Path(f"{config.MODELS_FOLDER_PATH}/{val['file_name']}")
            if not model_file_path.exists():
                logger.error(f"Could not find file_path: {model_file_path} for model: {key}.")
                continue
            if not model_file_path.is_file():
                logger.error(f"The file_path: {model_file_path} specified is not for a file, for model: {key}.")
                continue
            model_type = val["model_type"]
            classifier_class = ModelType(model_type).get_classifier_class()
            classifier: BaseClassifier = classifier_class()
            classifier.load_model(str(model_file_path.resolve()))
            val["model"] = classifier

    def add_model(
        self,
        model_file_path: Path,
        model_type: ModelType,
        **kwargs,
    ) -> None:
        model_dict = {
            "file_name": model_file_path.name,
            "model_type": model_type.value,
            **kwargs,
        }
        self._models_dict.update({self.get_hash_checksum(data=model_dict): model_dict})
        self.save_models_dict()

    def get_model(
        self,
        model_id: str,
    ) -> Optional[BaseClassifier]:
        return self.get_model_info(model_id=model_id).get("model")

    def get_model_info(
        self,
        model_id: str,
    ) -> Dict[str, Any]:
        return self._models_dict.get(model_id, {})

    def save_models_dict(self) -> None:
        serializable_models_dict = self.get_serializable_models_dict()
        self.save_models_dict_file(serializable_models_dict)

    def get_serializable_models_dict(self):
        serializable_models_dict = {
            key: {inner_key: value for inner_key, value in inner_dict.items() if inner_key != "model"}
            for key, inner_dict in self._models_dict.items()
        }
        return serializable_models_dict

    def save_models_dict_file(self, serializable_models_dict: Dict[str, Dict[str, Any]]):
        file_path = Path(self._models_dict_file_path)
        with open(str(file_path.resolve()), "w") as file:
            file.write(json.dumps(serializable_models_dict, indent=4))

    def get_hash_checksum(
        self,
        data: dict,
        digest_size: int = config.DEFAULT_DIGEST_SIZE_FOR_MODEL_KEYS,
    ) -> str:
        data_str = str({key: val for key, val in data.items() if key not in self.CHECKSUM_EXCLUDED_KEYS})
        hasher = blake2b(digest_size=digest_size)
        hasher.update(data_str.encode())
        return hasher.hexdigest()
