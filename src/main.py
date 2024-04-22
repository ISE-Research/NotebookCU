from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, Body, Query
from pydantic import BaseModel, Field
from typing_extensions import Annotated

import config
from extract_metrics import extract_notebook_metrics_from_ipynb_file
from logger import init_logger
from model_store import ModelStore

init_logger()
model_store = ModelStore()


description = """
API to check your coding quality.

## Models

You will be able to select the model which decides if your code is comprehensive enough based on the details about the training of the model.

## Notebooks

You will be able to upload your notebooks.

* **Upload Notebook**: Upload your notebook and get its identifier.
* **Get Metrics**: See the metrics about your notebook.
* **Get Prediction**: See if your code is comprehensive or not based on the selected model.
"""

app = FastAPI(
    title="Code Comprehension Service",
    description=description,
    summary="Check if your code is a good one.",
    version="0.0.1",
    contact={
        "name": "Masih Beigi Rizi",
        "email": "masihbr@gmail.com",
    },
)


class ModelInfo(BaseModel):
    id: str
    file_name: str
    model_type: str
    notebook_metrics_df_file_name: str
    notebook_scores_df_file_path: str
    notebook_metrics_filters: List[str]
    notebook_scores_filters: List[str]
    sort_by: str
    split_factor: float
    selection_ratio: float
    include_pt: bool
    metrics: dict


@app.get(
    "/models/",
    summary="Get a list of available models.",
    response_model=List[ModelInfo],
)
async def get_models() -> List[ModelInfo]:
    """
    This function retrieves information about the currently active models.

    Returns:
        A list containing information about active models.
    """
    return [ModelInfo(id=model_id, **model_detail) for model_id, model_detail in model_store.active_models.items()]


class UploadNotebookResponse(BaseModel):
    message: str = Field(examples=["c744d10c6a2d4ec49cd30f69b8301da3-14091946.ipynb"])


@app.post(
    "/notebook/upload/",
    summary="Upload a Jupyter notebook file.",
    response_model=UploadNotebookResponse,
)
def upload_notebook(
    file: Annotated[
        UploadFile,
        File(
            description="Selected jupyter notebook.",
            examples=["filename.ipynb"],
        ),
    ]
) -> UploadNotebookResponse:
    """
    This function uploads a Jupyter notebook file to the server.

    Args:
        file: The Jupyter notebook file to be uploaded.

    Returns:
        Dict[str, str]: A dictionary containing a message with the generated filename.
    """
    filename = f"{uuid4().hex}-{file.filename}"
    notebooks_folder_path = Path(config.NOTEBOOKS_FOLDER_PATH)
    filepath = notebooks_folder_path / filename
    with open(filepath, "wb") as buffer:
        buffer.write(file.file.read())
    file.file.close()
    return UploadNotebookResponse(message=filename)


class MetricsExtractionInfo(BaseModel):
    notebook_filename: str = Field(examples=["uuid-filename.ipynb"])
    base_code_df_filename: Optional[str] = Field(
        default=Path(config.CODE_DF_FILE_PATH).name, examples=[Path(config.CODE_DF_FILE_PATH).name]
    )
    chunk_size: Optional[int] = Field(default=config.CHUNK_SIZE, examples=[config.CHUNK_SIZE])


class MetricsExtractionResponse(BaseModel):
    message: Dict[str, Union[int, float]] = Field(
        examples=[
            {
                "LOC": 78,
                "BLC": 0,
                "UDF": 3,
                "I": 6,
                "EH": 0,
                "NVD": 3,
                "NEC": 1,
                "S": 17,
                "P": 82,
                "OPRND": 193,
                "OPRATOR": 93,
                "UOPRND": 185,
                "UOPRATOR": 29,
                "ID": 428,
                "LOCom": 16,
                "EAP": 2.3896513096877,
                "CW": 114,
                "ALID": 6.9,
                "MeanLCC": 7.8,
                "ALLC": 38.947288006111535,
                "KLCID": 4.522916666666666,
                "CyC": 1.2,
                "MLID": 25,
                "NBD": 10.363636363636363,
                "AID": 118,
                "CC": 10,
                "MC": 11,
                "MeanWMC": 19.90909090909091,
                "MeanLMC": 1.3636363636363635,
                "H1": 0,
                "H2": 3,
                "H3": 1,
                "MW": 219,
                "LMC": 15,
            }
        ]
    )


@app.post(
    "/notebook/metrics/",
    summary="Extract metrics from a Jupyter notebook file.",
    response_model=MetricsExtractionResponse,
)
def extract_metrics(info: MetricsExtractionInfo) -> MetricsExtractionResponse:
    """
    This function extracts metrics from a specified Jupyter notebook file.

    Takes:
        information about the notebook and extraction parameters.

    Returns:
        A dictionary containing the extracted metrics.
    """
    file_path = Path(config.NOTEBOOKS_FOLDER_PATH) / info.notebook_filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Specified notebook does not exist.")

    base_code_df_file_path = Path(config.DATAFRAMES_FOLDER_PATH) / info.base_code_df_filename
    if not base_code_df_file_path.is_file():
        raise HTTPException(status_code=404, detail="Specified code df does not exist.")

    extracted_notebook_metrics_df = extract_notebook_metrics_from_ipynb_file(
        file_path=str(file_path.resolve()),
        base_code_df_file_path=str(base_code_df_file_path.resolve()),
        chunk_size=info.chunk_size,
    )
    extracted_notebook_metrics_df.drop(["kernel_id"], axis=1, inplace=True)
    return MetricsExtractionResponse(message=extracted_notebook_metrics_df.iloc[[0]].to_dict(orient="index")[0])


class PredictionInfo(MetricsExtractionInfo):
    model_id: str = Field(examples=["1234ABCD"], description="Use ids in the response of models/ api.")
    pt_score: Optional[int] = Field(
        default=None, examples=[10], description="Only set when model has value True for its include_pt field."
    )


class PredictionResponse(BaseModel):
    message: Dict[str, Any] = Field(
        examples=[
            {
                "metrics": {
                    "LOC": 78,
                    "BLC": 0,
                    "UDF": 3,
                    "I": 6,
                    "EH": 0,
                    "NVD": 3,
                    "NEC": 1,
                    "S": 17,
                    "P": 82,
                    "OPRND": 193,
                    "OPRATOR": 93,
                    "UOPRND": 185,
                    "UOPRATOR": 29,
                    "ID": 428,
                    "LOCom": 16,
                    "EAP": 2.3896513096876997,
                    "CW": 114,
                    "ALID": 6.9,
                    "MeanLCC": 7.8,
                    "ALLCL": 38.947288006111535,
                    "KLCID": 4.522916666666666,
                    "CyC": 1.2,
                    "MLID": 25,
                    "NBD": 10.363636363636363,
                    "AID": 118,
                    "CC": 10,
                    "MC": 11,
                    "MeanWMC": 19.90909090909091,
                    "MeanLMC": 1.3636363636363635,
                    "H1": 0,
                    "H2": 3,
                    "H3": 1,
                    "MW": 219,
                    "LMC": 15,
                    "PT": 10,
                },
                "prediction": 0,
            }
        ]
    )


@app.post(
    "/notebook/predict/",
    summary="Make predictions using a model on a notebook's metrics.",
    response_model=PredictionResponse,
)
def predict(info: PredictionInfo) -> PredictionResponse:
    """
    This function performs predictions on the metrics extracted from a notebook using a specified model.

    Takes:
        information about the notebook, model, and prediction parameters.

    Returns:
        a dictionary containing the metrics and the prediction result.
    """
    file_path = Path(config.NOTEBOOKS_FOLDER_PATH) / info.notebook_filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Specified notebook does not exist.")

    base_code_df_file_path = Path(config.DATAFRAMES_FOLDER_PATH) / info.base_code_df_filename
    if not base_code_df_file_path.is_file():
        raise HTTPException(status_code=404, detail="Specified code df does not exist.")

    extracted_notebook_metrics_df = extract_notebook_metrics_from_ipynb_file(
        file_path=str(file_path.resolve()),
        base_code_df_file_path=str(base_code_df_file_path.resolve()),
        chunk_size=info.chunk_size,
    )
    extracted_notebook_metrics_df.drop(["kernel_id"], axis=1, inplace=True)

    classifier = model_store.get_model(info.model_id)
    if classifier is None:
        raise HTTPException(status_code=404, detail="Specified model id does not exist.")

    # TODO: standardize column names
    extracted_notebook_metrics_df.rename(
        columns={
            "ALLC": "ALLCL",
        },
        inplace=True,
    )
    if info.pt_score is not None:
        if not model_store.get_model_info(info.model_id).get("include_pt"):
            raise Exception(status_code=400, detail="Specified model does not take PT score.")
        extracted_notebook_metrics_df["PT"] = info.pt_score

    result = classifier.predict(x=extracted_notebook_metrics_df)
    return PredictionResponse(
        message={
            "metrics": extracted_notebook_metrics_df.iloc[[0]].to_dict(orient="index")[0],
            "prediction": int(result[0]),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
