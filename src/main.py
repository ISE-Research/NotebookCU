from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from typing_extensions import Annotated

import config
from extract_metrics import extract_notebook_metrics_from_ipynb_file
from logger import init_logger
from model_store import ModelStore

init_logger()
model_store = ModelStore()
app = FastAPI()


@app.get("/models/")
async def get_models() -> Dict[str, Dict[str, Union[str, bool, float, List[str]]]]:
    return model_store.active_models


@app.post("/notebook/upload/")
def upload_notebook(file: Annotated[UploadFile, File(description="Selected jupyter notebook.")]) -> Dict[str, str]:
    filename = f"{uuid4().hex}-{file.filename}"
    notebooks_folder_path = Path(config.NOTEBOOKS_FOLDER_PATH)
    filepath = notebooks_folder_path / filename
    with open(filepath, "wb") as buffer:
        buffer.write(file.file.read())
    file.file.close()
    return {"message": filename}


class MetricsExtractionInfo(BaseModel):
    notebook_filename: str
    base_code_df_filename: Optional[str] = Path(config.CODE_DF_FILE_PATH).name
    chunk_size: Optional[int] = config.CHUNK_SIZE


@app.post("/notebook/metrics/")
def extract_metrics(info: MetricsExtractionInfo) -> Dict[str, Dict[str, Union[int, float]]]:
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
    return {"message": extracted_notebook_metrics_df.iloc[[0]].to_dict(orient="index")[0]}


class PredictionInfo(MetricsExtractionInfo):
    model_id: str
    pt_score: Optional[int]


@app.post("/notebook/predict/")
def predict(info: PredictionInfo) -> Dict[str, Dict[str, Any]]:
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
        extracted_notebook_metrics_df["PT"] = 10

    result = classifier.predict(x=extracted_notebook_metrics_df)
    return {
        "message": {
            "metrics": extracted_notebook_metrics_df.iloc[[0]].to_dict(orient="index")[0],
            "prediction": int(result[0]),
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
