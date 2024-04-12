from pathlib import Path
from typing import Dict, List, Optional, Union
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile
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
async def get_models() -> Dict[str, Dict[str, Union[str, float, List[str]]]]:
    return model_store.active_models


@app.post("/notebook/upload/")
def upload_notebook(file: Annotated[UploadFile, File(description="Selected jupyter notebook.")]):
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
def extract_metrics(info: MetricsExtractionInfo):
    file_path = Path(config.NOTEBOOKS_FOLDER_PATH) / info.notebook_filename
    base_code_df_file_path = Path(config.DATAFRAMES_FOLDER_PATH) / info.base_code_df_filename
    extracted_notebook_metrics_df = extract_notebook_metrics_from_ipynb_file(
        file_path=str(file_path.resolve()),
        base_code_df_file_path=str(base_code_df_file_path.resolve()),
        chunk_size=info.chunk_size,
    )
    extracted_notebook_metrics_df.drop(["kernel_id"], axis=1, inplace=True)
    return {"message": extracted_notebook_metrics_df.iloc[[0]].to_dict(orient="index")[0]}


@app.post("/notebook/predict/")
def predict():
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
