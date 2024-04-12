import os
from pathlib import Path
from typing import Dict, List, Union
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile
from typing_extensions import Annotated

import config
from logger import init_logger
from model_store import ModelStore

init_logger()
model_store = ModelStore()
app = FastAPI()


@app.get("/models/")
async def get_models() -> Dict[str, Dict[str, Union[str, float, List[str]]]]:
    return model_store.active_models


@app.post("/notebook/upload/")
def upload_notebook(
    file: Annotated[
        UploadFile,
        File(
            description="Selected jupyter notebook.",
        ),
    ]
):
    filename = f"{uuid4().hex}-{file.filename}"
    notebooks_folder_path = Path(config.NOTEBOOKS_FOLDER_PATH)
    filepath = notebooks_folder_path / filename
    with open(filepath, "wb") as buffer:
        buffer.write(file.file.read())
    file.file.close()
    return {"message": filename}


@app.post("/notebook/metrics/")
def extract_metrics():
    pass


@app.post("/notebook/predict/")
def predict():
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
