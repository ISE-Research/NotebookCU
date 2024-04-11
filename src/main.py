from logger import init_logger
from fastapi import FastAPI
from model_store import ModelStore
from typing import Dict, Union

init_logger()
model_store = ModelStore()
app = FastAPI()


@app.get("/models/")
def get_models() -> Dict[str, Dict[str, Union[str, float]]]:
    return model_store.active_models


@app.post("/notebook/upload/")
def upload_notebook():
    pass


@app.post("/notebook/metrics/")
def extract_metrics():
    pass


@app.post("/notebook/predict/")
def predict():
    pass
