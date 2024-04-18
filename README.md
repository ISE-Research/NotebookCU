# Code Comprehension Predictor Service

This repository contains the source code for a code comprehension predictor service.

## Usage

To run the code, install the requirements by executing the following command:

```bash
pip install -r requirements.txt
```

After installing the requirements, you can run the desired functions by executing main.py.

## Data Requirements

To use the function, you will need certain CSV files containing notebook code and markdown cell data. These files can be found [here (DistilKaggle: a distilled dataset of Kaggle Jupyter notebooks)](https://zenodo.org/records/10317389) and [here (A Predictive Model to Identify Effective Metrics for the Comprehension of Computational Notebooks)](https://zenodo.org/records/8126338).

## Folder Structure
- [src](./src/): Contains the main code that provides code comprehension prediction and metrics evaluation.
- [dataframes](./dataframes/): Contains basic data of selected jupyter notebooks for training models. For example, code.csv that contains the source codes used in each notebook, and markdown.csv that has the markdown cells data.
- [metrics](./metrics/): Contains CSV files with metrics of selected Jupyter notebooks for training the models. For instance, code_cell_metrics.csv contains metrics of each code cell in the notebook, markdown_cell_metrics.csv contains markdown cell metrics of each notebook, and notebook_metrics.csv holds the aggregated metrics of all cells in the notebook.
- [notebooks](./notebooks/): Stores the notebooks provided to be predicted by the code.
- [models](./models/): Stores the trained models.
- [logs](./logs/): Keeps the log files.
- [cache](./cache/): Is used for cached data.

## CLI
First, cd to the src directory and then use below command and start your journey.
```bash
python cli.py --help
```
Use --help for each command to get further instructions.

## FastAPI
First, cd to the src directory and then use below command and start your journey.
```bash
uvicorn main:app --reload
```
after this you can see the documentation of the apis at http://localhost:8000/docs.

