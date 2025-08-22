# Predicting the Understandability of Computational Notebooks through Code Metrics Analysis

This repository contains the source code for a code comprehension predictor service for computational notesbooks.

## Usage

To run the code, install the requirements by executing the following command:

```bash
pip install -r requirements.txt
```

After installing the requirements, you can run the [CLI](#cli) or [API](#fastapi) and start using the service.

## Data Requirements

To use the functionalities provided in this repository, you will need certain CSV files containing notebook code and markdown cell data. These files can be found [here (DistilKaggle: a distilled dataset of Kaggle Jupyter notebooks)](https://zenodo.org/records/10317389) and [here (A Predictive Model to Identify Effective Metrics for the Comprehension of Computational Notebooks)](https://zenodo.org/records/8126338).

Use below download links to get started
- [notebook_metrics.csv](https://zenodo.org/records/10317389/files/notebook_metrics.csv?download=1): notebooks features file, mainly used to train models.
- [code.csv](https://zenodo.org/records/10317389/files/code.csv?download=1): mainly used for metrics extraction.
- [augmented_kernel_quality.csv](https://drive.google.com/uc?id=1rks7UbT8Bbl7TdQvfqoXx6fXhaPM8xOv): notebook scores file, mainly used to train models.
- [sample1050_labeled_by_experts.csv](https://drive.google.com/file/d/1hwdPgr2NUsbVBIopLykYa7dPxGFLi5DM/view?usp=drive_link): used to evaluate the models.

## Folder Structure
- [src](./src/): Contains the main code that provides code comprehension prediction and metrics evaluation.
    - [src/core](./src/core): includes the main python files of the project. These classes and functions do the actual work behind the interfaces.
    - [src/utils](./src/utils/): helper files used to manage the project like [config.py](./src/utils/config.py) where we manage all the configurations.
    - [src/notebooks](./src/utils/): base notebook files that support the paper's results.
- [dataframes](./dataframes/): Contains basic data of selected jupyter notebooks for training models. For example, code.csv that contains the source codes used in each notebook, and markdown.csv that has the markdown cells data.
- [metrics](./metrics/): Contains CSV files with metrics of selected Jupyter notebooks for training the models. For instance, code_cell_metrics.csv contains metrics of each code cell in the notebook, markdown_cell_metrics.csv contains markdown cell metrics of each notebook, and notebook_metrics.csv holds the aggregated metrics of all cells in the notebook.
- [notebooks](./notebooks/): Stores the notebooks provided to be predicted by the code.
- [models](./models/): Stores the trained models.
- [logs](./logs/): Keeps the log files.
- [cache](./cache/): Is used for cached data.

## CLI
First, cd to the src directory and then execute `cli.py` file and start your journey.
```bash
cd src
export PYTHONPATH="$(pwd)"
python cli.py --help
```
Use --help for each command to get further instructions. Some use cases are provided below.

```bash
python cli.py
python cli.py extract-dataframe-metrics --help
python cli.py extract-dataframe-metrics --chunk-size 100 --limit-chunk-count 5
python cli.py extract-dataframe-metrics ../dataframes/markdown.csv ../metrics/markdown_cell_metrics.csv --chunk-size 100 --limit-chunk-count 5 --file-type markdown

python cli.py aggregate-metrics --help
python cli.py aggregate-metrics ../metrics/code_cell_metrics.csv ../metrics/markdown_cell_metrics.csv ../metrics/notebook_metrics_lite.csv

python cli.py extract-notebook-metrics --help
python cli.py extract-notebook-metrics ../notebooks/file.ipynb ../notebooks/results.json
python cli.py extract-notebook-metrics ../notebooks/file.ipynb ../notebooks/results.csv

python cli.py predict ../notebooks/file.ipynb cat_boost ../models/catBoostClassifier.withOutPT.sf50.sr20.combined_score.v2.model 
python cli.py predict ../notebooks/file.ipynb cat_boost ../models/catBoostClassifier.withPT.sf50.sr20.combined_score.v2.model --pt-score 10
```

## FastAPI
First, cd to the src directory and then execute `main.py` file and start your journey.
```bash
cd src
export PYTHONPATH="$(pwd)"
python main.py
```
after this you can see the documentation of the apis at http://localhost:8000/docs.

## Docker
Use below command to build and run the image using docker compose
```bash
docker compose up --build
```

## Citation
This work is published in the Empirical Software Engineering journal, under the title of "Predicting the understandability of computational notebooks through code metrics analysis". 

Access the paper from: <a href="https://rdcu.be/ehKdi" target="_blank">https://rdcu.be/ehKdi</a>.

```python
@article{ghahfarokhi2025predicting,
  title={Predicting the understandability of computational notebooks through code metrics analysis},
  author={Ghahfarokhi, Mojtaba Mostafavi and Asadi, Alireza and Asgari, Arash and Mohammadi, Bardia and Heydarnoori, Abbas},
  journal={Empirical Software Engineering},
  volume={30},
  number={3},
  pages={98},
  year={2025},
  publisher={Springer}
}
```
