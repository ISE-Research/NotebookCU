import logging
from pathlib import Path

import kaggle
import pandas as pd
import requests
from retrying import retry
from sklearn.metrics import classification_report

from pprint import pp
from utils.logger import init_logger

logger = logging.getLogger(__name__)


@retry(stop_max_attempt_number=10, wait_fixed=1000)  # Retry 10 times, waiting 1 second between retries
def download_kernel(user: str, notebook_slug: str, file_path: Path) -> None:
    kaggle.api.kernels_pull(kernel=f"{user}/{notebook_slug}", path=file_path.parent)
    downloaded_file = file_path.parent / f"{notebook_slug}.ipynb"
    downloaded_file.rename(file_path)
    logger.info(f"Downloaded and saved {file_path.name} from Kaggle")


def download_notebooks(experts_scores_df: pd.DataFrame) -> None:
    kaggle.api.authenticate()
    for index, row in experts_scores_df.iterrows():
        download_notebook(url=row["url"], kernel_id=row["KernelId"])


def download_notebook(url: str, kernel_id: str) -> None:
    filename = f"{kernel_id}_test.ipynb"
    try:
        # Extract user and notebook slug from the URL
        url_parts = url.split("/")
        if len(url_parts) >= 7:
            user = url_parts[4]
            notebook_slug = url_parts[5]
            print(user, notebook_slug)

            file_path = Path(f"../notebooks/experts_rated/{filename}")

            if file_path.is_file():
                logger.info(f"{file_path} File already exists.")
                return

            # Download the notebook using the Kaggle API with retry mechanism
            download_kernel(user, notebook_slug, file_path)
        else:
            logger.error(f"Invalid URL format: {url}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")


def upload_file(url: str, file_path: str):
    try:
        with open(file_path, "rb") as file:
            files = {"file": file}
            response = requests.post(url, files=files)
            json_response = response.json()
            return json_response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_and_upload_notebooks(
    input_csv_path: Path, notebooks_dir: Path, upload_url: str, output_csv_path: Path
) -> None:
    """
    Processes the CSV file, checks for the existence of the required files,
    uploads them, updates the DataFrame with the server response, and saves
    the updated DataFrame to a new CSV file after processing each row.

    :param input_csv_path: Path to the input CSV file.
    :param notebooks_dir: Directory where the notebook files are stored.
    :param upload_url: URL to which the notebooks should be uploaded.
    :param output_csv_path: Path to the output CSV file.
    """
    # Read the input CSV file into a DataFrame
    df_input = pd.read_csv(input_csv_path)

    # Check if the output CSV file exists
    if output_csv_path.exists():
        df_output = pd.read_csv(output_csv_path)
    else:
        df_output = df_input.copy()
        if "KernelFile" not in df_output.columns:
            df_output["KernelFile"] = None
        if "ServerName" not in df_output.columns:
            df_output["ServerName"] = None

    # Iterate through each row in the input DataFrame
    for index, row in df_input.iterrows():
        kernel_id = row["KernelId"]

        # Check if ServerName is already set in the output DataFrame
        if not pd.isna(df_output.loc[df_output["KernelId"] == kernel_id, "ServerName"]).all():
            # If ServerName is already set, skip to the next row
            print(f"Skipping KernelId {kernel_id} as ServerName is already set.")
            continue

        file_name = f"{kernel_id}_test.ipynb"
        file_path = notebooks_dir / file_name

        # Check if the file exists
        if file_path.exists():
            df_output.at[index, "KernelFile"] = str(file_path)

            # Upload the file using a POST request
            with open(file_path, "rb") as f:
                files = {"file": (file_name, f, "application/octet-stream")}
                response = requests.post(upload_url, files=files)

                # Check if the request was successful
                if response.status_code == 200:
                    server_name = response.json().get("result")
                    df_output.at[index, "ServerName"] = server_name
                else:
                    print(f"Failed to upload {file_name}: {response.status_code} - {response.text}")
        else:
            print(f"File not found: {file_path}")

        # Save the updated DataFrame to the CSV file after processing each row
        df_output.to_csv(output_csv_path, index=False)
        print(f"Processed and saved row {index + 1}/{len(df_input)}")

    print("CSV file has been updated and saved.")


def process_and_predict(input_csv_path: Path, model_id: str, output_csv_path: Path, url: str) -> None:
    """
    Processes the CSV file, sends POST requests to the prediction endpoint for each row,
    and updates the DataFrame with the prediction results.

    :param input_csv_path: Path to the input CSV file.
    :param model_id: The model ID to be used in the POST request.
    :param output_csv_path: Path to the output CSV file.
    :param url: The URL to which the prediction POST request should be sent.
    """
    # Read the input CSV file into a DataFrame
    df_input = pd.read_csv(input_csv_path)

    # Check if the output CSV file exists
    prediction_col = f"prediction_{model_id}"
    if output_csv_path.exists():
        df_output = pd.read_csv(output_csv_path)
    else:
        df_output = df_input.copy()
        if prediction_col not in df_output.columns:
            df_output[prediction_col] = None

    # Iterate through each row in the DataFrame
    for index, row in df_output.iterrows():
        kernel_id = row["KernelId"]
        server_name = row["ServerName"]

        if pd.isna(server_name):
            print(f"Missing server name for {kernel_id}.")
            continue

        # Check if prediction is already made for this model_id
        if pd.isna(row[prediction_col]):
            # Send POST request with the required payload
            payload = {
                "notebook_filename": server_name,
                "model_id": model_id,
            }

            response = requests.post(url, json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                prediction = response.json().get("prediction")
                df_output.at[index, prediction_col] = prediction
            else:
                print(f"Failed to get prediction for {server_name}: {response.status_code} - {response.text}")
        else:
            print(f"Skipping row {index+1}/{len(df_output)}: Prediction already exists")

        # Save the DataFrame to the output CSV file after processing each row
        df_output.to_csv(output_csv_path, index=False)
        print(f"Processed and saved row {index+1}/{len(df_output)}")

    print("CSV file has been updated and saved.")


def get_classification_report(csv_file_path: Path, model_id: str) -> None:
    """
    Reads a CSV file, extracts 'expert_score' and 'prediction_101' columns,
    and generates a classification report using scikit-learn's classification_report function.

    :param csv_file_path: Path to the input CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Drop rows with NaN values in 'expert_score' or 'prediction_{model_id}' column
    df.dropna(subset=["expert_score", f"prediction_{model_id}"], inplace=True)

    # Extract 'expert_score' and 'prediction_101' columns as lists
    expert_scores = df["expert_score"].tolist()
    predictions = df[f"prediction_{model_id}"].tolist()

    # Generate classification report
    report = classification_report(expert_scores, predictions, output_dict=True)

    # Print the classification report
    pp(report)
    pp(
        {
            "0": {
                "precision": 0.6466165413533834,
                "recall": 0.5443037974683544,
                "f1-score": 0.5910652920962199,
                "support": 158.0,
            },
            "1": {
                "precision": 0.6065573770491803,
                "recall": 0.7025316455696202,
                "f1-score": 0.6510263929618768,
                "support": 158.0,
            },
            "accuracy": 0.6234177215189873,
            "macro avg": {
                "precision": 0.6265869592012818,
                "recall": 0.6234177215189873,
                "f1-score": 0.6210458425290484,
                "support": 316.0,
            },
            "weighted avg": {
                "precision": 0.6265869592012818,
                "recall": 0.6234177215189873,
                "f1-score": 0.6210458425290484,
                "support": 316.0,
            }
        }
    )


def run():
    init_logger()
    expert_scores_df_file_path = Path("../metrics/sample1050_labeled_by_experts_tests.csv")
    rated_notebooks_dir_path = Path("../notebooks/experts_rated/")
    experts_scores_df = pd.read_csv(expert_scores_df_file_path)
    logger.info(f"self.experts_scores_df:\ncolumns: {experts_scores_df.columns}\nshape: {experts_scores_df.shape}\n")
    # download_notebooks(experts_scores_df)
    # process_and_upload_notebooks(
    #     input_csv_path=expert_scores_df_file_path,
    #     notebooks_dir=rated_notebooks_dir_path,
    #     upload_url="http://46.249.101.150/notebook/upload/",
    #     output_csv_path=Path("../metrics/sample1050_labeled_by_experts_uploaded.csv"),
    # )
    # process_and_predict(
    #     input_csv_path=Path("../metrics/sample1050_labeled_by_experts_uploaded.csv"),
    #     model_id="0038d382b5303d1a",
    #     output_csv_path=Path("../metrics/sample1050_labeled_by_experts_predicted.csv"),
    #     url="http://46.249.101.150/notebook/predict/",
    # )
    get_classification_report(
        csv_file_path=Path("../metrics/sample1050_labeled_by_experts_predicted.csv"),
        model_id="0038d382b5303d1a", # cat_boost no pt
    )


if __name__ == "__main__":
    run()
