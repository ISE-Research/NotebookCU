from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from classification_data import DataSelector
from classifiers import BaseClassifier
from enums import ModelType

VERSION = "v2"

def run_test(
    model: ModelType,
    selected_score: str,
    split_factor: float,
    selection_ratio: float,
    notebook_metrics_filter: list,
    notebook_scores_filter: list,
    include_pt: bool,
    data_selector: DataSelector,
    notebook_metrics_df_file_path: Path,
    notebook_scores_df_file_path: Path,
    experts_scores_df_file_path: Path,
) -> pd.DataFrame:
    classifier: BaseClassifier = model.get_classifier_class().get_default_instance()

    x_train, x_test, y_train, y_test = data_selector.get_train_test_split(
        notebook_metrics_filters=notebook_metrics_filter,
        notebook_scores_filters=notebook_scores_filter,
        sort_by=selected_score,
        split_factor=split_factor,
        selection_ratio=selection_ratio,
        include_pt=include_pt,
    )
    classifier.train(X_train=x_train, y_train=y_train)
    metrics = {"default": None, "experts": None}
    metrics["default"] = classifier.test(X_test=x_test, y_test=y_test)

    x_test_experts, y_test_experts = data_selector.get_experts_test_split(
        notebook_metrics_filters=notebook_metrics_filter,
        include_pt=include_pt,
    )
    metrics["experts"] = classifier.test(X_test=x_test_experts, y_test=y_test_experts)

    data = dict(
        model=model.value,
        notebook_metrics_df_file_name=notebook_metrics_df_file_path.name,
        notebook_scores_df_file_path=notebook_scores_df_file_path.name,
        experts_scores_df_file_path=experts_scores_df_file_path.name,
        notebook_metrics_filters=notebook_metrics_filter,
        notebook_scores_filters=notebook_scores_filter,
        sort_by=selected_score,
        split_factor=split_factor,
        selection_ratio=selection_ratio,
        include_pt=include_pt,
        metrics=metrics,
    )

    return pd.json_normalize(data=data, sep="_")


def run():
    models = [member for member in ModelType]
    models = [models[1]]
    notebook_metrics_df_file_path = Path(config.METRICS_FOLDER_PATH) / "notebook_metrics.csv"
    notebook_scores_df_file_path = Path(config.METRICS_FOLDER_PATH) / "augmented_kernel_quality.csv"
    selected_scores = ["combined_score", "score_scaled", "TotalVotes", "vote_scaled"]
    experts_scores_df_file_path = Path(config.METRICS_FOLDER_PATH) / "sample1050_labeled_by_experts_new.csv"
    split_factors = [0.35, 0.5, 0.65]
    selection_ratios = [0.15, 0.25, 0.35]
    notebook_metrics_filters = [
        ["LOC < 500"],
        ["LOC < 5000"],
        [],
    ]
    notebook_scores_filters = [
        [],
        ["TotalViews >= 500"],
        ["TotalViews >= 2000"],
    ]
    include_pts = [True, False]

    result_df_file_path = Path(config.METRICS_FOLDER_PATH) / f"tests/tests_output.{models[0].value}.{VERSION}.csv"

    print(models)
    print(str(notebook_metrics_df_file_path.resolve()))
    print(str(notebook_scores_df_file_path.resolve()))
    print(selected_scores)
    print(str(experts_scores_df_file_path.resolve()))
    print(split_factors)
    print(selection_ratios)
    print(notebook_metrics_filters)
    print(notebook_scores_filters)
    print(include_pts)

    params_list = []

    for model in models:
        for selected_score in selected_scores:
            for split_factor in split_factors:
                for selection_ratio in selection_ratios:
                    for notebook_metrics_filter in notebook_metrics_filters:
                        for notebook_scores_filter in notebook_scores_filters:
                            for include_pt in include_pts:
                                params_list.append(
                                    {
                                        "model": model,
                                        "selected_score": selected_score,
                                        "split_factor": split_factor,
                                        "selection_ratio": selection_ratio,
                                        "notebook_metrics_filter": notebook_metrics_filter,
                                        "notebook_scores_filter": notebook_scores_filter,
                                        "include_pt": include_pt,
                                    }
                                )

    print(
        f"len(params_list): {len(params_list)}"
    )
    input("PRESS ENTER TO BEGIN!")
    data_selector = DataSelector(
        notebook_metrics_df_file_path=str(notebook_metrics_df_file_path.resolve()),
        notebook_scores_df_file_path=str(notebook_scores_df_file_path.resolve()),
        expert_scores_df_file_path=str(experts_scores_df_file_path.resolve()),
    )

    try:
        index = len(pd.read_csv(result_df_file_path))
        params_list = params_list[index:]
        print(f"Continuing from {index}... params_list @ {params_list[0]}")
    except FileNotFoundError:
        index = 0
        print(f"Starting...")

    for params_dict in tqdm(params_list, total=len(params_list)):
        print(f"Current params_dict: {params_dict}.")
        result_df = run_test(
            **params_dict,
            data_selector=data_selector,
            notebook_metrics_df_file_path=notebook_metrics_df_file_path,
            notebook_scores_df_file_path=notebook_scores_df_file_path,
            experts_scores_df_file_path=experts_scores_df_file_path,
        )
        df = result_df
        df.to_csv(
            str(result_df_file_path.resolve()),
            index=False,
            header=(index == 0),
            mode="w" if (index == 0) else "a",
        )
        index += 1


def concat_results():
    dfs = []
    models = [member for member in ModelType]
    for model in models:
        result_df_file_path = Path(config.METRICS_FOLDER_PATH) / f"tests/tests_output.{model.value}.{VERSION}.csv"
        dfs.append(pd.read_csv(str(result_df_file_path.resolve())))
    df = pd.concat(dfs, ignore_index=True)
    result_df_file_path = Path(config.METRICS_FOLDER_PATH) / f"tests/tests_output.{VERSION}.csv"
    df.to_csv(str(result_df_file_path.resolve()), index=False)


if __name__ == "__main__":
    run()
