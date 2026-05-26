"""
Treina dois modelos de regressao para prever o potencial solar do proximo dia
com base em dados diarios da NASA POWER API.
"""

from __future__ import annotations

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_utils import (
    DATASET_PATH,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SHAP_IMAGE_PATH,
    SHAP_IMPORTANCE_PATH,
    TARGET_CLASS_COLUMN,
    TARGET_COLUMN,
)

warnings.filterwarnings("ignore")

HOLDOUT_START_YEAR = 2025
RANDOM_SEED = 42
N_FEATURES_TO_SELECT = 18
EXCLUDED_COLUMNS = {
    "date",
    "year",
    TARGET_COLUMN,
    TARGET_CLASS_COLUMN,
    "target_next_day_temperature_avg_c",
    "target_next_day_pv_output_kwh_kwp",
}


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )
    return preprocessor, numeric_columns, categorical_columns


def create_date_based_splits(dates: pd.Series, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_dates = np.array(sorted(pd.to_datetime(dates).unique()))
    split_size = len(unique_dates) // (n_splits + 1)
    if split_size == 0:
        raise ValueError("Nao ha datas suficientes para validacao temporal.")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(1, n_splits + 1):
        train_end = split_size * fold
        validation_end = split_size * (fold + 1)

        if fold == n_splits:
            validation_dates = unique_dates[train_end:]
        else:
            validation_dates = unique_dates[train_end:validation_end]

        train_dates = unique_dates[:train_end]
        train_indices = np.flatnonzero(pd.to_datetime(dates).isin(train_dates).to_numpy())
        validation_indices = np.flatnonzero(pd.to_datetime(dates).isin(validation_dates).to_numpy())
        splits.append((train_indices, validation_indices))
    return splits


def build_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=mutual_info_regression, k=N_FEATURES_TO_SELECT)),
            ("model", model),
        ]
    )


def summarize_cv_scores(cv_scores: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "cv_MAE_mean": float(-cv_scores["test_mae"].mean()),
        "cv_MAE_std": float(cv_scores["test_mae"].std()),
        "cv_RMSE_mean": float(-cv_scores["test_rmse"].mean()),
        "cv_RMSE_std": float(cv_scores["test_rmse"].std()),
        "cv_R2_mean": float(cv_scores["test_r2"].mean()),
        "cv_R2_std": float(cv_scores["test_r2"].std()),
    }


def get_selected_feature_names(pipeline: Pipeline) -> np.ndarray:
    preprocessor = pipeline.named_steps["preprocessor"]
    selector = pipeline.named_steps["selector"]
    feature_names = preprocessor.get_feature_names_out()
    return feature_names[selector.get_support()]


def run_shap_analysis(pipeline: Pipeline, X_test: pd.DataFrame) -> pd.DataFrame:
    transformed = pipeline.named_steps["preprocessor"].transform(X_test)
    selected = pipeline.named_steps["selector"].transform(transformed)
    feature_names = get_selected_feature_names(pipeline)

    sample_size = min(500, len(X_test))
    sample = selected[:sample_size]
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(sample)

    shap.summary_plot(
        shap_values,
        sample,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(SHAP_IMAGE_PATH, dpi=180, bbox_inches="tight")
    plt.close()

    shap_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(SHAP_IMPORTANCE_PATH, index=False)
    return shap_importance


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError("Dataset nao encontrado. Rode primeiro: python src/generate_dataset.py")

    dataset = pd.read_csv(DATASET_PATH)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset.sort_values(["date", "city"]).reset_index(drop=True)

    train_df = dataset[dataset["year"] < HOLDOUT_START_YEAR].copy()
    test_df = dataset[dataset["year"] >= HOLDOUT_START_YEAR].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("O recorte temporal nao gerou conjuntos de treino e teste validos.")

    feature_columns = [column for column in dataset.columns if column not in EXCLUDED_COLUMNS]
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    preprocessor, _, _ = build_preprocessor(X_train)
    cv_splits = create_date_based_splits(train_df["date"], n_splits=5)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
        ),
    }

    results: dict[str, dict[str, float]] = {}
    trained_pipelines: dict[str, Pipeline] = {}
    holdout_predictions = []

    for model_name, model in models.items():
        pipeline = build_pipeline(preprocessor, model)
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_splits,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results[model_name] = {
            **summarize_cv_scores(cv_scores),
            **{f"test_{metric}": value for metric, value in regression_metrics(y_test, predictions).items()},
        }
        trained_pipelines[model_name] = pipeline

        joblib.dump(pipeline, MODELS_DIR / f"{model_name}.joblib")
        holdout_predictions.append(
            pd.DataFrame(
                {
                    "date": test_df["date"].dt.strftime("%Y-%m-%d"),
                    "city": test_df["city"],
                    "model": model_name,
                    "actual_next_day_ghi": y_test,
                    "predicted_next_day_ghi": predictions,
                }
            )
        )

    best_model_name = min(
        results.keys(),
        key=lambda model_name: (results[model_name]["cv_RMSE_mean"], -results[model_name]["cv_R2_mean"]),
    )
    best_pipeline = trained_pipelines[best_model_name]
    joblib.dump(best_pipeline, MODEL_PATH)

    shap_importance = run_shap_analysis(best_pipeline, X_test)
    selected_features = get_selected_feature_names(best_pipeline).tolist()

    metrics_df = pd.DataFrame(results).T.sort_values("cv_RMSE_mean")
    metrics_df.to_csv(REPORTS_DIR / "metrics.csv")
    pd.concat(holdout_predictions, ignore_index=True).to_csv(
        REPORTS_DIR / "holdout_predictions.csv", index=False
    )
    shap_importance.head(20).to_csv(REPORTS_DIR / "top_shap_features.csv", index=False)

    metrics_payload = {
        "best_model": best_model_name,
        "holdout_year": HOLDOUT_START_YEAR,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "selected_features": selected_features,
        "results": results,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(metrics_payload, file_handle, indent=4, ensure_ascii=False)

    model_metadata = {
        "best_model": best_model_name,
        "target_column": TARGET_COLUMN,
        "holdout_year": HOLDOUT_START_YEAR,
        "feature_columns": feature_columns,
        "selected_features": selected_features,
        "test_metrics": {
            "MAE": results[best_model_name]["test_MAE"],
            "RMSE": results[best_model_name]["test_RMSE"],
            "R2": results[best_model_name]["test_R2"],
        },
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as file_handle:
        json.dump(model_metadata, file_handle, indent=4, ensure_ascii=False)

    print("Comparacao de modelos:")
    print(metrics_df[["cv_MAE_mean", "cv_RMSE_mean", "cv_R2_mean", "test_MAE", "test_RMSE", "test_R2"]])
    print(f"\nMelhor modelo: {best_model_name}")
    print("\nTop 10 variaveis por SHAP:")
    print(shap_importance.head(10))


if __name__ == "__main__":
    main()
