"""
Treina e compara dois modelos para prever densidade populacional urbana.
Modelos usados:
1. Random Forest Regressor
2. Gradient Boosting Regressor

Também gera análise SHAP para interpretabilidade.
"""

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "urban_density_dataset.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "population_density"
DROP_COLS = ["image_path", "density_class", TARGET]


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def get_feature_names(preprocessor, numeric_cols, categorical_cols):
    names = list(numeric_cols)
    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        names.extend(ohe.get_feature_names_out(categorical_cols).tolist())
    return names


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError("Dataset não encontrado. Rode primeiro: python src/generate_dataset.py")

    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=DROP_COLS)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_SEED,
        ),
    }

    results = {}
    trained_pipelines = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        results[name] = regression_metrics(y_test, preds)
        trained_pipelines[name] = pipeline
        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")

    # Escolha: maior R2; se empatar, menor RMSE.
    best_model_name = sorted(
        results.keys(), key=lambda k: (-results[k]["R2"], results[k]["RMSE"])
    )[0]
    best_pipeline = trained_pipelines[best_model_name]
    joblib.dump(best_pipeline, MODELS_DIR / "best_model.joblib")

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "best_model": best_model_name}, f, indent=4, ensure_ascii=False)

    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(REPORTS_DIR / "metrics.csv")
    print("Métricas:")
    print(metrics_df)
    print(f"\nMelhor modelo: {best_model_name}")

    # SHAP no melhor modelo. Para modelos de árvore, usamos TreeExplainer no estimador final.
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    fitted_model = best_pipeline.named_steps["model"]
    X_test_transformed = fitted_preprocessor.transform(X_test)
    feature_names = get_feature_names(fitted_preprocessor, numeric_cols, categorical_cols)

    # Converte sparse para dense quando necessário.
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    sample_size = min(250, X_test_transformed.shape[0])
    X_shap = X_test_transformed[:sample_size]

    explainer = shap.TreeExplainer(fitted_model)
    shap_values = explainer.shap_values(X_shap)

    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Importância média absoluta para colocar no README/apresentação.
    shap_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)
    print("\nTop 10 variáveis por SHAP:")
    print(shap_importance.head(10))


if __name__ == "__main__":
    main()
