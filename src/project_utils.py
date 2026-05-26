from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DATASET_PATH = DATA_DIR / "nasa_power_solar_dataset.csv"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
SHAP_IMAGE_PATH = REPORTS_DIR / "shap_summary.png"
SHAP_IMPORTANCE_PATH = REPORTS_DIR / "shap_importance.csv"

TARGET_COLUMN = "target_next_day_ghi"
TARGET_CLASS_COLUMN = "target_next_day_solar_class"

NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_PARAMETERS = {
    "ALLSKY_SFC_SW_DWN": "solar_irradiance_today",
    "CLRSKY_SFC_SW_DWN": "clear_sky_irradiance_today",
    "T2M": "temperature_avg_c",
    "T2M_MAX": "temperature_max_c",
    "T2M_MIN": "temperature_min_c",
    "RH2M": "relative_humidity_pct",
    "PRECTOTCORR": "precipitation_mm",
    "WS2M": "wind_speed_ms",
    "CLOUD_AMT": "cloud_cover_pct",
    "PS": "surface_pressure_kpa",
    "ALLSKY_KT": "clearness_index_today",
}

CITIES = [
    {"city": "Sao Paulo", "state": "SP", "region": "sudeste", "latitude": -23.5505, "longitude": -46.6333},
    {"city": "Rio de Janeiro", "state": "RJ", "region": "sudeste", "latitude": -22.9068, "longitude": -43.1729},
    {"city": "Belo Horizonte", "state": "MG", "region": "sudeste", "latitude": -19.9167, "longitude": -43.9345},
    {"city": "Brasilia", "state": "DF", "region": "centro_oeste", "latitude": -15.7939, "longitude": -47.8828},
    {"city": "Goiania", "state": "GO", "region": "centro_oeste", "latitude": -16.6869, "longitude": -49.2648},
    {"city": "Cuiaba", "state": "MT", "region": "centro_oeste", "latitude": -15.6014, "longitude": -56.0979},
    {"city": "Campo Grande", "state": "MS", "region": "centro_oeste", "latitude": -20.4697, "longitude": -54.6201},
    {"city": "Curitiba", "state": "PR", "region": "sul", "latitude": -25.4284, "longitude": -49.2733},
    {"city": "Porto Alegre", "state": "RS", "region": "sul", "latitude": -30.0346, "longitude": -51.2177},
    {"city": "Florianopolis", "state": "SC", "region": "sul", "latitude": -27.5949, "longitude": -48.5482},
    {"city": "Salvador", "state": "BA", "region": "nordeste", "latitude": -12.9777, "longitude": -38.5016},
    {"city": "Recife", "state": "PE", "region": "nordeste", "latitude": -8.0476, "longitude": -34.8770},
    {"city": "Fortaleza", "state": "CE", "region": "nordeste", "latitude": -3.7319, "longitude": -38.5267},
    {"city": "Natal", "state": "RN", "region": "nordeste", "latitude": -5.7945, "longitude": -35.2110},
    {"city": "Manaus", "state": "AM", "region": "norte", "latitude": -3.1190, "longitude": -60.0217},
    {"city": "Belem", "state": "PA", "region": "norte", "latitude": -1.4558, "longitude": -48.4902},
]


def classify_solar_potential(value: float) -> str:
    if value < 4.0:
        return "baixo"
    if value < 5.5:
        return "medio"
    return "alto"


def estimate_pv_output(ghi_value: float, temperature_avg_c: float) -> float:
    derating = max(temperature_avg_c - 25.0, 0.0) * 0.004
    efficiency_factor = max(0.70, 0.86 - derating)
    return max(0.0, ghi_value * efficiency_factor)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["dayofyear"] = data["date"].dt.dayofyear

    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12.0)
    data["dayofyear_sin"] = np.sin(2 * np.pi * data["dayofyear"] / 366.0)
    data["dayofyear_cos"] = np.cos(2 * np.pi * data["dayofyear"] / 366.0)

    data["temperature_range_c"] = data["temperature_max_c"] - data["temperature_min_c"]
    data["clear_sky_gap"] = data["clear_sky_irradiance_today"] - data["solar_irradiance_today"]
    data["irradiance_utilization_ratio"] = (
        data["solar_irradiance_today"] / data["clear_sky_irradiance_today"].replace(0, np.nan)
    )
    data["storm_intensity_proxy"] = data["cloud_cover_pct"] * data["precipitation_mm"]
    data["heat_humidity_index"] = data["temperature_avg_c"] * (data["relative_humidity_pct"] / 100.0)
    data["wind_cloud_interaction"] = data["wind_speed_ms"] * (data["cloud_cover_pct"] / 100.0)
    data["rain_indicator"] = np.where(data["precipitation_mm"] >= 1.0, "yes", "no")
    return data
