"""
Coleta dados reais da NASA POWER API para montar uma base de Machine Learning
relacionada a Economia Espacial.

Problema do projeto:
prever o potencial solar do proximo dia em grandes centros urbanos brasileiros
usando dados diarios de irradiancia e clima derivados de satelites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from project_utils import (
    CITIES,
    DATASET_PATH,
    DATA_DIR,
    NASA_POWER_BASE_URL,
    NASA_POWER_PARAMETERS,
    TARGET_CLASS_COLUMN,
    TARGET_COLUMN,
    classify_solar_potential,
    estimate_pv_output,
    engineer_features,
)

START_DATE = "20220101"
END_DATE = "20251231"
TIMEOUT_SECONDS = 60


def fetch_city_data(city_metadata: dict[str, object]) -> pd.DataFrame:
    params = {
        "parameters": ",".join(NASA_POWER_PARAMETERS.keys()),
        "community": "RE",
        "longitude": city_metadata["longitude"],
        "latitude": city_metadata["latitude"],
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
    }
    response = requests.get(NASA_POWER_BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()

    payload = response.json()
    raw_parameters = payload["properties"]["parameter"]
    city_frame = pd.DataFrame(
        {column_name: pd.Series(date_values) for column_name, date_values in raw_parameters.items()}
    ).reset_index()
    city_frame = city_frame.rename(columns={"index": "date", **NASA_POWER_PARAMETERS})
    city_frame = city_frame.replace(-999.0, np.nan)
    for column in NASA_POWER_PARAMETERS.values():
        city_frame[column] = pd.to_numeric(city_frame[column], errors="coerce")
    for key, value in city_metadata.items():
        city_frame[key] = value
    return city_frame


def build_dataset() -> pd.DataFrame:
    raw_frames = []
    for city_metadata in CITIES:
        print(f"Coletando dados de {city_metadata['city']}...")
        raw_frames.append(fetch_city_data(city_metadata))

    dataset = pd.concat(raw_frames, ignore_index=True)
    dataset = engineer_features(dataset)
    dataset = dataset.sort_values(["city", "date"]).reset_index(drop=True)

    dataset[TARGET_COLUMN] = dataset.groupby("city")["solar_irradiance_today"].shift(-1)
    dataset["target_next_day_temperature_avg_c"] = dataset.groupby("city")["temperature_avg_c"].shift(-1)
    dataset["target_next_day_pv_output_kwh_kwp"] = dataset.apply(
        lambda row: estimate_pv_output(row[TARGET_COLUMN], row["target_next_day_temperature_avg_c"])
        if pd.notna(row[TARGET_COLUMN]) and pd.notna(row["target_next_day_temperature_avg_c"])
        else pd.NA,
        axis=1,
    )
    dataset[TARGET_CLASS_COLUMN] = dataset[TARGET_COLUMN].apply(
        lambda value: classify_solar_potential(value) if pd.notna(value) else pd.NA
    )

    dataset = dataset.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    dataset["date"] = dataset["date"].dt.strftime("%Y-%m-%d")
    return dataset


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    dataset = build_dataset()
    dataset.to_csv(DATASET_PATH, index=False)

    print(f"\nDataset salvo em: {DATASET_PATH}")
    print(f"Linhas e colunas: {dataset.shape}")
    print("\nColunas:")
    print(list(dataset.columns))
    print("\nAmostra:")
    print(dataset.head())


if __name__ == "__main__":
    main()
