"""
Aplicacao Streamlit para prever o potencial solar do proximo dia
com base em dados da NASA POWER API.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import joblib
import pandas as pd
import streamlit as st

from project_utils import (
    DATASET_PATH,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    SHAP_IMAGE_PATH,
    TARGET_COLUMN,
    classify_solar_potential,
    engineer_features,
    estimate_pv_output,
)

st.set_page_config(page_title="Space Solar Intelligence", layout="wide")

NUMERIC_INPUT_COLUMNS = [
    "solar_irradiance_today",
    "clear_sky_irradiance_today",
    "temperature_avg_c",
    "temperature_max_c",
    "temperature_min_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "wind_speed_ms",
    "cloud_cover_pct",
    "surface_pressure_kpa",
    "clearness_index_today",
]

LABELS = {
    "solar_irradiance_today": "Irradiancia solar observada hoje (kWh/m²/dia)",
    "clear_sky_irradiance_today": "Irradiancia de ceu limpo hoje (kWh/m²/dia)",
    "temperature_avg_c": "Temperatura media hoje (C)",
    "temperature_max_c": "Temperatura maxima hoje (C)",
    "temperature_min_c": "Temperatura minima hoje (C)",
    "relative_humidity_pct": "Umidade relativa hoje (%)",
    "precipitation_mm": "Precipitacao hoje (mm)",
    "wind_speed_ms": "Velocidade do vento hoje (m/s)",
    "cloud_cover_pct": "Cobertura de nuvens hoje (%)",
    "surface_pressure_kpa": "Pressao de superficie hoje (kPa)",
    "clearness_index_today": "Indice de claridade atmosferica hoje",
}


@st.cache_data
def load_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(DATASET_PATH)
    dataset["date"] = pd.to_datetime(dataset["date"])
    return dataset


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def get_city_profile(dataset: pd.DataFrame, city: str) -> pd.Series:
    return dataset.loc[dataset["city"] == city].iloc[0]


def get_default_values(dataset: pd.DataFrame, city: str, selected_month: int) -> pd.Series:
    city_slice = dataset[dataset["city"] == city]
    monthly_slice = city_slice[city_slice["month"] == selected_month]
    reference_slice = monthly_slice if not monthly_slice.empty else city_slice
    return reference_slice[NUMERIC_INPUT_COLUMNS].median()


def build_prediction_frame(
    dataset: pd.DataFrame,
    city: str,
    reference_date: date,
    numeric_inputs: dict[str, float],
) -> pd.DataFrame:
    city_profile = get_city_profile(dataset, city)
    base_record = {
        "date": pd.Timestamp(reference_date),
        "city": city_profile["city"],
        "state": city_profile["state"],
        "region": city_profile["region"],
        "latitude": city_profile["latitude"],
        "longitude": city_profile["longitude"],
        **numeric_inputs,
    }
    feature_frame = engineer_features(pd.DataFrame([base_record]))
    return feature_frame


st.title("Space Solar Intelligence")
st.caption(
    "Previsao do potencial solar do proximo dia para cidades brasileiras com dados reais da NASA POWER API."
)

if not DATASET_PATH.exists() or not MODEL_PATH.exists() or not MODEL_METADATA_PATH.exists():
    st.error("Artefatos nao encontrados. Rode primeiro: python src/generate_dataset.py e python src/train.py")
    st.stop()

dataset = load_dataset()
model = load_model()
metadata = load_json(MODEL_METADATA_PATH)
metrics = load_json(METRICS_PATH) if METRICS_PATH.exists() else {}

cities = sorted(dataset["city"].unique().tolist())
left_column, right_column = st.columns([1.1, 1.0])

with left_column:
    st.subheader("Entrada")
    selected_city = st.selectbox("Cidade", cities)
    reference_date = st.date_input("Data de referencia", value=date.today())
    default_values = get_default_values(dataset, selected_city, reference_date.month)

    st.caption("Os valores abaixo ja comecam preenchidos com a mediana historica da cidade e do mes escolhido.")
    with st.form("prediction_form"):
        numeric_inputs = {}
        for column in NUMERIC_INPUT_COLUMNS:
            numeric_inputs[column] = st.number_input(
                LABELS[column],
                value=float(default_values[column]),
                format="%.3f",
            )
        submitted = st.form_submit_button("Prever proximo dia")

with right_column:
    st.subheader("Resultado")
    if submitted:
        prediction_frame = build_prediction_frame(dataset, selected_city, reference_date, numeric_inputs)
        model_input = prediction_frame[metadata["feature_columns"]]
        predicted_ghi = float(model.predict(model_input)[0])
        predicted_class = classify_solar_potential(predicted_ghi)
        estimated_pv_output = estimate_pv_output(predicted_ghi, numeric_inputs["temperature_avg_c"])

        forecast_date = reference_date + timedelta(days=1)
        st.metric(
            "Potencial solar previsto para o proximo dia",
            f"{predicted_ghi:.2f} kWh/m²/dia",
        )
        st.metric(
            "Energia fotovoltaica estimada",
            f"{estimated_pv_output:.2f} kWh/kWp",
        )
        st.info(f"Classificacao prevista para {forecast_date.strftime('%d/%m/%Y')}: {predicted_class.upper()}")

        st.write("Registro enviado ao modelo:")
        preview_columns = [
            "city",
            "state",
            "region",
            "solar_irradiance_today",
            "cloud_cover_pct",
            "temperature_avg_c",
            "relative_humidity_pct",
            "precipitation_mm",
            "wind_speed_ms",
            "clearness_index_today",
        ]
        st.dataframe(prediction_frame[preview_columns], use_container_width=True)
    else:
        st.write("Preencha os dados e clique em `Prever proximo dia` para executar a inferencia.")

    if metrics:
        best_model = metrics.get("best_model", metadata.get("best_model", "n/d"))
        best_metrics = metrics.get("results", {}).get(best_model, {})
        st.subheader("Desempenho do melhor modelo")
        metric_columns = st.columns(3)
        metric_columns[0].metric("MAE", f"{best_metrics.get('test_MAE', 0):.3f}")
        metric_columns[1].metric("RMSE", f"{best_metrics.get('test_RMSE', 0):.3f}")
        metric_columns[2].metric("R²", f"{best_metrics.get('test_R2', 0):.3f}")
        st.caption(f"Melhor algoritmo selecionado automaticamente: {best_model}.")

if SHAP_IMAGE_PATH.exists():
    st.subheader("Interpretabilidade global com SHAP")
    st.image(str(SHAP_IMAGE_PATH), caption="Variaveis que mais influenciam a previsao do modelo")

with st.expander("Sobre o pipeline"):
    st.markdown(
        """
        - Fonte dos dados: NASA POWER API.
        - Problema: prever a irradiancia solar do proximo dia para apoio a planejamento energetico.
        - Tecnicas comparadas: Random Forest Regressor e Gradient Boosting Regressor.
        - Etapas do pipeline: coleta via API, engenharia de atributos, selecao de atributos, validacao temporal,
          comparacao de desempenho, escolha do melhor modelo e deploy com Streamlit.
        """
    )

with st.expander("Amostra da base"):
    sample_columns = [
        "date",
        "city",
        "state",
        "region",
        "solar_irradiance_today",
        "cloud_cover_pct",
        "temperature_avg_c",
        TARGET_COLUMN,
    ]
    st.dataframe(dataset[sample_columns].head(20), use_container_width=True)
