"""
Aplicação Streamlit para estimar densidade populacional urbana.
Execute com:
streamlit run src/app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
SHAP_IMAGE = ROOT / "reports" / "shap_summary.png"
DATASET_PATH = ROOT / "data" / "urban_density_dataset.csv"

st.set_page_config(page_title="Space Urban Density IA", page_icon="🛰️", layout="wide")


def extract_image_features(image):
    arr = np.asarray(image.convert("RGB").resize((64, 64))).astype(float)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]

    brightness = arr.mean()
    contrast = arr.std()
    mean_red = red.mean()
    mean_green = green.mean()
    mean_blue = blue.mean()
    vegetation_index = (green - red).mean() / ((green + red).mean() + 1e-6)
    water_index = (blue - red).mean() / ((blue + red).mean() + 1e-6)
    built_up_proxy = ((red + blue) / 2 - green).mean() / (arr.mean() + 1e-6)
    horizontal_texture = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
    vertical_texture = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean()
    texture = (horizontal_texture + vertical_texture) / 2
    grayness = 1 - (np.std(arr, axis=2).mean() / 128)
    grayness = float(np.clip(grayness, 0, 1))

    return {
        "mean_red": mean_red,
        "mean_green": mean_green,
        "mean_blue": mean_blue,
        "brightness": brightness,
        "contrast": contrast,
        "vegetation_index": vegetation_index,
        "water_index": water_index,
        "built_up_proxy": built_up_proxy,
        "texture": texture,
        "grayness": grayness,
    }


def classify_density(value):
    if value < 3500:
        return "Baixa densidade"
    if value < 9000:
        return "Média densidade"
    return "Alta densidade"


st.title("🛰️ Estimativa de Densidade Populacional Urbana com IA")
st.markdown(
    "Projeto de Economia Espacial: o modelo usa atributos extraídos de imagens de satélite "
    "para estimar a densidade de pessoas em áreas urbanas."
)

if not MODEL_PATH.exists():
    st.error("Modelo não encontrado. Rode primeiro: python src/generate_dataset.py e python src/train.py")
    st.stop()

model = joblib.load(MODEL_PATH)

left, right = st.columns([1, 1])

with left:
    st.subheader("Entrada")
    uploaded_file = st.file_uploader("Envie uma imagem de satélite ou use os controles manuais", type=["png", "jpg", "jpeg"])

    zone_type = st.selectbox("Tipo de zona urbana", ["centro", "residencial", "industrial", "periurbano", "parque_urbano"])
    built_ratio = st.slider("Proporção construída", 0.0, 1.0, 0.55, 0.01)
    vegetation_ratio = st.slider("Proporção de vegetação", 0.0, 1.0, 0.25, 0.01)
    water_ratio = st.slider("Proporção de água", 0.0, 1.0, 0.03, 0.01)
    road_density = st.slider("Densidade de vias", 0.0, 1.0, 0.50, 0.01)
    distance_to_center_km = st.slider("Distância até o centro urbano (km)", 0.0, 30.0, 8.0, 0.1)
    night_light_index = st.slider("Índice de luz noturna", 0.0, 1.0, 0.55, 0.01)
    public_transport_score = st.slider("Score de transporte público", 0.0, 1.0, 0.50, 0.01)

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem enviada", use_container_width=True)
        image_features = extract_image_features(image)
    else:
        # Valores neutros aproximados quando não há imagem.
        image_features = {
            "mean_red": 110,
            "mean_green": 110,
            "mean_blue": 100,
            "brightness": 106,
            "contrast": 42,
            "vegetation_index": 0.05,
            "water_index": -0.02,
            "built_up_proxy": 0.08,
            "texture": 24,
            "grayness": 0.75,
        }

input_data = {
    "zone_type": zone_type,
    "built_ratio": built_ratio,
    "vegetation_ratio": vegetation_ratio,
    "water_ratio": water_ratio,
    "road_density": road_density,
    "distance_to_center_km": distance_to_center_km,
    "night_light_index": night_light_index,
    "public_transport_score": public_transport_score,
    **image_features,
}

input_df = pd.DataFrame([input_data])
prediction = float(model.predict(input_df)[0])

with right:
    st.subheader("Resultado")
    st.metric("Densidade estimada", f"{prediction:,.0f} pessoas/km²".replace(",", "."))
    st.info(classify_density(prediction))

    st.write("Dados usados pelo modelo:")
    st.dataframe(input_df, use_container_width=True)

    if SHAP_IMAGE.exists():
        st.subheader("Interpretabilidade global com SHAP")
        st.image(str(SHAP_IMAGE), caption="Variáveis que mais influenciaram as previsões do modelo")
with st.expander("Sobre o projeto"):
    st.markdown(
        """
        **Problema:** estimar a densidade populacional urbana usando sinais visuais e urbanos associados a imagens de satélite.

        **Técnicas testadas:** Random Forest Regressor e Gradient Boosting Regressor.

        **Pipeline:** geração/obtenção dos dados, extração de atributos da imagem, pré-processamento, treinamento, validação, comparação de métricas, escolha do melhor modelo e deploy com Streamlit.
        """
    )

if DATASET_PATH.exists():
    with st.expander("Amostra da base de dados"):
        df = pd.read_csv(DATASET_PATH).head(20)
        st.dataframe(df, use_container_width=True)
