# Space Solar Intelligence

Pipeline completo de Inteligencia Artificial e Machine Learning aplicado a um problema real da Economia Espacial: **prever o potencial solar do proximo dia em grandes centros urbanos brasileiros usando dados orbitais da NASA POWER API**.

## 1. Contexto do problema

A Economia Espacial nao se resume a foguetes e satelites. Um dos seus pilares mais importantes e o uso de dados de observacao da Terra para apoiar decisoes em energia, planejamento urbano, agricultura, seguros e infraestrutura.

Neste projeto, a aplicacao pratica foi direcionada para o setor de energia: prever a irradiancia solar do proximo dia para apoiar planejamento de geracao fotovoltaica, operacao de microgrids, dimensionamento de baterias e analise de risco energetico urbano.

## 2. Objetivo do projeto

Construir um pipeline end-to-end de IA/ML que:

- colete dados reais por API;
- realize pre-processamento, engenharia e selecao de atributos;
- treine e compare pelo menos dois modelos preditivos;
- escolha automaticamente o melhor modelo;
- explique as previsoes com SHAP;
- disponibilize uma aplicacao interativa em Streamlit.

## 3. Fonte dos dados

Foi utilizada a **NASA POWER API**:

- Endpoint: `https://power.larc.nasa.gov/api/temporal/daily/point`
- Comunidade: `RE` (Renewable Energy)
- Periodo coletado: `2022-01-01` ate `2025-12-31`
- Cobertura geografica: 16 capitais/cidades brasileiras
- Tamanho final da base: **23.360 linhas e 35 colunas**

### Variaveis coletadas via API

- `ALLSKY_SFC_SW_DWN` -> `solar_irradiance_today`
- `CLRSKY_SFC_SW_DWN` -> `clear_sky_irradiance_today`
- `T2M` -> `temperature_avg_c`
- `T2M_MAX` -> `temperature_max_c`
- `T2M_MIN` -> `temperature_min_c`
- `RH2M` -> `relative_humidity_pct`
- `PRECTOTCORR` -> `precipitation_mm`
- `WS2M` -> `wind_speed_ms`
- `CLOUD_AMT` -> `cloud_cover_pct`
- `PS` -> `surface_pressure_kpa`
- `ALLSKY_KT` -> `clearness_index_today`

### Cidades usadas

Sao Paulo, Rio de Janeiro, Belo Horizonte, Brasilia, Goiania, Cuiaba, Campo Grande, Curitiba, Porto Alegre, Florianopolis, Salvador, Recife, Fortaleza, Natal, Manaus e Belem.

## 4. Problema preditivo

O alvo do modelo e:

- `target_next_day_ghi`: irradiancia solar total do **proximo dia** em `kWh/m²/dia`

Tambem foi gerada uma variavel derivada de negocio:

- `target_next_day_pv_output_kwh_kwp`: estimativa simplificada de energia fotovoltaica do proximo dia

## 5. Pipeline de Machine Learning

O pipeline implementado atende ao descritivo do trabalho:

1. **Obtencao dos dados por API**
   - coleta automatica de dados diarios da NASA POWER API.
2. **Pre-processamento**
   - imputacao de faltantes;
   - padronizacao de atributos numericos;
   - One Hot Encoding para variaveis categoricas.
3. **Engenharia de atributos**
   - variaveis ciclicas (`month_sin`, `month_cos`, `dayofyear_sin`, `dayofyear_cos`);
   - `temperature_range_c`;
   - `clear_sky_gap`;
   - `irradiance_utilization_ratio`;
   - `storm_intensity_proxy`;
   - `heat_humidity_index`;
   - `wind_cloud_interaction`;
   - `rain_indicator`.
4. **Selecao de atributos**
   - `SelectKBest` com `mutual_info_regression`.
5. **Treinamento de modelos**
   - `RandomForestRegressor`
   - `GradientBoostingRegressor`
6. **Validacao e comparacao**
   - validacao temporal com janelas crescentes no periodo de treino;
   - holdout final em `2025`.
7. **Escolha do melhor modelo**
   - criterio principal: menor `CV RMSE`.
8. **Interpretabilidade**
   - explicacao global com SHAP.
9. **Deploy**
   - aplicacao interativa em Streamlit.

## 6. Resultados obtidos

Treino: `2022-2024`  
Teste holdout: `2025`

| Modelo | CV MAE | CV RMSE | CV R2 | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8263 | 1.0957 | 0.4991 | 0.7665 | 1.0350 | 0.5562 |
| Random Forest | 0.8362 | 1.1074 | 0.4886 | 0.7728 | 1.0458 | 0.5470 |

### Melhor modelo

- Melhor algoritmo selecionado: `GradientBoostingRegressor`

Arquivos gerados:

- `models/best_model.joblib`
- `models/model_metadata.json`
- `reports/metrics.csv`
- `reports/metrics.json`
- `reports/holdout_predictions.csv`

## 7. Interpretabilidade com SHAP

O SHAP foi utilizado para explicar a influencia das variaveis nas previsoes do melhor modelo.

### Principais variaveis encontradas

1. `solar_irradiance_today`
2. `clear_sky_irradiance_today`
3. `cloud_cover_pct`
4. `latitude`
5. `irradiance_utilization_ratio`
6. `relative_humidity_pct`
7. `surface_pressure_kpa`
8. `dayofyear`

Arquivos gerados:

- `reports/shap_summary.png`
- `reports/shap_importance.csv`

Leitura de negocio:

- irradiancia observada hoje e irradiancia de ceu limpo sao os maiores sinais para prever o proximo dia;
- cobertura de nuvens e umidade derrubam o potencial solar;
- latitude e sazonalidade ajudam a capturar padroes regionais e de estacao do ano.

## 8. Aplicacao

A aplicacao foi criada em **Streamlit** e permite:

- escolher a cidade;
- informar os indicadores climaticos observados hoje;
- prever a irradiancia do proximo dia;
- estimar a energia fotovoltaica esperada;
- visualizar a interpretabilidade global com SHAP.

### Link da aplicacao em funcionamento

Depois de executar localmente, a aplicacao fica disponivel em:

- `http://localhost:8501`

Se quiser publicar externamente, voce pode subir a mesma app no Streamlit Community Cloud, Render ou expor localmente com Ngrok.

## 9. Como executar

### 9.1 Criar ambiente virtual

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### 9.2 Instalar dependencias

```bash
pip install -r requirements.txt
```

### 9.3 Coletar os dados via API

```bash
python src/generate_dataset.py
```

Saida principal:

```text
data/nasa_power_solar_dataset.csv
```

### 9.4 Treinar os modelos

```bash
python src/train.py
```

### 9.5 Rodar a aplicacao

```bash
streamlit run src/app.py
```

## 10. Estrutura do projeto

```text
space_urban_density_ml/
|-- data/
|   |-- nasa_power_solar_dataset.csv
|-- models/
|   |-- best_model.joblib
|   |-- gradient_boosting.joblib
|   |-- random_forest.joblib
|   |-- model_metadata.json
|-- reports/
|   |-- metrics.csv
|   |-- metrics.json
|   |-- holdout_predictions.csv
|   |-- shap_importance.csv
|   |-- shap_summary.png
|   |-- top_shap_features.csv
|-- src/
|   |-- app.py
|   |-- generate_dataset.py
|   |-- project_utils.py
|   |-- train.py
|-- requirements.txt
|-- README.md
```

## 11. Reprodutibilidade

O projeto esta totalmente reproduzivel:

- codigo fonte em Python no repositorio;
- coleta automatica dos dados via API;
- treinamento deterministico com `random_state`;
- artefatos de modelo, metricas e interpretabilidade gerados por script;
- aplicacao local pronta para execucao.
