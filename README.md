# Space Urban Density IA

Projeto de Inteligência Artificial e Machine Learning aplicado à **Economia Espacial**.

## 1. Contexto do problema

O crescimento urbano acelerado exige formas mais inteligentes de monitorar a ocupação das cidades. Imagens de satélite podem ajudar governos, empresas de infraestrutura, mobilidade, saneamento, logística e defesa civil a entenderem onde há maior concentração de pessoas.

Este projeto propõe um pipeline de IA capaz de estimar a **densidade populacional urbana**, em pessoas por km², usando atributos extraídos de imagens de satélite e variáveis urbanas simuladas.

## 2. Fonte dos dados

Para garantir reprodutibilidade, foi criada uma base sintética com:

- 1.200 linhas;
- mais de 10 colunas;
- imagens sintéticas estilo satélite, com áreas construídas, vegetação, água e vias;
- variável alvo: `population_density`.

Arquivo gerado:

```bash
data/urban_density_dataset.csv
```

As imagens ficam em:

```bash
data/images/
```

## 3. Metodologia

O pipeline contém:

1. Geração das imagens e dados tabulares;
2. Extração de atributos visuais das imagens;
3. Pré-processamento com imputação, normalização e One Hot Encoding;
4. Treinamento de dois modelos preditivos;
5. Validação e comparação de métricas;
6. Escolha automática do melhor modelo;
7. Interpretabilidade com SHAP;
8. Deploy com Streamlit.

## 4. Variáveis principais

Exemplos de atributos usados pelo modelo:

- `built_ratio`: proporção de área construída;
- `vegetation_ratio`: proporção de vegetação;
- `water_ratio`: proporção de água;
- `road_density`: densidade de vias;
- `distance_to_center_km`: distância até o centro urbano;
- `night_light_index`: índice de luz noturna;
- `public_transport_score`: score de transporte público;
- `brightness`: brilho médio da imagem;
- `contrast`: contraste da imagem;
- `texture`: textura visual da imagem;
- `built_up_proxy`: proxy visual de concreto/telhados;
- `vegetation_index`: índice aproximado de vegetação;
- `water_index`: índice aproximado de água.

## 5. Modelos testados

Foram comparados dois modelos de regressão:

1. Random Forest Regressor;
2. Gradient Boosting Regressor.

## 6. Métricas avaliadas

As métricas usadas foram:

- MAE: erro médio absoluto;
- RMSE: raiz do erro quadrático médio;
- R²: capacidade explicativa do modelo.

Os resultados são salvos em:

```bash
reports/metrics.csv
reports/metrics.json
```

## 7. Interpretabilidade com SHAP

O SHAP foi usado para explicar quais variáveis mais influenciaram as previsões.

Arquivos gerados:

```bash
reports/shap_summary.png
reports/shap_importance.csv
```

Em geral, espera-se que variáveis como `built_ratio`, `road_density`, `night_light_index`, `vegetation_ratio` e `distance_to_center_km` estejam entre as mais importantes.

## 8. Como executar

### 8.1 Criar ambiente virtual

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

### 8.2 Instalar dependências

```bash
pip install -r requirements.txt
```

### 8.3 Gerar dataset

```bash
python src/generate_dataset.py
```

### 8.4 Treinar modelos

```bash
python src/train.py
```

### 8.5 Rodar aplicação

```bash
streamlit run src/app.py
```

## 9. Link da aplicação

Coloque aqui o link quando fizer deploy no Streamlit Cloud, Hugging Face Spaces, Render ou outro serviço.

Exemplo:

```text
https://seu-projeto.streamlit.app
```

## 10. Organização do projeto

```text
space_urban_density_ml/
├── data/
│   ├── images/
│   └── urban_density_dataset.csv
├── models/
│   ├── best_model.joblib
│   ├── random_forest.joblib
│   └── gradient_boosting.joblib
├── reports/
│   ├── metrics.csv
│   ├── metrics.json
│   ├── shap_importance.csv
│   └── shap_summary.png
├── src/
│   ├── app.py
│   ├── generate_dataset.py
│   └── train.py
├── requirements.txt
└── README.md
```

## 11. Observação importante

A base é sintética para fins acadêmicos. Em um cenário real, os dados poderiam ser substituídos por imagens de satélite reais de fontes como Sentinel, Landsat, Google Earth Engine ou APIs comerciais, e os rótulos de população poderiam vir de dados censitários ou bases como WorldPop.
