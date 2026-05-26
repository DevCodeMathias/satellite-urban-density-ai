"""
Gera uma base sintética com imagens de satélite simuladas e atributos numéricos.
Tema: estimar densidade populacional urbana a partir de sinais visuais de imagens orbitais.

Saída:
- data/urban_density_dataset.csv com 1.200 linhas e 15+ colunas
- data/images/*.png com imagens 64x64 simulando áreas urbanas, vegetação e água
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

RANDOM_SEED = 42
N_SAMPLES = 1200
IMG_SIZE = 64
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"

rng = np.random.default_rng(RANDOM_SEED)


def clamp(value, min_value=0, max_value=255):
    return int(max(min_value, min(max_value, value)))


def create_satellite_like_image(built_ratio: float, vegetation_ratio: float, water_ratio: float, road_density: float, idx: int):
    """Cria imagem sintética parecida com patch de satélite urbano."""
    # Fundo: solo / vegetação leve
    base_green = clamp(90 + 110 * vegetation_ratio + rng.normal(0, 8))
    base_red = clamp(95 - 35 * vegetation_ratio + rng.normal(0, 8))
    base_blue = clamp(80 + 55 * water_ratio + rng.normal(0, 8))
    image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (base_red, base_green, base_blue))
    draw = ImageDraw.Draw(image)

    # Áreas de água
    water_shapes = int(water_ratio * 8)
    for _ in range(water_shapes):
        x0, y0 = rng.integers(0, IMG_SIZE - 10, size=2)
        x1 = clamp(x0 + rng.integers(8, 25), 0, IMG_SIZE)
        y1 = clamp(y0 + rng.integers(8, 25), 0, IMG_SIZE)
        draw.ellipse([int(x0), int(y0), int(x1), int(y1)], fill=(40, 85, 135))

    # Construções: retângulos cinzas/brancos/marrons
    n_buildings = int(5 + built_ratio * 55)
    for _ in range(n_buildings):
        w = int(rng.integers(3, 11))
        h = int(rng.integers(3, 11))
        x = int(rng.integers(0, IMG_SIZE - w))
        y = int(rng.integers(0, IMG_SIZE - h))
        shade = clamp(115 + rng.normal(0, 35))
        draw.rectangle([x, y, x + w, y + h], fill=(shade, shade, clamp(shade - 10)))

    # Vias: linhas claras/escuras cruzando a imagem
    n_roads = int(1 + road_density * 10)
    for _ in range(n_roads):
        if rng.random() > 0.5:
            y = int(rng.integers(0, IMG_SIZE))
            draw.line([(0, y), (IMG_SIZE, clamp(y + rng.integers(-8, 9), 0, IMG_SIZE))], fill=(70, 70, 70), width=int(rng.integers(1, 3)))
        else:
            x = int(rng.integers(0, IMG_SIZE))
            draw.line([(x, 0), (clamp(x + rng.integers(-8, 9), 0, IMG_SIZE), IMG_SIZE)], fill=(70, 70, 70), width=int(rng.integers(1, 3)))

    # Pequeno ruído visual para textura
    arr = np.asarray(image).astype(np.int16)
    noise = rng.normal(0, 10, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr)

    path = IMG_DIR / f"sat_patch_{idx:04d}.png"
    image.save(path)
    return path


def extract_image_features(image_path: Path):
    """Extrai features simples da imagem, como se fosse um pré-processamento de sensoriamento remoto."""
    arr = np.asarray(Image.open(image_path).convert("RGB")).astype(float)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]

    brightness = arr.mean()
    contrast = arr.std()
    mean_red = red.mean()
    mean_green = green.mean()
    mean_blue = blue.mean()

    # Índices aproximados usando RGB comum, apenas para projeto didático.
    vegetation_index = (green - red).mean() / ((green + red).mean() + 1e-6)
    water_index = (blue - red).mean() / ((blue + red).mean() + 1e-6)
    built_up_proxy = ((red + blue) / 2 - green).mean() / (arr.mean() + 1e-6)

    # Textura simples: diferença entre pixels vizinhos.
    horizontal_texture = np.abs(arr[:, 1:, :] - arr[:, :-1, :]).mean()
    vertical_texture = np.abs(arr[1:, :, :] - arr[:-1, :, :]).mean()
    texture = (horizontal_texture + vertical_texture) / 2

    # Proporção de pixels muito cinzas, proxy para telhados/concreto.
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


def main():
    DATA_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    zones = ["centro", "residencial", "industrial", "periurbano", "parque_urbano"]

    for i in range(N_SAMPLES):
        zone = rng.choice(zones, p=[0.22, 0.34, 0.16, 0.18, 0.10])

        if zone == "centro":
            built_ratio = rng.uniform(0.65, 0.95)
            vegetation_ratio = rng.uniform(0.02, 0.20)
            road_density = rng.uniform(0.55, 0.95)
            water_ratio = rng.uniform(0.00, 0.08)
        elif zone == "residencial":
            built_ratio = rng.uniform(0.35, 0.70)
            vegetation_ratio = rng.uniform(0.15, 0.45)
            road_density = rng.uniform(0.35, 0.75)
            water_ratio = rng.uniform(0.00, 0.10)
        elif zone == "industrial":
            built_ratio = rng.uniform(0.45, 0.80)
            vegetation_ratio = rng.uniform(0.05, 0.30)
            road_density = rng.uniform(0.30, 0.70)
            water_ratio = rng.uniform(0.00, 0.10)
        elif zone == "periurbano":
            built_ratio = rng.uniform(0.10, 0.40)
            vegetation_ratio = rng.uniform(0.35, 0.75)
            road_density = rng.uniform(0.10, 0.45)
            water_ratio = rng.uniform(0.00, 0.12)
        else:  # parque_urbano
            built_ratio = rng.uniform(0.02, 0.25)
            vegetation_ratio = rng.uniform(0.60, 0.95)
            road_density = rng.uniform(0.02, 0.25)
            water_ratio = rng.uniform(0.02, 0.20)

        distance_to_center_km = float(np.clip(18 * (1 - built_ratio) + rng.normal(0, 2.5), 0.2, 30))
        night_light_index = float(np.clip(0.15 + 0.75 * built_ratio + 0.25 * road_density + rng.normal(0, 0.08), 0, 1))
        public_transport_score = float(np.clip(0.10 + 0.55 * road_density + 0.35 * built_ratio + rng.normal(0, 0.10), 0, 1))

        image_path = create_satellite_like_image(built_ratio, vegetation_ratio, water_ratio, road_density, i)
        features = extract_image_features(image_path)

        # Target sintético: pessoas por km².
        # Construção alta, vias e luz noturna aumentam a densidade.
        # Vegetação, água e distância ao centro reduzem.
        density = (
            900
            + 9800 * built_ratio
            + 2400 * road_density
            + 2600 * night_light_index
            + 1200 * public_transport_score
            - 3600 * vegetation_ratio
            - 2800 * water_ratio
            - 90 * distance_to_center_km
            + rng.normal(0, 650)
        )
        population_density = float(np.clip(density, 100, 22000))

        # Classe derivada para uso opcional em classificação.
        if population_density < 3500:
            density_class = "baixa"
        elif population_density < 9000:
            density_class = "media"
        else:
            density_class = "alta"

        row = {
            "image_path": str(image_path.relative_to(ROOT)),
            "zone_type": zone,
            "built_ratio": built_ratio,
            "vegetation_ratio": vegetation_ratio,
            "water_ratio": water_ratio,
            "road_density": road_density,
            "distance_to_center_km": distance_to_center_km,
            "night_light_index": night_light_index,
            "public_transport_score": public_transport_score,
            "population_density": population_density,
            "density_class": density_class,
        }
        row.update(features)
        rows.append(row)

    df = pd.DataFrame(rows)
    output = DATA_DIR / "urban_density_dataset.csv"
    df.to_csv(output, index=False)
    print(f"Dataset salvo em: {output}")
    print(df.shape)
    print(df.head())


if __name__ == "__main__":
    main()
