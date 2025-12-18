"""
visualize_gaf_example.py

Genera una figura con:
- la serie temporal diaria normalizada (24 valores)
- la imagen GAF correspondiente (24x24)

Ejemplo concreto: EQUIP_ID = 63, fecha = 2022-12-19.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Parámetros del ejemplo
EQUIP_ID = 63
DATE_STR = "2022-12-19"   # formato YYYY-MM-DD

PARQUET_PATH = Path("data_processed/daily_24_normalized.parquet")
GAF_NPY_PATH = Path("data_processed/gaf_images") / "63_20221219.npy"
OUTPUT_FIG = Path("data_processed") / "example_63_20221219_serie_gaf.png"


def load_daily_series(equip_id: int, date_str: str) -> pd.Series:
    df = pd.read_parquet(PARQUET_PATH)
    row = df[(df["EQUIP_ID"] == equip_id) & (df["date"] == date_str)].iloc[0]
    values = [row[f"h{i:02d}"] for i in range(24)]
    return pd.Series(values, index=range(24))


def main():
    print("Cargando serie diaria normalizada...")
    serie = load_daily_series(EQUIP_ID, DATE_STR)

    print("Cargando GAF desde:", GAF_NPY_PATH)
    gaf = np.load(GAF_NPY_PATH)

    print("Serie shape:", serie.shape)
    print("GAF shape:", gaf.shape)

    # Figura conjunta
    plt.figure(figsize=(10, 4))

    # Subgráfico 1: serie temporal
    plt.subplot(1, 2, 1)
    plt.plot(serie.index, serie.values, marker="o")
    plt.title(f"Serie normalizada\nEQUIP_ID {EQUIP_ID} - {DATE_STR}")
    plt.xlabel("Hora del día")
    plt.ylabel("Valor normalizado [-1, 1]")
    plt.grid(True)

    # Subgráfico 2: imagen GAF
    plt.subplot(1, 2, 2)
    plt.imshow(gaf, cmap="viridis")
    plt.title("Imagen GAF 24×24")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150)
    print("Figura guardada en:", OUTPUT_FIG)



if __name__ == "__main__":
    main()
