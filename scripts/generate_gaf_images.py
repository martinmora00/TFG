"""
generate_gaf_images.py

Genera imágenes GAF (Gramian Angular Field) 24x24 a partir del dataset
daily_24_normalized.parquet.

- Entrada:  data_processed/daily_24_normalized.parquet
- Salida:
    * data_processed/gaf_images/*.npy   (matrices 24x24 en float32)
    * data_processed/gaf_index.csv      (índice con EQUIP_ID, date, ruta_npy)
    * Opcional: algunas .png para visualización si MAKE_PNGS = True
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# CONFIGURACIÓN
# =========================

INPUT_PARQUET = Path("data_processed/daily_24_normalized.parquet")
OUTPUT_DIR = Path("data_processed/gaf_images")

# Guardar también algunas imágenes en PNG para el TFG (visualización)
MAKE_PNGS = True
MAX_PNGS = 30  # número máximo de PNGs para no llenar el disco

# Colormap para las PNG (no afecta a los .npy)
CMAP = "viridis"


# =========================
# FUNCIONES AUXILIARES
# =========================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def compute_gaf_from_series(x: np.ndarray) -> np.ndarray:
    """
    Calcula el Gramian Angular Summation Field (GASF) para una serie 1D.

    Asume que x está normalizado en [-1, 1].

    Fórmula equivalente sin usar arccos directamente:
        GASF = cos(phi_i + phi_j)
             = x_i * x_j - sqrt(1 - x_i^2) * sqrt(1 - x_j^2)
    """
    # Aseguramos rango válido numéricamente
    x = np.clip(x, -1.0, 1.0)

    # x_i * x_j
    xx = np.outer(x, x)

    # sqrt(1 - x_i^2) * sqrt(1 - x_j^2)
    sqrt_term = np.sqrt(1.0 - x**2)
    yy = np.outer(sqrt_term, sqrt_term)

    gaf = xx - yy
    return gaf.astype(np.float32)


def row_to_series(row) -> np.ndarray:
    """Extrae el vector horario h00..h23 de una fila de pandas."""
    values = [getattr(row, f"h{i:02d}") for i in range(24)]
    return np.array(values, dtype=np.float32)


# =========================
# PROCESO PRINCIPAL
# =========================

def main():
    print(f"1) Cargando dataset normalizado desde: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"   Días disponibles: {len(df)}")
    print(f"   Equipos distintos: {df['EQUIP_ID'].nunique()}")

    ensure_dir(OUTPUT_DIR)

    # Intentamos usar tqdm si está instalado (barra de progreso opcional)
    try:
        from tqdm import tqdm
        iterator = tqdm(df.itertuples(index=False), total=len(df))
    except ImportError:
        iterator = df.itertuples(index=False)

    index_rows = []
    png_count = 0

    for row in iterator:
        equip_id = row.EQUIP_ID
        date = pd.to_datetime(row.date)  # por si viene como string

        series = row_to_series(row)
        gaf = compute_gaf_from_series(series)  # matriz 24x24

        # Nombre de fichero: EQUIPID_YYYYMMDD.npy
        fname_base = f"{equip_id}_{date.strftime('%Y%m%d')}"
        npy_path = OUTPUT_DIR / f"{fname_base}.npy"

        # Guardamos la matriz GAF en formato numpy
        np.save(npy_path, gaf)

        # Guardamos el índice
        index_rows.append(
            {
                "EQUIP_ID": equip_id,
                "date": date.strftime("%Y-%m-%d"),
                "gaf_npy": str(npy_path.relative_to(OUTPUT_DIR.parent)),
            }
        )

        # Opcional: algunas PNG para visualización en el TFG
        if MAKE_PNGS and png_count < MAX_PNGS:
            try:
                import matplotlib.pyplot as plt

                png_path = OUTPUT_DIR / f"{fname_base}.png"
                plt.imshow(gaf, cmap=CMAP)
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(png_path, bbox_inches="tight", pad_inches=0)
                plt.close()
                png_count += 1
            except Exception as e:
                print(f"   Aviso: no se pudo guardar PNG para {fname_base}: {e}")

    # Guardamos índice en CSV
    index_df = pd.DataFrame(index_rows)
    index_path = OUTPUT_DIR.parent / "gaf_index.csv"
    index_df.to_csv(index_path, index=False)

    print("2) Proceso completado.")
    print(f"   GAF .npy guardados en: {OUTPUT_DIR}")
    print(f"   Índice de imágenes en: {index_path}")
    if MAKE_PNGS:
        print(f"   Imágenes PNG generadas: {png_count} (máx. {MAX_PNGS})")


if __name__ == "__main__":
    main()
