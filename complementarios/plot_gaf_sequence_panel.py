import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data_processed")
GAF_DIR = DATA_DIR / "gaf_images"
INDEX_PATH = DATA_DIR / "vit_index.csv"

WINDOW = 7  # número de días previos
EQUIP_ID = 63  


def load_index_for_equip(equip_id: int):
    df = pd.read_csv(INDEX_PATH)
    df_e = df[df["EQUIP_ID"] == equip_id].sort_values("date").reset_index(drop=True)
    return df_e


def get_last_window_dates(df_e: pd.DataFrame):
    """
    Devuelve:
      - lista de fechas de los 7 días previos
      - fecha objetivo (día siguiente)
    usando la última ventana posible en el índice.
    """
    if len(df_e) <= WINDOW:
        raise ValueError("No hay suficientes días para construir una ventana.")

    start_idx = len(df_e) - (WINDOW + 1)
    end_idx = start_idx + WINDOW
    target_idx = end_idx

    prev_rows = df_e.iloc[start_idx:end_idx]
    target_row = df_e.iloc[target_idx]

    prev_dates = list(prev_rows["date"])
    target_date = target_row["date"]

    return prev_dates, target_date


def load_gaf(equip_id: int, date_str: str):
    """
    Carga la GAF .npy para un equip_id y una fecha (YYYY-MM-DD).
    El fichero esperado es: gaf_images/{equip_id}_{YYYYMMDD}.npy
    """
    date_compact = date_str.replace("-", "")
    filename = f"{equip_id}_{date_compact}.npy"
    gaf_path = GAF_DIR / filename
    print("Cargando GAF:", gaf_path)
    gaf = np.load(gaf_path)
    return gaf


def main():
    df_e = load_index_for_equip(EQUIP_ID)
    print(f"Días disponibles para EQUIP_ID={EQUIP_ID}: {len(df_e)}")

    prev_dates, target_date = get_last_window_dates(df_e)

    print("Fechas de la ventana:")
    for i, d in enumerate(prev_dates):
        print(f" t-{WINDOW - i}: {d}")
    print("Fecha objetivo (t+1):", target_date)

    # Cargamos GAFs
    gaf_prev = [load_gaf(EQUIP_ID, d) for d in prev_dates]
    gaf_target = load_gaf(EQUIP_ID, target_date)

    # Panel 2x4: 7 previos + objetivo
    plt.figure(figsize=(12, 6))

    all_gafs = gaf_prev + [gaf_target]
    titles = [f"{d}\n(t-{WINDOW - i})" for i, d in enumerate(prev_dates)]
    titles.append(f"{target_date}\n(t+1 objetivo)")

    for idx, (gaf, title) in enumerate(zip(all_gafs, titles), start=1):
        ax = plt.subplot(2, 4, idx)
        ax.imshow(gaf, cmap="viridis")
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    out_path = DATA_DIR / f"panel_gaf_seq_{EQUIP_ID}_{target_date.replace('-','')}.png"
    plt.savefig(out_path, dpi=200)
    print("Panel guardado en:", out_path)


if __name__ == "__main__":
    main()
