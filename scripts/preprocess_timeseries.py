"""
preprocess_timeseries.py

Script para:
1. Cargar el DataSet.csv original.
2. Re-muestrear a frecuencia horaria (suma).
3. Construir días completos (24 valores por día).
4. Filtrar equipos con suficientes días.
5. Normalizar cada día a [-1, 1].
6. Guardar cada paso en ficheros separados para trazabilidad.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# =========================
# CONFIGURACIÓN
# =========================

# Ruta al CSV original
INPUT_CSV = Path("data_raw/DataSet.csv")

# Carpeta de salida
OUTPUT_DIR = Path("data_processed")

# Mínimo de días completos por equipo para considerarlo en el dataset
MIN_DAYS_PER_EQUIP = 100


# =========================
# FUNCIONES AUXILIARES
# =========================

def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Carga el CSV original, convierte fechas y ordena."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["EQUIP_ID", "date"]).reset_index(drop=True)
    return df


def build_daily_24(df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir del dataframe original, genera un dataframe con:
    - EQUIP_ID
    - date (solo fecha, sin hora)
    - h00 ... h23 (valores horarios, suma de REALINCREMENT)
    Solo incluye días con 24 horas completas.
    """
    rows = []

    for equip_id, group in df.groupby("EQUIP_ID"):
        # Resample a 1H sumando REALINCREMENT
        g = group.set_index("date").resample("1h").sum()

        # Agrupamos por día (solo la parte de fecha)
        daily = g.groupby(g.index.date)

        for day, grp in daily:
            if len(grp) != 24:
                # descartamos días incompletos
                continue

            values = grp["REALINCREMENT"].values
            row = {
                "EQUIP_ID": equip_id,
                "date": pd.to_datetime(day)
            }
            for i, v in enumerate(values):
                row[f"h{i:02d}"] = v

            rows.append(row)

    daily_df = pd.DataFrame(rows)
    daily_df = daily_df.sort_values(["EQUIP_ID", "date"]).reset_index(drop=True)
    return daily_df


def filter_equipments_by_days(daily_df: pd.DataFrame,
                              min_days: int) -> pd.DataFrame:
    """
    Filtra el dataframe de días para quedarse solo con los equipos
    que tienen al menos 'min_days' días completos.
    """
    counts = daily_df.groupby("EQUIP_ID")["date"].nunique()
    valid_ids = counts[counts >= min_days].index
    filtered = daily_df[daily_df["EQUIP_ID"].isin(valid_ids)].copy()
    filtered = filtered.sort_values(["EQUIP_ID", "date"]).reset_index(drop=True)
    return filtered


def normalize_daily_rows(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalización por día a [-1, 1].
    Para cada fila (día), se normalizan las columnas h00..h23.
    Si todos los valores de un día son iguales (por ejemplo todo ceros),
    se deja el vector a todo ceros.
    """
    hour_cols = [f"h{i:02d}" for i in range(24)]

    values = daily_df[hour_cols].to_numpy(dtype=float)
    v_min = values.min(axis=1, keepdims=True)
    v_max = values.max(axis=1, keepdims=True)

    # Máscara de días planos (max == min)
    flat_rows = (v_max == v_min).ravel()

    # Inicializamos con ceros
    norm_values = np.zeros_like(values)

    # Para días no planos aplicamos escala a [-1, 1]
    non_flat = ~flat_rows
    if np.any(non_flat):
        vals_nf = values[non_flat]
        v_min_nf = v_min[non_flat]
        v_max_nf = v_max[non_flat]
        norm_nf = 2 * (vals_nf - v_min_nf) / (v_max_nf - v_min_nf) - 1
        norm_values[non_flat] = norm_nf

    normalized_df = daily_df.copy()
    normalized_df[hour_cols] = norm_values

    return normalized_df


def ensure_output_dir(path: Path):
    """Crea la carpeta de salida si no existe."""
    path.mkdir(parents=True, exist_ok=True)


# =========================
# PROCESO PRINCIPAL
# =========================

def main():
    ensure_output_dir(OUTPUT_DIR)

    print(f"1) Cargando dataset original desde: {INPUT_CSV}")
    df = load_raw_dataset(INPUT_CSV)
    print(f"   Registros cargados: {len(df)}")

    print("2) Construyendo días de 24 horas por EQUIP_ID...")
    daily_df = build_daily_24(df)
    print(f"   Días completos generados: {len(daily_df)}")

    # Guardamos el primer dataset intermedio
    raw_daily_path = OUTPUT_DIR / "daily_24_raw.parquet"
    print(f"   Guardando daily_24_raw en: {raw_daily_path}")
    daily_df.to_parquet(raw_daily_path, index=False)

    # También podemos guardar un resumen de días por equipo
    counts = daily_df.groupby("EQUIP_ID")["date"].nunique().reset_index()
    counts_path = OUTPUT_DIR / "equip_day_counts.csv"
    print(f"   Guardando resumen de días por equipo en: {counts_path}")
    counts.to_csv(counts_path, index=False)

    print(f"3) Filtrando equipos con al menos {MIN_DAYS_PER_EQUIP} días...")
    daily_filtered = filter_equipments_by_days(daily_df, MIN_DAYS_PER_EQUIP)
    print(f"   Días restantes tras filtrar: {len(daily_filtered)}")
    print(f"   Equipos válidos: {daily_filtered['EQUIP_ID'].nunique()}")

    filtered_path = OUTPUT_DIR / "daily_24_filtered.parquet"
    print(f"   Guardando daily_24_filtered en: {filtered_path}")
    daily_filtered.to_parquet(filtered_path, index=False)

    print("4) Normalizando cada día a [-1, 1]...")
    daily_normalized = normalize_daily_rows(daily_filtered)

    normalized_path = OUTPUT_DIR / "daily_24_normalized.parquet"
    print(f"   Guardando daily_24_normalized en: {normalized_path}")
    daily_normalized.to_parquet(normalized_path, index=False)

    print("✅ Proceso completado.")


if __name__ == "__main__":
    main()
