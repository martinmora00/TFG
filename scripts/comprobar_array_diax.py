import numpy as np
import pandas as pd

PARQUET_PATH = PARQUET_PATH = "../data_processed/daily_24_normalized.parquet"

equip_id = 63
day = "2022-12-23"

df = pd.read_parquet(PARQUET_PATH)

row = df[(df["EQUIP_ID"] == equip_id) & (df["date"].astype(str).str[:10] == day)]
assert len(row) == 1, f"Esperaba 1 fila y hay {len(row)}"

hour_cols = [f"h{i:02d}" for i in range(24)]
x = row.iloc[0][hour_cols].to_numpy(dtype=np.float32)

print("Vector x (h00..h23) usado para la GAF:")
print(x)
print("min/max:", float(x.min()), float(x.max()))
