import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModel

DATA_DIR = Path("data_processed")
EMB_DIR = DATA_DIR / "vit_embeddings"
INDEX_PATH = DATA_DIR / "vit_index.csv"
MODEL_CKPT = Path("models/llama_regressor/llama_regressor.pt")

LLAMA_PATH = "meta-llama/Llama-3.2-1B-Instruct"
WINDOW = 7
EMB_DIM = 768
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EQUIP_ID = 63 


class LlamaEmbeddingRegressor(nn.Module):
    def __init__(self, llama_model, hidden_size: int):
        super().__init__()
        self.llama = llama_model

        for p in self.llama.parameters():
            p.requires_grad = False

        self.input_proj = nn.Linear(EMB_DIM, hidden_size)
        self.output_head = nn.Linear(hidden_size, EMB_DIM)

    def forward(self, x):
        llama_device = next(self.llama.parameters()).device
        llama_dtype  = next(self.llama.parameters()).dtype

        x = x.to(device=llama_device, dtype=torch.float32)
        x_proj = self.input_proj(x)
        x_proj = x_proj.to(dtype=llama_dtype)

        outputs = self.llama(inputs_embeds=x_proj)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        last_hidden = last_hidden.to(dtype=torch.float32)

        pred_emb = self.output_head(last_hidden)
        return pred_emb


def load_model():
    print("Cargando modelo base LLaMA...")
    base_model = AutoModel.from_pretrained(
        LLAMA_PATH,
        dtype=torch.float16,
        device_map="cuda" if DEVICE == "cuda" else "cpu",
        low_cpu_mem_usage=True
    )
    hidden_size = base_model.config.hidden_size
    model = LlamaEmbeddingRegressor(base_model, hidden_size).to(DEVICE)

    print("Cargando checkpoint de regresor...")
    ckpt = torch.load(MODEL_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    return model


def load_embeddings_for_equip(equip_id: int):
    df = pd.read_csv(INDEX_PATH)
    df_e = df[df["EQUIP_ID"] == equip_id].sort_values("date").reset_index(drop=True)
    return df_e


def build_example_sequence(df_e):
    if len(df_e) <= WINDOW:
        raise ValueError("No hay suficientes días para construir una ventana.")

    start_idx = len(df_e) - (WINDOW + 1)
    end_idx = start_idx + WINDOW
    target_idx = end_idx

    window_rows = df_e.iloc[start_idx:end_idx]
    target_row = df_e.iloc[target_idx]

    emb_list = []
    for emb_rel_path in window_rows["embedding"]:
        emb_path = DATA_DIR / emb_rel_path
        emb = torch.load(emb_path)
        emb_list.append(emb)

    X = torch.stack(emb_list).unsqueeze(0)  # (1, W, 768)

    y_true_path = DATA_DIR / target_row["embedding"]
    y_true = torch.load(y_true_path).unsqueeze(0)  # (1, 768)

    info = {
        "EQUIP_ID": int(target_row["EQUIP_ID"]),
        "target_date": target_row["date"]
    }

    return X, y_true, info


def main():
    df_e = load_embeddings_for_equip(EQUIP_ID)
    X, y_true, info = build_example_sequence(df_e)

    model = load_model()

    with torch.no_grad():
        X_dev = X.to(DEVICE)
        y_pred = model(X_dev)

    y_true_cpu = y_true.squeeze(0).cpu()
    y_pred_cpu = y_pred.squeeze(0).cpu()

    mse = F.mse_loss(y_pred_cpu, y_true_cpu).item()
    cos_sim = F.cosine_similarity(y_pred_cpu, y_true_cpu, dim=0).item()

    print("======================================")
    print(f"EQUIP_ID: {info['EQUIP_ID']}")
    print(f"Fecha objetivo real: {info['target_date']}")
    print(f"MSE embedding: {mse:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print("======================================")

    # Vector de error por dimensión
    err = (y_pred_cpu - y_true_cpu).numpy()
    err_sq = err ** 2  # error cuadrático por dimensión

    # Reorganizamos en una matriz 24x32 para visualizar como mapa de calor
    err_2d = err_sq.reshape(24, 32)

    plt.figure(figsize=(6, 4))
    im = plt.imshow(err_2d, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Error cuadrático por dimensión")
    plt.title(f"Mapa de calor del error de embedding\nEQUIP_ID {info['EQUIP_ID']} - {info['target_date']}")
    plt.xlabel("Bloque de dimensiones")
    plt.ylabel("Fila de la rejilla (reordenado)")
    plt.tight_layout()

    out_path = DATA_DIR / f"embedding_error_heatmap_{info['EQUIP_ID']}_{info['target_date'].replace('-','')}.png"
    plt.savefig(out_path, dpi=200)
    print("Heatmap guardado en:", out_path)


if __name__ == "__main__":
    main()
