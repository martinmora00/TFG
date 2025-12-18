import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoModel

DATA_DIR = Path("data_processed")
EMB_DIR = DATA_DIR / "vit_embeddings"
INDEX_PATH = DATA_DIR / "vit_index.csv"
MODEL_CKPT = Path("models/llama_regressor/llama_regressor.pt")

LLAMA_PATH = "meta-llama/Llama-3.2-1B-Instruct"
WINDOW = 7
EMB_DIM = 768
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    # Usamos la última ventana posible de 7 días + día objetivo
    if len(df_e) <= WINDOW:
        raise ValueError("No hay suficientes días para construir una ventana.")

    # Índice del primer día de la ventana
    start_idx = len(df_e) - (WINDOW + 1)
    end_idx = start_idx + WINDOW  # último día de la ventana
    target_idx = end_idx          # día objetivo t+1

    window_rows = df_e.iloc[start_idx:end_idx]
    target_row = df_e.iloc[target_idx]

    emb_list = []
    for emb_rel_path in window_rows["embedding"]:
        emb_path = DATA_DIR / emb_rel_path  # p.ej. vit_embeddings/63_20221124.pt
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
    equip_id = 63  
    print(f"Evaluando para EQUIP_ID = {equip_id}")

    df_e = load_embeddings_for_equip(equip_id)
    print(f"Días disponibles para {equip_id}: {len(df_e)}")

    X, y_true, info = build_example_sequence(df_e)

    model = load_model()

    with torch.no_grad():
        X_dev = X.to(DEVICE)
        y_pred = model(X_dev)        # (1, 768)

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

    # Guardar gráfica comparando las dos curvas de embedding
    plt.figure(figsize=(10,4))
    plt.plot(y_true_cpu.numpy(), label="Real", alpha=0.8)
    plt.plot(y_pred_cpu.numpy(), label="Predicho", alpha=0.8)
    plt.title(f"Embedding real vs predicho - EQUIP_ID {info['EQUIP_ID']} - {info['target_date']}")
    plt.xlabel("Dimensión del embedding")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()

    out_path = DATA_DIR / f"example_embedding_pred_vs_true_{info['EQUIP_ID']}_{info['target_date'].replace('-','')}.png"
    plt.savefig(out_path)
    print("Gráfica guardada en:", out_path)


if __name__ == "__main__":
    main()
