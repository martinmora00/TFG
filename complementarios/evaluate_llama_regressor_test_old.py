"""
evaluate_llama_regressor_test.py

Evalúa el modelo LLaMA entrenado sobre TODO el conjunto test.pt
y calcula métricas agregadas:
- MSE medio
- Cosine similarity media
- Percentiles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoModel

# =========================
# CONFIGURACIÓN
# =========================

DATA_DIR = Path("data_processed/llama_dataset")
MODEL_CKPT = Path("models/llama_regressor/llama_regressor.pt")

LLAMA_PATH = "meta-llama/Llama-3.2-1B-Instruct"

WINDOW = 7
EMB_DIM = 768
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4  # para no quedarnos sin VRAM


# =========================
# MODELO
# =========================

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

        return self.output_head(last_hidden)


# =========================
# CARGA MODELO
# =========================

def load_model():
    print("Cargando modelo base LLaMA...")
    base_model = AutoModel.from_pretrained(
        LLAMA_PATH,
        torch_dtype=torch.float16,
        device_map="cuda" if DEVICE == "cuda" else "cpu",
        low_cpu_mem_usage=True
    )

    hidden_size = base_model.config.hidden_size
    model = LlamaEmbeddingRegressor(base_model, hidden_size).to(DEVICE)

    print("Cargando checkpoint entrenado...")
    ckpt = torch.load(MODEL_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    return model


# =========================
# MAIN
# =========================

def main():
    print("Cargando dataset de test...")
    X_test, Y_test = torch.load(DATA_DIR / "test.pt")

    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)

    model = load_model()

    mse_list = []
    cos_list = []

    n = X_test.shape[0]

    print(f"Evaluando {n} ejemplos...")
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            xb = X_test[i:i+BATCH_SIZE].to(DEVICE)
            yb = Y_test[i:i+BATCH_SIZE].to(DEVICE)

            pred = model(xb)

            # MSE por ejemplo
            mse_batch = F.mse_loss(pred, yb, reduction="none").mean(dim=1)
            mse_list.extend(mse_batch.cpu().numpy())

            # Cosine similarity por ejemplo
            cos_batch = F.cosine_similarity(pred, yb, dim=1)
            cos_list.extend(cos_batch.cpu().numpy())

    mse_arr = np.array(mse_list)
    cos_arr = np.array(cos_list)

    print("\n========= RESULTADOS TEST =========")
    print(f"N ejemplos: {n}")
    print(f"MSE media: {mse_arr.mean():.6f}")
    print(f"Cosine similarity media: {cos_arr.mean():.6f}")

    print("\n--- Percentiles ---")
    for p in [50, 75, 90]:
        print(f"MSE p{p}: {np.percentile(mse_arr, p):.6f}")
        print(f"Cosine p{p}: {np.percentile(cos_arr, p):.6f}")

    print("===================================")


if __name__ == "__main__":
    main()
