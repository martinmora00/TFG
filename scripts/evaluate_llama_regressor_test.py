"""


Genera gráficas a partir de la evaluación del conjunto test:
- Histograma Cosine similarity
- Histograma MSE
- Curvas de percentiles (Cosine y MSE)
- CDF (distribución acumulada)


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoModel

# =========================
# CONFIGURACIÓN
# =========================


DATA_DIR = Path("../data_processed/llama_dataset")  # contiene test.pt
MODEL_CKPT = Path("../models/llama_regressor/llama_regressor.pt")

LLAMA_PATH = "meta-llama/Llama-3.2-1B-Instruct"
EMB_DIM = 768
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

# Carpeta donde guardar figuras
OUT_FIG_DIR = Path("../data_processed/figures_test")
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

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


def compute_metrics_on_test(model):
    print("Cargando test.pt...")
    X_test, Y_test = torch.load(DATA_DIR / "test.pt")  # (N,7,768) y (N,768)
    n = X_test.shape[0]
    print("X_test:", X_test.shape, "Y_test:", Y_test.shape)
    print(f"Evaluando {n} ejemplos...")

    mse_list = []
    cos_list = []

    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            xb = X_test[i:i+BATCH_SIZE].to(DEVICE)
            yb = Y_test[i:i+BATCH_SIZE].to(DEVICE)

            pred = model(xb)

            # MSE por ejemplo: media de las 768 dims
            mse_batch = F.mse_loss(pred, yb, reduction="none").mean(dim=1)
            mse_list.extend(mse_batch.detach().cpu().numpy())

            # Coseno por ejemplo
            cos_batch = F.cosine_similarity(pred, yb, dim=1)
            cos_list.extend(cos_batch.detach().cpu().numpy())

    mse_arr = np.array(mse_list, dtype=float)
    cos_arr = np.array(cos_list, dtype=float)

    return mse_arr, cos_arr


def save_histogram(values, title, xlabel, out_path, bins=40):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Guardada:", out_path)


def save_percentile_curve(values, title, ylabel, out_path):
    # Percentiles 0..100
    ps = np.arange(0, 101)
    vals = np.percentile(values, ps)

    plt.figure(figsize=(8, 5))
    plt.plot(ps, vals)
    plt.title(title)
    plt.xlabel("Percentil")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Guardada:", out_path)


def save_cdf(values, title, xlabel, out_path):
    # CDF: valores ordenados vs proporción acumulada
    v = np.sort(values)
    y = np.linspace(0, 1, len(v), endpoint=True)

    plt.figure(figsize=(8, 5))
    plt.plot(v, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Proporción acumulada")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Guardada:", out_path)


def main():
    model = load_model()
    mse_arr, cos_arr = compute_metrics_on_test(model)

    # Resumen numérico en consola (por si lo quieres copiar a la memoria)
    print("\n========= RESUMEN =========")
    print("N ejemplos:", len(mse_arr))
    print("MSE media:", float(mse_arr.mean()))
    print("Cosine media:", float(cos_arr.mean()))
    for p in [50, 75, 90]:
        print(f"MSE p{p}:", float(np.percentile(mse_arr, p)))
        print(f"Cos p{p}:", float(np.percentile(cos_arr, p)))
    print("==========================\n")

    # 1) Histogramas
    save_histogram(
        cos_arr,
        title="Distribución de similitud coseno en test",
        xlabel="Cosine similarity",
        out_path=OUT_FIG_DIR / "hist_cosine_test.png",
        bins=40
    )

    save_histogram(
        mse_arr,
        title="Distribución de MSE en test",
        xlabel="MSE",
        out_path=OUT_FIG_DIR / "hist_mse_test.png",
        bins=40
    )

    # 2) Curvas de percentiles
    save_percentile_curve(
        cos_arr,
        title="Curva de percentiles - similitud coseno (test)",
        ylabel="Cosine similarity",
        out_path=OUT_FIG_DIR / "percentiles_cosine_test.png"
    )

    save_percentile_curve(
        mse_arr,
        title="Curva de percentiles - MSE (test)",
        ylabel="MSE",
        out_path=OUT_FIG_DIR / "percentiles_mse_test.png"
    )

    # 3) CDF (opcional pero queda muy bien)
    save_cdf(
        cos_arr,
        title="CDF - similitud coseno (test)",
        xlabel="Cosine similarity",
        out_path=OUT_FIG_DIR / "cdf_cosine_test.png"
    )

    save_cdf(
        mse_arr,
        title="CDF - MSE (test)",
        xlabel="MSE",
        out_path=OUT_FIG_DIR / "cdf_mse_test.png"
    )

    print("Figuras guardadas en:", OUT_FIG_DIR.resolve())


if __name__ == "__main__":
    main()
