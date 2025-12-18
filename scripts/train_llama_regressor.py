import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel

DATA_DIR = Path("data_processed/llama_dataset")
MODEL_DIR = Path("models/llama_regressor")

LLAMA_PATH = "meta-llama/Llama-3.2-1B-Instruct"   # usa la caché de HuggingFace

WINDOW = 7
EMB_DIM = 768
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
#  Modelo regresor: proyección de embeddings -> LLaMA base -> cabeza MLP
# ---------------------------------------------------------------------
class LlamaEmbeddingRegressor(nn.Module):
    def __init__(self, llama_model, hidden_size: int):
        super().__init__()
        self.llama = llama_model

        # Congelamos todos los parámetros del modelo base
        for p in self.llama.parameters():
            p.requires_grad = False

        # Capa que convierte embedding visual (768) a dimensión interna de LLaMA
        self.input_proj = nn.Linear(EMB_DIM, hidden_size)

        # Capa que convierte la salida de LLaMA en un embedding visual predicho
        self.output_head = nn.Linear(hidden_size, EMB_DIM)

    def forward(self, x):
        # x: (batch, W, 768)
        llama_device = next(self.llama.parameters()).device
        llama_dtype  = next(self.llama.parameters()).dtype

        # 1) Convertimos la entrada a float32 para la proyección
        x = x.to(device=llama_device, dtype=torch.float32)

        # 2) Proyectamos al espacio oculto del modelo LLaMA
        x_proj = self.input_proj(x)

        # 3) Convertimos la proyección al dtype del modelo (float16)
        x_proj = x_proj.to(dtype=llama_dtype)

        # 4) Pasamos por el modelo LLaMA base como inputs_embeds
        outputs = self.llama(inputs_embeds=x_proj)

        # 5) Obtenemos la representación del último token
        last_hidden = outputs.last_hidden_state[:, -1, :]

        # 6) Volvemos a float32 para mayor estabilidad numérica
        last_hidden = last_hidden.to(dtype=torch.float32)

        # 7) Pasamos por la cabeza que predice el embedding siguiente
        pred_emb = self.output_head(last_hidden)

        return pred_emb


# ---------------------------------------------------------------------
#  Cargar dataset
# ---------------------------------------------------------------------
def load_data():
    (X_train, Y_train) = torch.load(DATA_DIR / "train.pt")
    (X_val, Y_val)     = torch.load(DATA_DIR / "val.pt")
    return X_train, Y_train, X_val, Y_val


# ---------------------------------------------------------------------
#  Entrenamiento
# ---------------------------------------------------------------------
def main():
    print("Cargando dataset...")
    X_train, Y_train, X_val, Y_val = load_data()

    print("Cargando modelo LLaMA base desde:", LLAMA_PATH)
    base_model = AutoModel.from_pretrained(
        LLAMA_PATH,
        dtype=torch.float16,
        device_map="cuda" if DEVICE == "cuda" else "cpu",
        low_cpu_mem_usage=True
    )

    hidden_size = base_model.config.hidden_size
    print("hidden_size =", hidden_size)

    model = LlamaEmbeddingRegressor(base_model, hidden_size).to(DEVICE)

    # Solo se entrenan las capas nuevas
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    loss_fn = nn.MSELoss()

    batch_size = 4
    epochs = 3

    n_train = X_train.shape[0]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # Mezclamos el dataset para esta época
        perm = torch.randperm(n_train)
        X_train = X_train[perm]
        Y_train = Y_train[perm]

        for i in range(0, n_train, batch_size):
            xb = X_train[i:i+batch_size].to(DEVICE)
            yb = Y_train[i:i+batch_size].to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / n_train

        # Validación
        model.eval()
        with torch.no_grad():
            Xv = X_val.to(DEVICE)
            Yv = Y_val.to(DEVICE)
            pred_v = model(Xv)
            val_loss = loss_fn(pred_v, Yv).item()

        print(f"Epoch {epoch+1}/{epochs} - Loss train: {avg_train_loss:.6f} - Loss val: {val_loss:.6f}")

    # Guardar modelo
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_size": hidden_size,
        "window": WINDOW,
        "emb_dim": EMB_DIM
    }, MODEL_DIR / "llama_regressor.pt")

    print("Modelo guardado en:", MODEL_DIR / "llama_regressor.pt")


# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
