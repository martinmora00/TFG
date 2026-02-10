import torch
import pandas as pd
from pathlib import Path

# =========================
# CONFIGURACIÓN
# =========================

EMB_DIR = Path("data_processed/vit_embeddings")
INDEX = Path("data_processed/vit_index.csv")

WINDOW = 7  # tamaño de la ventana temporal

OUT_DIR = Path("data_processed/llama_dataset_v2")
OUT_DIR.mkdir(exist_ok=True)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# =========================
# UTILIDADES
# =========================

def load_index():
    df = pd.read_csv(INDEX)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["EQUIP_ID", "date"])
    return df


def load_embedding(path: Path):
    return torch.load(path, map_location="cpu").float()


def build_sequences_from_embeddings(embs):
    X, Y = [], []
    for i in range(len(embs) - WINDOW):
        X.append(torch.stack(embs[i:i + WINDOW]))
        Y.append(embs[i + WINDOW])
    return X, Y


# =========================
# PROCESO PRINCIPAL
# =========================

def main():
    print("Cargando índice de embeddings...")
    df = load_index()

    X_train, Y_train = [], []
    X_val, Y_val = [], []
    X_test, Y_test = [], []

    for equip_id, group in df.groupby("EQUIP_ID"):
        group = group.sort_values("date")

        paths = [
            EMB_DIR / Path(p).name
            for p in group["embedding"]
        ]
        embs = [load_embedding(p) for p in paths]

        n_days = len(embs)
        if n_days < WINDOW + 1:
            continue

        # Split temporal por días
        n_train = int(n_days * TRAIN_RATIO)
        n_val = int(n_days * VAL_RATIO)

        train_embs = embs[:n_train]
        val_embs = embs[n_train:n_train + n_val]
        test_embs = embs[n_train + n_val:]

        # Ventanas SOLO dentro de cada split
        Xt, Yt = build_sequences_from_embeddings(train_embs)
        Xv, Yv = build_sequences_from_embeddings(val_embs)
        Xs, Ys = build_sequences_from_embeddings(test_embs)

        X_train.extend(Xt)
        Y_train.extend(Yt)
        X_val.extend(Xv)
        Y_val.extend(Yv)
        X_test.extend(Xs)
        Y_test.extend(Ys)

    # Apilamos
    X_train = torch.stack(X_train)
    Y_train = torch.stack(Y_train)
    X_val = torch.stack(X_val)
    Y_val = torch.stack(Y_val)
    X_test = torch.stack(X_test)
    Y_test = torch.stack(Y_test)

    # Guardado con sufijo v2
    torch.save((X_train, Y_train), OUT_DIR / "train_v2.pt")
    torch.save((X_val, Y_val), OUT_DIR / "val_v2.pt")
    torch.save((X_test, Y_test), OUT_DIR / "test_v2.pt")

    print("✅ Dataset v2 generado correctamente")
    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)
    print("Ruta:", OUT_DIR)


if __name__ == "__main__":
    main()
