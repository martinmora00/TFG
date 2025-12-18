import torch
import pandas as pd
from pathlib import Path

EMB_DIR = Path("data_processed/vit_embeddings")
INDEX = Path("data_processed/vit_index.csv")

WINDOW = 7  # tamaño de la ventana de contexto
OUT_DIR = Path("data_processed/llama_dataset")
OUT_DIR.mkdir(exist_ok=True)

def load_index():
    df = pd.read_csv(INDEX)
    df = df.sort_values(["EQUIP_ID", "date"])
    return df

def load_embedding(path):
    return torch.load(path)

def build_sequences(df):
    X, Y = [], []

    for equip_id, group in df.groupby("EQUIP_ID"):
        paths = [EMB_DIR / p.split("/")[-1] for p in group["embedding"]]
        embs = [load_embedding(p) for p in paths]

        for i in range(len(embs) - WINDOW):
            seq = torch.stack(embs[i:i+WINDOW])        # (W, 768)
            nxt = embs[i+WINDOW]                       # (768)
            X.append(seq)
            Y.append(nxt)

    X = torch.stack(X)
    Y = torch.stack(Y)
    return X, Y

def temporal_split(X, Y):
    n = len(X)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val,   Y_val   = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test,  Y_test  = X[n_train+n_val:],        Y[n_train+n_val:]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

def main():
    print("Cargando índice de embeddings…")
    df = load_index()

    print("Generando secuencias...")
    X, Y = build_sequences(df)
    print("X:", X.shape, "Y:", Y.shape)

    print("Dividiendo en train/val/test…")
    (Xt, Yt), (Xv, Yv), (Xs, Ys) = temporal_split(X, Y)

    torch.save((Xt, Yt), OUT_DIR / "train.pt")
    torch.save((Xv, Yv), OUT_DIR / "val.pt")
    torch.save((Xs, Ys), OUT_DIR / "test.pt")

    print("Dataset generado en:", OUT_DIR)

if __name__ == "__main__":
    main()
