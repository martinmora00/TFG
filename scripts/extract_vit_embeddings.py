"""
extract_vit_embeddings.py

Extrae embeddings visuales a partir de las imágenes GAF (24×24)
usando un modelo Vision Transformer ligero.

- Entrada:
    data_processed/gaf_images/*.npy
    data_processed/gaf_index.csv

- Salida:
    data_processed/vit_embeddings/*.pt
    data_processed/vit_index.csv
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ============================
# CONFIGURACIÓN
# ============================

GAF_INDEX = Path("data_processed/gaf_index.csv")
GAF_DIR = Path("data_processed/gaf_images")
OUT_DIR = Path("data_processed/vit_embeddings")

# Modelo ViT muy ligero
MODEL_NAME = "google/vit-base-patch16-224-in21k"  # se puede cambiar por tiny o small
EMB_LAYER = "pooler_output"  # capa donde obtenemos el embedding


# ============================
# FUNCIONES AUXILIARES
# ============================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_vit_model():
    """
    Carga un Vision Transformer de HuggingFace.
    También carga los transformadores de entrada.
    """
    from transformers import ViTModel, ViTImageProcessor

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTModel.from_pretrained(MODEL_NAME)

    model.eval()
    model.cuda()  # enviar a GPU si existe

    return processor, model


def gaf_to_pil(gaf: np.ndarray) -> Image.Image:
    """
    Convierte una matriz 24×24 en una imagen PIL de 3 canales,
    escalada a 224×224 para ViT.
    """
    # Convertimos GAF 24x24 a imagen 3 canales duplicando
    gaf_img = np.stack([gaf, gaf, gaf], axis=-1)  # → 24×24×3
    gaf_img = (gaf_img - gaf_img.min()) / (gaf_img.max() - gaf_img.min() + 1e-6)
    gaf_img = (gaf_img * 255).astype(np.uint8)

    return Image.fromarray(gaf_img)


# ============================
# MÉTODO PRINCIPAL
# ============================

def main():
    ensure_dir(OUT_DIR)

    print("1) Cargando índice de GAF:", GAF_INDEX)
    df = pd.read_csv(GAF_INDEX)
    print(f"   Total de imágenes GAF: {len(df)}")

    print("2) Cargando modelo ViT:", MODEL_NAME)
    processor, model = load_vit_model()

    embeddings = []
    total = len(df)

    for i, row in df.iterrows():
        equip_id = row["EQUIP_ID"]
        date = row["date"]
        gaf_path = GAF_DIR.parent / row["gaf_npy"]

        # Cargar matriz GAF
        gaf = np.load(gaf_path)

        # Convertir a imagen PIL
        image = gaf_to_pil(gaf)

        # Preprocesar para ViT
        inputs = processor(images=image, return_tensors="pt").to("cuda")

        # Inferencia (sin gradientes)
        with torch.no_grad():
            output = model(**inputs)
            emb = output.pooler_output.squeeze().cpu()  # vector 768

        # Guardar embedding
        fname = f"{equip_id}_{date.replace('-', '')}.pt"
        out_path = OUT_DIR / fname
        torch.save(emb, out_path)

        embeddings.append({
            "EQUIP_ID": equip_id,
            "date": date,
            "embedding": str(out_path.relative_to(OUT_DIR.parent))
        })

        if i % 500 == 0:
            print(f"   Procesados {i}/{total} embeddings...")

    # Guardar índice
    index_df = pd.DataFrame(embeddings)
    index_df.to_csv(OUT_DIR.parent / "vit_index.csv", index=False)

    print("3) Proceso completado.")
    print("   Embeddings guardados en:", OUT_DIR)
    print("   Índice guardado en: data_processed/vit_index.csv")


if __name__ == "__main__":
    main()
