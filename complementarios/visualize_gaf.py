import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

GAF_DIR = Path("data_processed/gaf_images")

def visualize_gaf(equip_id, date_str):
    filename = f"{equip_id}_{date_str}.npy"
    gaf_path = GAF_DIR / filename

    print("Cargando:", gaf_path)

    gaf = np.load(gaf_path)

    plt.figure(figsize=(4,4))
    plt.imshow(gaf, cmap="viridis")
    plt.title(f"GAF {equip_id} - {date_str}")
    plt.axis("off")

    out_file = Path("data_processed") / f"GAF_{equip_id}_{date_str}.png"
    plt.savefig(out_file, bbox_inches='tight')
    print("Imagen guardada en:", out_file)

if __name__ == "__main__":
    # Ejemplo concreto:
    visualize_gaf(63, "20230315")
