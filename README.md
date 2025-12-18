# TFG - Series temporales -> GAF -> ViT -> LLaMA (1B)

Este repositorio contiene los scripts del pipeline desarrollado en WSL2 para transformar series temporales diarias en representaciones visuales (GAF), extraer embeddings con ViT y entrenar un modelo autoregresivo basado en LLaMA 1B para predecir el embedding del día siguiente.

## Estructura
- scripts/: scripts del pipeline
- data_raw/: datos originales (NO incluidos)
- data_processed/: datos generados (NO incluidos)
- models/: pesos generados (NO incluidos)

## Requisitos
- Python 3.12
- Entorno Linux (recomendado WSL2 Ubuntu)

## Instalación
```bash
python -m venv llamaenv
source llamaenv/bin/activate
pip install -r requirements.txt
Pipeline (orden de ejecución)

Preprocesado (CSV -> días 24h normalizados):

python scripts/preprocess_timeseries.py


Generación de imágenes GAF:

python scripts/generate_gaf_images.py


Extracción de embeddings con ViT:

python scripts/extract_vit_embeddings.py


Construcción del dataset secuencial para LLaMA:

python scripts/build_llama_dataset.py


Entrenamiento del regresor con LLaMA 1B:

python scripts/train_llama_regressor.py


Evaluación de un ejemplo:

python scripts/evaluate_llama_regressor_example.py

Nota sobre datos y outputs
Por tamaño/confidencialidad, no se incluyen datasets ni artefactos generados (parquet, npy, pt, png).



