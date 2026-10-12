# Proyecto: Machine Learning y Deep Learning Aplicados

**Universidad:** Universidad de Ingeniería y Tecnología (UTEC)  
**Profesor:** Marcelo Jafett Contreras Cabrera (`mcontreras@utec.edu.pe`)  
**Entorno de Ejecución:** Ejecutado en local con aceleración por GPU (CUDA)

---

## 1. Resumen Ejecutivo del Proyecto

Este repositorio contiene la implementación completa, modular y reproducible del proyecto final del curso, abarcando dos tareas fundamentales de clasificación multiclase:

1. **Tarea 1: Food-101 (Visión Computacional)**
   * **Problema:** Clasificación de 101 clases de alimentos (101,000 imágenes).
   * **Método 1 (Baseline):** ResNet-18 con cabezal denso y Global Average Pooling.
   * **Método 2 (Avanzado):** EfficientNet-B0 con escalamiento compuesto, Label Smoothing y optimizador AdamW con planificador de coseno.
   * **Interpretabilidad:** t-SNE de representaciones del espacio latente y mapas de activación visual con Grad-CAM.

2. **Tarea 2: AG News (Procesamiento de Lenguaje Natural)**
   * **Problema:** Clasificación de 4 categorías de noticias (World, Sports, Business, Sci/Tech) con 127,600 textos.
   * **Método 1 (Clásico):** Extracción de características n-gramas TF-IDF con Support Vector Machine Lineal (LinearSVC).
   * **Método 2 (Deep Learning):** Fine-Tuning de Transformer pre-entrenado (DistilBERT-base-uncased) con Automatic Mixed Precision (AMP FP16).
   * **Interpretabilidad:** Coeficientes de pesos TF-IDF por categoría y proyección t-SNE del token `[CLS]`.

---

## 2. Estructura del Repositorio

```text
MLDP/
├── Deep_learning_project.pdf          # Especificaciones oficiales del proyecto
├── README.md                          # Guía de reproducción y documentación técnica
├── .gitignore                         # Exclusiones de Git (datasets, checkpoints, caches)
├── .vscode/settings.json              # Configuración de exclusión para file watchers del IDE
│
├── Cod/
│   ├── notebooks/                     # Notebooks Jupyter interactivos listos para ejecutar
│   │   ├── 01_food101_classification.ipynb
│   │   └── 02_agnews_classification.ipynb
│   ├── src/                           # Módulos Python reutilizables
│   │   ├── __init__.py
│   │   ├── config.py                  # Configuración de GPU local (TF32, cuDNN, semillas)
│   │   ├── metrics.py                 # Accuracy, Precision, Recall, F1 (Macro/Weighted), Conf. Matrix
│   │   ├── visualize.py               # Curvas de pérdida/acc, t-SNE, PCA, UMAP
│   │   ├── trainer.py                 # Loop de entrenamiento PyTorch con AMP FP16, GradScaler y Checkpointing
│   │   ├── data_nlp.py                # Dataset y DataLoader para AG News (splits estratificados)
│   │   ├── models_nlp.py              # TF-IDF + LinearSVC y TransformerClassifier
│   │   ├── data_vision.py             # Food-101 Dataset, RandAugment y DataLoaders
│   │   └── models_vision.py           # ResNet-18, EfficientNet-B0 y Grad-CAM
│   └── checkpoints/                   # Directorio de pesos entrenados (*.pt)
│
├── data/                              # Datasets descargados y listos
│   ├── ag_news/                       # train.csv, test.csv y formato Arrow
│   └── food101/                       # Directorio extraído 'food-101' con 101,000 imágenes
│
├── Report/                            # Reporte formal en formato IEEE Conference
│   ├── main.tex                       # Código fuente en LaTeX con todas las secciones y figuras
│   ├── main.pdf                       # PDF final compilado con Tectonic (7 páginas)
│   ├── figures/                       # Figuras reales extraídas de los notebooks (12 gráficos)
│   └── IEEEtran.cls                   # Clase oficial IEEE Conference
│
└── scripts/                           # Scripts utilitarios de descarga y automatización
    ├── download_food101.py            # Descargador con reanudación por rangos HTTP
    └── build_notebooks.py             # Generador de notebooks Jupyter
```

---

## 3. Requisitos del Sistema y Entorno

* **Python:** 3.14.7 (en entorno virtual `.venv`)
* **PyTorch:** `2.14.0+cu130` con soporte CUDA acelerado
* **Torchvision:** `0.29.0+cu130`
* **Transformers:** `5.19.0`
* **Scikit-Learn:** `1.9.1`

Las optimizaciones específicas habilitadas en el entorno local con GPU incluyen:
* **TensorFloat-32 (TF32):** `torch.backends.cuda.matmul.allow_tf32 = True`
* **cuDNN Benchmark:** `torch.backends.cudnn.benchmark = True`
* **Automatic Mixed Precision (AMP):** FP16 con `torch.amp.GradScaler('cuda')`
* **DataLoaders optimizados:** `pin_memory=True` y hasta 8 subprocesos de lectura asíncrona.

---

## 4. Instrucciones para Ejecutar los Notebooks

Abre Antigravity / VS Code o inicia Jupyter Lab:

1. **Para la Tarea 1 (Food-101):**
   * Abre `Cod/notebooks/01_food101_classification.ipynb`.
   * Selecciona el kernel de Python correspondiente a `.venv` (`Python 3 (.venv)`).
   * Ejecuta las celdas secuencialmente (`Run All` o celda por celda).
   * *Tip:* La variable `SUBSET_FRACTION` en la celda 2 está en `1.0` (dataset completo). Puedes ponerla en `0.1` si deseas hacer una prueba preliminar ultra rápida en 2 minutos.

2. **Para la Tarea 2 (AG News):**
   * Abre `Cod/notebooks/02_agnews_classification.ipynb`.
   * Selecciona el kernel `.venv`.
   * Ejecuta las celdas para comparar TF-IDF + LinearSVC contra DistilBERT, graficar matrices de confusión, t-SNE de embeddings y realizar la ablación de hiperparámetros.

---

## 5. Compilación del Reporte IEEE

El informe formal en formato IEEE Conference se encuentra en `Report/main.tex`. Para recompilarlo:

```bash
tectonic Report/main.tex -o Report/
```

El archivo generado será `Report/main.pdf`.
