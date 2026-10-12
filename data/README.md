# Datasets del Proyecto

Este directorio almacena los conjuntos de datos del proyecto:

1. **Food-101 (`data/food101/food-101/`)**
   * 101,000 imágenes gastronómicas distribuidas en 101 categorías.
   * Se descarga y extrae ejecutando:
     ```bash
     python scripts/download_food101.py
     ```

2. **AG News (`data/ag_news/`)**
   * 120,000 textos de entrenamiento y 7,600 de prueba en 4 clases temáticas.
   * Almacenado en formato CSV y Arrow.

> **Nota:** La carpeta `data/` y sus archivos pesados están excluidos de Git mediante `.gitignore` para mantener el repositorio ligero y eficiente.
