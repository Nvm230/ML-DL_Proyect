import os
import json

NOTEBOOKS_DIR = "/home/nvm/Documents/UTEC/MLDP/Cod/notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.14.7"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

# ==========================================
# 1. FOOD-101 NOTEBOOK
# ==========================================
food_cells = [
    md_cell("""# Proyecto: Machine Learning y Deep Learning Aplicados (UTEC)
## Tarea 1: Clasificación de Imágenes en Food-101 (101 Clases)
**Profesor:** Marcelo Jafett Contreras Cabrera  
**Hardware Optimizado:** NVIDIA GeForce RTX 5070 (12GB VRAM) + PyTorch 2.14 AMP (FP16/TF32)

---
### 1. Definición del Problema Matemático
Se busca aprender un mapeo paramétrico $f_\\theta: \\mathcal{X} \\to \\mathcal{Y}$, donde $\\mathcal{X} \\subset \\mathbb{R}^{H \\times W \\times 3}$ es el espacio de imágenes de alimentos y $\\mathcal{Y} = \\{0, 1, \\dots, 100\\}$ representa las 101 clases gastronómicas.
El objetivo es encontrar los pesos óptimos $\\theta^*$ que minimizan la función de pérdida de entropía cruzada categórica (Categorical Cross-Entropy):
$$\\mathcal{L}_{CE}(\\theta) = -\\frac{1}{N} \\sum_{i=1}^N \\sum_{c=1}^{C} y_{i,c} \\log(\\hat{y}_{i,c})$$
donde $\\hat{y}_{i,c} = \\frac{e^{z_{i,c}}}{\\sum_{j=1}^C e^{z_{i,j}}}$ es la distribución softmax estimada."""),

    code_cell("""# 1. Configuración de Hardware y Reproducibilidad
import sys
import os
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Agregar Cod/src al path
sys.path.append(os.path.abspath("../src"))

from config import setup_hardware, set_seed, FOOD101_DIR, CHECKPOINTS_DIR
from metrics import compute_classification_metrics, plot_confusion_matrix, print_metrics_summary
from visualize import plot_convergence_curves, plot_latent_space, extract_torch_embeddings
from trainer import Trainer
from data_vision import create_food101_dataloaders
from models_vision import FoodVisionBaseline, FoodVisionAdvanced, SimpleGradCAM

# Inicializar optimizaciones de GPU (TF32, cuDNN Benchmark) y fijar semilla aleatoria
hw_info = setup_hardware()
set_seed(42)"""),

    md_cell("""---
### 2. Preprocesamiento de Datos y Data Augmentation (Sección 5.1)
El preprocesamiento consiste en:
1. **Redimensionamiento y Recorte:** Redimensionamiento y recorte aleatorio a $224 \\times 224$ píxeles (`RandomResizedCrop`).
2. **Data Augmentation:** `RandAugment` con transformaciones aleatorias (rotaciones, cambios de brillo, contraste y traslaciones) y volteo horizontal (`RandomHorizontalFlip`).
3. **Normalización:** Estandarización de canales utilizando las medias y desviaciones de ImageNet:
   $$\\mu = [0.485, 0.456, 0.406], \\quad \\sigma = [0.229, 0.224, 0.225]$$
4. **DataLoaders Eficientes:** `num_workers=4`, `pin_memory=True` para transferencias directas DMA a la memoria VRAM de la GPU."""),

    code_cell("""# 2. Carga del Dataset Food-101 y DataLoaders
# NOTA: subset_fraction=1.0 usa todo el dataset (101,000 imágenes).
# Puedes usar subset_fraction=0.1 para pruebas y depuración ultra rápidas.
SUBSET_FRACTION = 1.0  
BATCH_SIZE = 64

train_loader, val_loader, test_loader, class_names = create_food101_dataloaders(
    root_dir=FOOD101_DIR,
    batch_size=BATCH_SIZE,
    img_size=224,
    num_workers=hw_info["recommended_workers"],
    subset_fraction=SUBSET_FRACTION
)

print(f"Número total de clases: {len(class_names)}")
print(f"Batches de entrenamiento: {len(train_loader)} | Validación: {len(val_loader)} | Prueba: {len(test_loader)}")"""),

    code_cell("""# Visualización de un batch de imágenes con sus etiquetas
images, labels = next(iter(train_loader))

plt.figure(figsize=(14, 7))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    # Desnormalizar para visualización
    img = images[i].permute(1, 2, 0).numpy()
    img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    plt.imshow(img)
    plt.title(class_names[labels[i].item()], fontsize=11, fontweight='bold')
    plt.axis("off")
plt.tight_layout()
plt.suptitle("Muestras del Conjunto de Entrenamiento (Food-101 con Augmentation)", y=1.02, fontsize=14, fontweight='bold')
plt.show()"""),

    md_cell("""---
### 3. Metodología: Comparación de Dos Algoritmos (Sección 4)

#### **Método 1: Baseline Ligero (ResNet-18)**
* **Espacio de Hipótesis:** Red residual profunda con conexiones de salto $y = \\mathcal{F}(x, \\{W_i\\}) + x$, agrupamiento global (Global Average Pooling) y clasificador lineal con dropout ($p=0.2$).
* **Optimizador:** Adam ($lr = 10^{-3}$) o SGD con momentum ($0.9$), weight decay $10^{-4}$.
* **Supuestos:** Las representaciones jerárquicas aprendidas en ImageNet son transferibles a la clasificación de alimentos sin desvanecimiento de gradiente.

#### **Método 2: Deep Learning Avanzado (EfficientNet-B0)**
* **Espacio de Hipótesis:** Bloques MBConv (Mobile Inverted Bottleneck Convolution) optimizados mediante búsqueda de arquitectura neuronal (NAS) y escalamiento compuesto uniforme de profundidad, ancho y resolución.
* **Optimizador:** AdamW con regularización por decaimiento de pesos desacoplado ($lr = 5 \\times 10^{-4}$) y Cosine Annealing Learning Rate Scheduler.
* **Regularización Adicional:** Dropout ($p=0.3$) y Label Smoothing ($\\alpha = 0.1$)."""),

    code_cell("""# 3. Método 1: Entrenamiento del Baseline ResNet-18
model_baseline = FoodVisionBaseline(num_classes=101, pretrained=True, freeze_backbone=False)
optimizer_baseline = torch.optim.Adam(model_baseline.parameters(), lr=1e-3, weight_decay=1e-4)
criterion_baseline = torch.nn.CrossEntropyLoss()

trainer_baseline = Trainer(
    model=model_baseline,
    optimizer=optimizer_baseline,
    criterion=criterion_baseline,
    device=hw_info["device"],
    use_amp=hw_info["use_amp"],
    checkpoint_dir=CHECKPOINTS_DIR,
    model_name="food101_resnet18_baseline"
)

# Entrenar modelo
EPOCHS = 10
history_baseline = trainer_baseline.fit(train_loader, val_loader, epochs=EPOCHS, early_stopping_patience=3)"""),

    code_cell("""# 4. Método 2: Entrenamiento del Modelo Avanzado EfficientNet-B0
model_advanced = FoodVisionAdvanced(num_classes=101, pretrained=True, dropout=0.3)
optimizer_advanced = torch.optim.AdamW(model_advanced.parameters(), lr=5e-4, weight_decay=1e-2)
scheduler_advanced = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_advanced, T_max=EPOCHS, eta_min=1e-6)
# Cross-Entropy con Label Smoothing (Sección 3.1)
criterion_advanced = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

trainer_advanced = Trainer(
    model=model_advanced,
    optimizer=optimizer_advanced,
    criterion=criterion_advanced,
    scheduler=scheduler_advanced,
    device=hw_info["device"],
    use_amp=hw_info["use_amp"],
    checkpoint_dir=CHECKPOINTS_DIR,
    model_name="food101_efficientnet_advanced"
)

# Entrenar modelo avanzado
history_advanced = trainer_advanced.fit(train_loader, val_loader, epochs=EPOCHS, early_stopping_patience=3)"""),

    md_cell("""---
### 4. Análisis de Convergencia (Sección 5.4)
Se presentan las curvas de pérdida y exactitud a lo largo de las épocas para entrenamiento y validación.
Se analiza:
* Presencia de **overfitting** (brecha entre curvas de train y val).
* Estabilidad del entrenamiento con optimizadores Adam vs AdamW + Cosine Scheduler.
* Beneficio del escalado de gradientes y precisión mixta (AMP)."""),

    code_cell("""# Curvas de convergencia para ambos métodos
fig1, _ = plot_convergence_curves(history_baseline, title="Convergencia Método 1: ResNet-18 Baseline")
plt.show()

fig2, _ = plot_convergence_curves(history_advanced, title="Convergencia Método 2: EfficientNet-B0 con Label Smoothing")
plt.show()"""),

    md_cell("""---
### 5. Métricas de Desempeño y Comparación en Test (Sección 5.3)
Evaluación formal sobre el conjunto de prueba independiente ($25,250$ imágenes de prueba).
Métricas calculadas:
* **Top-1 Accuracy**
* **Top-5 Accuracy**
* **Precision Macro & Weighted**
* **Recall Macro & Weighted**
* **F1-Score Macro & Weighted**"""),

    code_cell("""# Inferencia y cálculo de métricas en conjunto de test
preds_m1, probs_m1, targets_m1 = trainer_baseline.predict(test_loader)
metrics_m1 = compute_classification_metrics(targets_m1, preds_m1, y_probs=probs_m1, class_names=class_names)

preds_m2, probs_m2, targets_m2 = trainer_advanced.predict(test_loader)
metrics_m2 = compute_classification_metrics(targets_m2, preds_m2, y_probs=probs_m2, class_names=class_names)

# Tabla comparativa formal (para el reporte IEEE)
comparison_df = pd.DataFrame({
    "Métrica": ["Top-1 Accuracy (%)", "Top-5 Accuracy (%)", "Precision Macro (%)", "Recall Macro (%)", "F1-Score Macro (%)"],
    "Método 1 (ResNet-18)": [
        metrics_m1["accuracy"] * 100,
        metrics_m1.get("top5_accuracy", 0) * 100,
        metrics_m1["precision_macro"] * 100,
        metrics_m1["recall_macro"] * 100,
        metrics_m1["f1_macro"] * 100,
    ],
    "Método 2 (EfficientNet-B0)": [
        metrics_m2["accuracy"] * 100,
        metrics_m2.get("top5_accuracy", 0) * 100,
        metrics_m2["precision_macro"] * 100,
        metrics_m2["recall_macro"] * 100,
        metrics_m2["f1_macro"] * 100,
    ]
})
print("="*65)
print(" TABLA COMPARATIVA DE DESEMPEÑO - CONJUNTO DE PRUEBA")
print("="*65)
print(comparison_df.to_string(index=False))"""),

    code_cell("""# Matriz de Confusión Normalizada
fig, ax = plot_confusion_matrix(targets_m2, preds_m2, class_names=class_names, title="Matriz de Confusión - Método 2 (EfficientNet-B0)")
plt.show()"""),

    md_cell("""---
### 6. Interpretabilidad y Análisis del Espacio Latente (Sección 5.5)
1. **Reducción de Dimensionalidad (t-SNE / PCA):** Se proyectan las representaciones vectoriales del espacio latente (penúltima capa) a 2D para evaluar separabilidad de clases y posibles solapamientos gastronómicos.
2. **Grad-CAM (Gradient-weighted Class Activation Mapping):** Mapas de calor visuales que indican qué regiones de los alimentos impulsan la predicción del modelo."""),

    code_cell("""# Proyección t-SNE de las características del espacio latente
feats_m2, labels_m2 = extract_torch_embeddings(model_advanced, test_loader, hw_info["device"], max_batches=15)

plot_latent_space(
    features=feats_m2,
    labels=labels_m2,
    class_names=class_names,
    method="tsne",
    n_samples=600,
    title="Interpretabilidad: Proyección t-SNE del Espacio Latente (EfficientNet-B0)"
)
plt.show()"""),

    code_cell("""# Interpretabilidad Visual con Grad-CAM
grad_cam = SimpleGradCAM(model_advanced, model_advanced.backbone.features[-1])

# Seleccionar un ejemplo de test
sample_img, sample_lbl = next(iter(test_loader))
sample_img_gpu = sample_img[0:1].to(hw_info["device"])
heatmap = grad_cam.generate_heatmap(sample_img_gpu, target_class=sample_lbl[0].item())

# Visualizar original vs mapa de activación
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
raw_img = sample_img[0].permute(1, 2, 0).numpy()
raw_img = np.clip(raw_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
plt.imshow(raw_img)
plt.title(f"Original: {class_names[sample_lbl[0].item()]}", fontweight='bold')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(raw_img)
plt.imshow(heatmap, cmap="jet", alpha=0.5)
plt.title("Grad-CAM: Regiones de Interés", fontweight='bold')
plt.axis("off")
plt.tight_layout()
plt.show()"""),

    md_cell("""---
### 7. Estudio de Ablación (Sección 6.1)
Se evalúa el impacto de la técnica de regularización:
* **Configuración A:** Transfer Learning con Fine-Tuning completo (todas las capas entrenables).
* **Configuración B:** Feature Extraction puro (Backbone congelado, solo cabezal entrenable).
Este estudio permite cuantificar experimentalmente cuánto valor aporta la adaptación de capas profundas frente a pesos fijos de ImageNet."""),

    code_cell("""# Estudio de Ablación: Backbone Congelado vs Fine-Tuning Completo
model_frozen = FoodVisionBaseline(num_classes=101, pretrained=True, freeze_backbone=True)
opt_frozen = torch.optim.Adam(filter(lambda p: p.requires_grad, model_frozen.parameters()), lr=1e-3)
trainer_frozen = Trainer(model_frozen, opt_frozen, device=hw_info["device"], use_amp=hw_info["use_amp"], checkpoint_dir=CHECKPOINTS_DIR, model_name="ablation_frozen")

hist_frozen = trainer_frozen.fit(train_loader, val_loader, epochs=3)

print("\n[Ablación] Desempeño con Backbone Congelado:")
print(f"Mejor Val Acc: {max(hist_frozen['val_acc'])*100:.2f}%")
print(f"Mejor Val Acc con Fine-Tuning Completo: {max(history_baseline['val_acc'])*100:.2f}%")""")
]

# ==========================================
# 2. AG NEWS NOTEBOOK
# ==========================================
nlp_cells = [
    md_cell("""# Proyecto: Machine Learning y Deep Learning Aplicados (UTEC)
## Tarea 2: Clasificación de Textos en AG News (4 Clases)
**Profesor:** Marcelo Jafett Contreras Cabrera  
**Hardware Optimizado:** NVIDIA GeForce RTX 5070 (12GB VRAM) + PyTorch 2.14 AMP (FP16/TF32)

---
### 1. Definición del Problema Matemático
Se formaliza el problema de clasificación de textos de noticias en $C=4$ categorías:
1. **World** (Mundo)
2. **Sports** (Deportes)
3. **Business** (Negocios)
4. **Sci/Tech** (Ciencia y Tecnología)

Dado un texto no estructurado $x = (w_1, w_2, \\dots, w_T)$, se busca una hipótesis $f_\\theta(x) \\in \\{0, 1, 2, 3\\}$.
La optimización minimiza la pérdida de entropía cruzada multiclase:
$$\\mathcal{L}(\\theta) = -\\frac{1}{N} \\sum_{i=1}^N \\sum_{c=1}^4 y_{i,c} \\log(\\hat{y}_{i,c})$$"""),

    code_cell("""# 1. Configuración de Hardware y Reproducibilidad
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Agregar Cod/src al path
sys.path.append(os.path.abspath("../src"))

from config import setup_hardware, set_seed, AGNEWS_DIR, CHECKPOINTS_DIR
from metrics import compute_classification_metrics, plot_confusion_matrix, print_metrics_summary
from visualize import plot_convergence_curves, plot_latent_space, extract_torch_embeddings
from trainer import Trainer
from data_nlp import load_agnews_data, create_nlp_dataloaders, AG_NEWS_CLASSES
from models_nlp import ClassicalNLPBaseline, TransformerClassifier, get_nlp_tokenizer

hw_info = setup_hardware()
set_seed(42)"""),

    md_cell("""---
### 2. Exploración y Preprocesamiento de Datos (Sección 5.1)
El dataset AG News cuenta con $120,000$ ejemplos de entrenamiento balanceados ($30,000$ por clase) y $7,600$ ejemplos de prueba.
Se genera una partición estratificada de validación ($10\\%$ de entrenamiento = $12,000$ textos) garantizando la misma distribución de clases."""),

    code_cell("""# 2. Carga y Partición Estratificada de Datos
train_df, val_df, test_df = load_agnews_data(
    data_dir=AGNEWS_DIR,
    val_split=0.1,
    sample_fraction=1.0, # 1.0 = dataset completo
    random_state=42
)

# Análisis de balance de clases y longitud de los textos
train_df["text_len"] = train_df["text"].apply(lambda t: len(t.split()))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.countplot(data=train_df, x="label", palette="viridis")
plt.xticks(ticks=range(4), labels=AG_NEWS_CLASSES)
plt.title("Distribución de Clases en Train", fontweight='bold')

plt.subplot(1, 2, 2)
sns.histplot(train_df["text_len"], bins=40, kde=True, color="purple")
plt.title("Distribución de Longitud de Palabras", fontweight='bold')
plt.xlabel("Palabras por Noticia")
plt.tight_layout()
plt.show()"""),

    md_cell("""---
### 3. Metodología: Comparación de Dos Algoritmos (Sección 4)

#### **Método 1: Machine Learning Clásico (TF-IDF + LinearSVC)**
* **Espacio de Hipótesis:** Hiperplanos separadores lineales sobre representaciones ponderadas n-gramas sublineales (unigramas y bigramas, vocabulario $V = 25,000$):
  $$f(x) = \\arg\\max_{c \\in \\{1..4\\}} (w_c^T \\phi_{\\text{tfidf}}(x) + b_c)$$
* **Objetivo de Optimización:** Minimización de la pérdida Hinge cuadrática con penalización regularizadora $L_2$:
  $$\\min_{w, b} \\frac{1}{2} \\|w\\|^2 + C \\sum_{i=1}^N \\max(0, 1 - y_i (w^T \\phi(x_i) + b))^2$$
* **Ventajas:** Inferencia ultra veloz ($<1$ ms por texto), bajo costo computacional y explicabilidad directa mediante los coeficientes de n-gramas.

#### **Método 2: Deep Learning con Transfer Learning (DistilBERT)**
* **Espacio de Hipótesis:** Red neuronal de transformadores con auto-atención bidireccional de 6 capas y 66M parámetros pre-entrenada con Masked Language Modeling:
  $$h_{[CLS]} = \\text{DistilBERT}(x), \\quad \\hat{y} = \\text{Softmax}(W h_{[CLS]} + b)$$
* **Optimizador:** AdamW ($lr = 2 \\times 10^{-5}$, weight decay $0.01$) con precisión mixta automática (AMP FP16) en los Tensor Cores de la GPU RTX 5070."""),

    code_cell("""# 3. Método 1: Entrenamiento y Evaluación del Pipeline Clásico
classical_model = ClassicalNLPBaseline(classifier_type="svm", max_features=25000, ngram_range=(1, 2), C=1.0)

# Entrenar sobre train_df
fit_time = classical_model.fit(train_df["text"], train_df["label"])

# Predicción en el conjunto de prueba
test_preds_m1, test_probs_m1 = classical_model.predict(test_df["text"])
metrics_m1 = compute_classification_metrics(test_df["label"], test_preds_m1, y_probs=test_probs_m1, class_names=AG_NEWS_CLASSES)
print_metrics_summary(metrics_m1, model_name="Método 1: TF-IDF + LinearSVC")"""),

    code_cell("""# 4. Método 2: Preparación de Tokenizer y DataLoaders para Transformer
tokenizer = get_nlp_tokenizer("distilbert-base-uncased")

train_loader, val_loader, test_loader = create_nlp_dataloaders(
    tokenizer=tokenizer,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    batch_size=64,
    max_length=128,
    num_workers=hw_info["recommended_workers"]
)"""),

    code_cell("""# 5. Método 2: Entrenamiento de DistilBERT con Fine-Tuning y AMP
model_transformer = TransformerClassifier(model_name="distilbert-base-uncased", num_labels=4, freeze_backbone=False)

optimizer_transformer = torch.optim.AdamW(model_transformer.parameters(), lr=2e-5, weight_decay=0.01)
criterion_transformer = torch.nn.CrossEntropyLoss()

trainer_transformer = Trainer(
    model=model_transformer,
    optimizer=optimizer_transformer,
    criterion=criterion_transformer,
    device=hw_info["device"],
    use_amp=hw_info["use_amp"],
    checkpoint_dir=CHECKPOINTS_DIR,
    model_name="agnews_distilbert"
)

# Fine-Tuning por 3 épocas (suficiente para convergencia óptima en Transformers)
history_transformer = trainer_transformer.fit(train_loader, val_loader, epochs=3, early_stopping_patience=2)"""),

    md_cell("""---
### 4. Análisis de Convergencia (Sección 5.4)
Se visualiza la evolución del error de entropía cruzada y la exactitud durante el fine-tuning del Transformer."""),

    code_cell("""# Curvas de convergencia para DistilBERT
plot_convergence_curves(history_transformer, title="Curvas de Convergencia: DistilBERT Fine-Tuning")
plt.show()"""),

    md_cell("""---
### 5. Comparación Cuantitativa y Discusión de Resultados (Sección 5.3)
Se contrastan ambos métodos considerando:
* Desempeño predictivo (Accuracy, F1-Score).
* Complejidad del modelo y costo computacional (tiempo de inferencia, número de parámetros)."""),

    code_cell("""# Evaluación de DistilBERT en el conjunto de prueba
test_preds_m2, test_probs_m2, _ = trainer_transformer.predict(test_loader)
metrics_m2 = compute_classification_metrics(test_df["label"], test_preds_m2, y_probs=test_probs_m2, class_names=AG_NEWS_CLASSES)

comparison_df = pd.DataFrame({
    "Métrica": ["Accuracy (%)", "Precision Macro (%)", "Recall Macro (%)", "F1-Score Macro (%)", "Complejidad"],
    "Método 1 (TF-IDF + LinearSVC)": [
        metrics_m1["accuracy"] * 100,
        metrics_m1["precision_macro"] * 100,
        metrics_m1["recall_macro"] * 100,
        metrics_m1["f1_macro"] * 100,
        "25,000 pesos lineales (0.1 MB)"
    ],
    "Método 2 (DistilBERT)": [
        metrics_m2["accuracy"] * 100,
        metrics_m2["precision_macro"] * 100,
        metrics_m2["recall_macro"] * 100,
        metrics_m2["f1_macro"] * 100,
        "66.3M parámetros (260 MB)"
    ]
})

print("="*75)
print(" COMPARATIVA DE MODELOS EN TEST (AG NEWS)")
print("="*75)
print(comparison_df.to_string(index=False))"""),

    code_cell("""# Matrices de Confusión Comparativas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# M1
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(test_df["label"], test_preds_m1, normalize='true')
sns.heatmap(cm1, annot=True, fmt=".2f", cmap="Blues", xticklabels=AG_NEWS_CLASSES, yticklabels=AG_NEWS_CLASSES, ax=axes[0])
axes[0].set_title("Método 1: TF-IDF + LinearSVC", fontweight='bold')
axes[0].set_ylabel("True Label")
axes[0].set_xlabel("Predicted")

# M2
cm2 = confusion_matrix(test_df["label"], test_preds_m2, normalize='true')
sns.heatmap(cm2, annot=True, fmt=".2f", cmap="Greens", xticklabels=AG_NEWS_CLASSES, yticklabels=AG_NEWS_CLASSES, ax=axes[1])
axes[1].set_title("Método 2: DistilBERT Transformer", fontweight='bold')
axes[1].set_ylabel("True Label")
axes[1].set_xlabel("Predicted")

plt.tight_layout()
plt.show()"""),

    md_cell("""---
### 6. Interpretabilidad y Análisis del Espacio de Características (Sección 5.5)
1. **N-gramas Más Influyentes:** Inspección de los pesos de los hiperplanos lineales de LinearSVC para identificar las palabras clave que definen cada una de las 4 clases.
2. **Proyección t-SNE de Embeddings:** Mapeo 2D de las representaciones contextuales $[CLS]$ generadas por DistilBERT para evaluar la separación semántica de las noticias."""),

    code_cell("""# Interpretabilidad 1: Palabras más informativas por clase en LinearSVC
feature_names = classical_model.vectorizer.get_feature_names_out()
coef = classical_model.model.coef_

plt.figure(figsize=(15, 8))
for i, class_name in enumerate(AG_NEWS_CLASSES):
    plt.subplot(2, 2, i + 1)
    top_indices = np.argsort(coef[i])[-10:]
    top_features = feature_names[top_indices]
    top_weights = coef[i][top_indices]
    
    plt.barh(range(10), top_weights, color=sns.color_palette("tab10")[i])
    plt.yticks(range(10), top_features, fontsize=10)
    plt.title(f"Top Palabras - Clase: {class_name}", fontweight='bold')
    plt.xlabel("Peso del Coeficiente")
plt.tight_layout()
plt.suptitle("Interpretabilidad de Características (TF-IDF Coeficientes)", y=1.02, fontsize=14, fontweight='bold')
plt.show()"""),

    code_cell("""# Interpretabilidad 2: t-SNE sobre los Embeddings [CLS] de DistilBERT
feats_m2, labels_m2 = extract_torch_embeddings(model_transformer, test_loader, hw_info["device"], max_batches=20)

plot_latent_space(
    features=feats_m2,
    labels=labels_m2,
    class_names=AG_NEWS_CLASSES,
    method="tsne",
    n_samples=800,
    title="Interpretabilidad: Proyección t-SNE del Espacio Semántico ([CLS] DistilBERT)"
)
plt.show()"""),

    md_cell("""---
### 7. Estudio de Ablación (Sección 6.1)
Se compara el impacto de la arquitectura y la estrategia de transferencia:
* **Ablación 1:** Transformer con Backbone Congelado (Feature Extractor) vs Fine-Tuning Completo.
* **Ablación 2:** Impacto del tamaño del vocabulario TF-IDF ($5,000$ vs $25,000$ vs $50,000$ n-gramas)."""),

    code_cell("""# Estudio de Ablación: Tamaño de Vocabulario en TF-IDF
vocab_sizes = [5000, 15000, 25000, 50000]
ablation_accs = []

for vs in vocab_sizes:
    m = ClassicalNLPBaseline(classifier_type="svm", max_features=vs)
    m.fit(train_df["text"], train_df["label"])
    p, _ = m.predict(test_df["text"])
    acc = (p == test_df["label"]).mean()
    ablation_accs.append(acc * 100)
    print(f"Vocabulario {vs:>5d} features -> Test Accuracy: {acc*100:.2f}%")

plt.figure(figsize=(7, 4))
plt.plot(vocab_sizes, ablation_accs, 'o-', color='navy', linewidth=2, markersize=8)
plt.title("Ablación: Exactitud vs Tamaño del Vocabulario TF-IDF", fontweight='bold')
plt.xlabel("Número Máximo de Features")
plt.ylabel("Accuracy en Test (%)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()""")
]

with open(os.path.join(NOTEBOOKS_DIR, "01_food101_classification.ipynb"), "w") as f:
    json.dump(make_notebook(food_cells), f, indent=2)

with open(os.path.join(NOTEBOOKS_DIR, "02_agnews_classification.ipynb"), "w") as f:
    json.dump(make_notebook(nlp_cells), f, indent=2)

print("Notebooks creados con éxito en:", NOTEBOOKS_DIR)
