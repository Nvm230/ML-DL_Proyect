import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

def compute_classification_metrics(y_true, y_pred, y_probs=None, class_names=None):
    """
    Calcula Accuracy, Precision, Recall y F1-score (macro y weighted).
    Opcionalmente calcula Top-5 accuracy si se proporcionan probabilidades.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    
    # Métricas macro (pondera cada clase por igual)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Métricas weighted (pondera por soporte)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
    }
    
    if y_probs is not None:
        y_probs = np.asarray(y_probs)
        if y_probs.shape[1] >= 5:
            # Top-5 Accuracy
            top5_preds = np.argsort(y_probs, axis=1)[:, -5:]
            top5_correct = np.any(top5_preds == y_true[:, None], axis=1)
            metrics["top5_accuracy"] = float(np.mean(top5_correct))
            
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Matriz de Confusión", normalize=True, max_classes_display=20, figsize=(10, 8)):
    """
    Genera y grafica la matriz de confusión.
    Si hay muchas clases (ej. Food-101), permite recortar a las primeras N clases o las más frecuentes.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"
        
    n_classes = len(np.unique(y_true))
    
    # Si hay más de max_classes_display clases, recortar para que sea legible
    if n_classes > max_classes_display:
        print(f"[Visualización] Mostrando las primeras {max_classes_display} clases de {n_classes} para legibilidad.")
        cm_display = cm_display[:max_classes_display, :max_classes_display]
        if class_names:
            class_names = class_names[:max_classes_display]
            
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names if class_names is not None else "auto",
        yticklabels=class_names if class_names is not None else "auto",
        cbar=True,
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Etiqueta Verdadera (Ground Truth)", fontsize=12)
    ax.set_xlabel("Etiqueta Predicha", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig, ax

def print_metrics_summary(metrics_dict, model_name="Modelo"):
    """Imprime un resumen legible de las métricas obtenidas."""
    print("=" * 60)
    print(f" RESUMEN DE EVALUACIÓN - {model_name}")
    print("=" * 60)
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            print(f"  • {k.replace('_', ' ').capitalize():<24}: {v*100:.2f}% ({v:.4f})")
        else:
            print(f"  • {k.replace('_', ' ').capitalize():<24}: {v}")
    print("=" * 60)
