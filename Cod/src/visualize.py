import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

def plot_convergence_curves(history, title="Curvas de Convergencia (Loss y Accuracy)", save_path=None):
    """
    Grafica las curvas de pérdida y precisión para entrenamiento y validación.
    Permite diagnosticar underfitting, overfitting y estabilidad de optimización.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.set_theme(style="whitegrid")
    
    # 1. Pérdida (Cross-Entropy Loss)
    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#1f77b4', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 's--', label='Validation Loss', color='#ff7f0e', linewidth=2)
    axes[0].set_title("Función de Pérdida (Loss)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Épocas", fontsize=11)
    axes[0].set_ylabel("Pérdida (Loss)", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].set_xticks(list(epochs))
    
    # 2. Precisión (Accuracy)
    if 'train_acc' in history and history['train_acc']:
        axes[1].plot(epochs, [a * 100 if a <= 1.0 else a for a in history['train_acc']], 'o-', label='Train Acc (%)', color='#2ca02c', linewidth=2)
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(epochs, [a * 100 if a <= 1.0 else a for a in history['val_acc']], 's--', label='Validation Acc (%)', color='#d62728', linewidth=2)
    axes[1].set_title("Precisión (Accuracy %)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Épocas", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].set_xticks(list(epochs))
    
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualización] Gráfico de convergencia guardado en: {save_path}")
        
    return fig, axes

def plot_latent_space(features, labels, class_names=None, method="tsne", title=None, n_samples=1000, random_state=42, save_path=None):
    """
    Proyecta representaciones aprendidas a 2D mediante PCA, t-SNE o UMAP.
    Cumple con el requisito de interpretabilidad cualitativa del espacio latente.
    """
    features = np.asarray(features)
    labels = np.asarray(labels)
    
    # Submuestrear si el número de puntos es muy grande
    if len(features) > n_samples:
        indices = np.random.RandomState(random_state).choice(len(features), n_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
        
    print(f"[Interpretabilidad] Aplicando reducción {method.upper()} a {features.shape[0]} muestras con dimensión {features.shape[1]}...")
    
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(features)
        var_ratio = reducer.explained_variance_ratio_
        default_title = f"Interpretabilidad: PCA (Varianza explicada: {var_ratio[0]*100:.1f}% + {var_ratio[1]*100:.1f}%)"
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=random_state, max_iter=1000)
        coords = reducer.fit_transform(features)
        default_title = "Interpretabilidad: t-SNE del Espacio Latente Aprendido"
    elif method == "umap":
        if not HAS_UMAP:
            print("[Advertencia] umap-learn no está disponible, cambiando a PCA.")
            return plot_latent_space(features, labels, class_names, method="pca", title=title, n_samples=n_samples, random_state=random_state, save_path=save_path)
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(features)
        default_title = "Interpretabilidad: Proyección UMAP del Espacio Latente"
    else:
        raise ValueError(f"Método desconocido: {method}")
        
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    palette = sns.color_palette("tab10" if len(unique_labels) <= 10 else "husl", len(unique_labels))
    
    for idx, lab in enumerate(unique_labels):
        mask = (labels == lab)
        name = class_names[lab] if (class_names is not None and lab < len(class_names)) else f"Clase {lab}"
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[palette[idx]],
            label=name,
            alpha=0.7,
            edgecolors='none',
            s=40
        )
        
    ax.set_title(title or default_title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Componente 1", fontsize=11)
    ax.set_ylabel("Componente 2", fontsize=11)
    if len(unique_labels) <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=10)
    else:
        print("[Visualización] Muchas clases para mostrar leyenda individual completa.")
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualización] Proyección guardada en: {save_path}")
        
    return fig, ax

def extract_torch_embeddings(model, dataloader, device, max_batches=20):
    """Extrae las representaciones de la penúltima capa de una red neuronal PyTorch."""
    model.eval()
    embeddings_list = []
    labels_list = []
    
    # Hook para capturar la penúltima capa si el modelo tiene extract_features o pooling
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(device), batch[1]
            elif isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
                targets = batch["label"]
            else:
                continue
                
            # Extraer features
            if hasattr(model, "extract_features"):
                feats = model.extract_features(inputs)
            else:
                feats = model(inputs)
                if isinstance(feats, tuple):
                    feats = feats[0]
                    
            if torch.is_tensor(feats):
                feats = feats.view(feats.size(0), -1).cpu().numpy()
            embeddings_list.append(feats)
            labels_list.append(targets.cpu().numpy() if torch.is_tensor(targets) else np.array(targets))
            
    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return embeddings, labels
