import os
import random
import torch
import numpy as np

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
AGNEWS_DIR = os.path.join(DATA_DIR, "ag_news")
FOOD101_DIR = os.path.join(DATA_DIR, "food101")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "Cod", "checkpoints")

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

def setup_hardware():
    """Configura optimizaciones para la GPU local y núcleos de CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        # Habilitar TensorFloat-32 (TF32) para Tensor Cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Habilitar cuDNN auto-tuner para convoluciones más rápidas
        torch.backends.cudnn.benchmark = True
        
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cap = torch.cuda.get_device_capability(0)
        print(f"[Hardware] GPU Detectada: {gpu_name}")
        print(f"[Hardware] VRAM Total: {vram_gb:.2f} GB | Capacidad de Cómputo: sm_{cap[0]}{cap[1]}")
        print(f"[Hardware] TF32 habilitado: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"[Hardware] cuDNN Benchmark habilitado: {torch.backends.cudnn.benchmark}")
    else:
        print("[Hardware] Advertencia: Ejecutando en CPU.")
        
    num_cpus = os.cpu_count() or 4
    # Usar hasta 8 workers para dataloaders manteniendo headroom para el sistema
    recommended_workers = min(8, max(2, num_cpus // 3))
    print(f"[Hardware] CPU Cores: {num_cpus} | Workers recomendados para DataLoader: {recommended_workers}")
    
    return {
        "device": device,
        "recommended_workers": recommended_workers,
        "use_amp": torch.cuda.is_available(),
        "amp_dtype": torch.float16 if torch.cuda.is_available() else torch.float32
    }

def set_seed(seed=42):
    """Garantiza la reproducibilidad de los experimentos fijando las semillas aleatorias."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinismo en cuDNN
        torch.backends.cudnn.deterministic = False # False para mantener cuDNN benchmark optimizado a velocidad máxima
    print(f"[Reproducibilidad] Semilla establecida en: {seed}")
