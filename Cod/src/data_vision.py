import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

def get_food101_transforms(img_size=224, use_randaugment=True):
    """
    Retorna pipelines de transformaciones para entrenamiento y validación/prueba.
    Cumple con el plan experimental de preprocesamiento, escalamiento, normalización y data augmentation.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    train_transform_list = [
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    
    if use_randaugment:
        train_transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
    else:
        train_transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    val_transform_list = [
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ]
    
    return transforms.Compose(train_transform_list), transforms.Compose(val_transform_list)

class Food101Dataset(Dataset):
    """
    Dataset optimizado para Food-101 que lee directamente las imágenes y etiquetas.
    Soporta rutas desde el archivo meta de ETHZ (train.txt, test.txt).
    """
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Encontrar directorio base (maneja data/food101 o data/food101/food-101)
        base_dir = root_dir
        if os.path.exists(os.path.join(root_dir, "food-101")):
            base_dir = os.path.join(root_dir, "food-101")
            
        self.images_dir = os.path.join(base_dir, "images")
        self.meta_dir = os.path.join(base_dir, "meta")
        
        # Cargar clases
        classes_file = os.path.join(self.meta_dir, "classes.txt")
        if os.path.exists(classes_file):
            with open(classes_file, "r") as f:
                self.classes = [line.strip() for line in f if line.strip()]
        else:
            self.classes = sorted(os.listdir(self.images_dir)) if os.path.exists(self.images_dir) else []
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Cargar lista de imágenes
        split_file = os.path.join(self.meta_dir, f"{split}.txt")
        self.samples = []
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                for line in f:
                    rel_path = line.strip()
                    if not rel_path:
                        continue
                    img_path = os.path.join(self.images_dir, f"{rel_path}.jpg")
                    cls_name = rel_path.split("/")[0]
                    label = self.class_to_idx.get(cls_name, 0)
                    self.samples.append((img_path, label))
        else:
            print(f"[Advertencia] Archivo {split_file} no encontrado aún.")
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def create_food101_dataloaders(
    root_dir="/home/nvm/Documents/UTEC/MLDP/data/food101",
    batch_size=64,
    img_size=224,
    num_workers=4,
    subset_fraction=1.0,
    val_split_from_train=0.1,
    random_state=42
):
    """
    Genera DataLoaders de entrenamiento, validación y prueba para Food-101.
    """
    train_transform, val_transform = get_food101_transforms(img_size=img_size)
    
    full_train_ds = Food101Dataset(root_dir, split="train", transform=train_transform)
    test_ds = Food101Dataset(root_dir, split="test", transform=val_transform)
    
    # Separar train en train y validation
    n_total = len(full_train_ds)
    if n_total == 0:
        raise RuntimeError("No se encontraron imágenes en el directorio de Food-101.")
        
    n_val = int(n_total * val_split_from_train)
    n_train = n_total - n_val
    
    g = torch.Generator().manual_seed(random_state)
    train_idx, val_idx = torch.utils.data.random_split(range(n_total), [n_train, n_val], generator=g)
    
    train_subset = Subset(full_train_ds, train_idx)
    # Crear una versión con transform de validación para el conjunto de validación
    val_base_ds = Food101Dataset(root_dir, split="train", transform=val_transform)
    val_subset = Subset(val_base_ds, val_idx)
    
    # Submuestreo opcional para pruebas rápidas
    if subset_fraction < 1.0:
        n_sub_train = int(len(train_subset) * subset_fraction)
        n_sub_val = int(len(val_subset) * subset_fraction)
        n_sub_test = int(len(test_ds) * subset_fraction)
        train_subset = Subset(train_subset, range(n_sub_train))
        val_subset = Subset(val_subset, range(n_sub_val))
        test_ds = Subset(test_ds, range(n_sub_test))
        print(f"[Food-101] Submuestreo al {subset_fraction*100:.1f}%: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_ds)}")
    else:
        print(f"[Food-101] Dataset completo: Train={len(train_subset):,}, Val={len(val_subset):,}, Test={len(test_ds):,}")
        
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader, test_loader, full_train_ds.classes
