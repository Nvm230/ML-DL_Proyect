import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

AG_NEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
LABEL_MAP = {i: name for i, name in enumerate(AG_NEWS_CLASSES)}

class TextClassificationDataset(Dataset):
    """
    Dataset PyTorch para clasificación de texto con HuggingFace Tokenizers.
    Convierte cadenas de texto en tensores input_ids y attention_mask.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def load_agnews_data(data_dir="/home/nvm/Documents/UTEC/MLDP/data/ag_news", val_split=0.1, sample_fraction=1.0, random_state=42):
    """
    Carga train.csv y test.csv de AG News y genera una partición estratificada de validación.
    Permite usar sample_fraction < 1.0 para iteraciones ultra rápidas si se desea.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Archivos de AG News no encontrados en {data_dir}. Asegúrese de ejecutar la descarga.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if sample_fraction < 1.0:
        train_df, _ = train_test_split(
            train_df,
            train_size=sample_fraction,
            random_state=random_state,
            stratify=train_df["label"]
        )
        train_df = train_df.reset_index(drop=True)
        print(f"[Data] Subconjunto muestreado: {len(train_df)} muestras de entrenamiento.")

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_split,
        random_state=random_state,
        stratify=train_df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[Data AG News] Conjunto de Entrenamiento : {len(train_df):,} muestras")
    print(f"[Data AG News] Conjunto de Validación    : {len(val_df):,} muestras")
    print(f"[Data AG News] Conjunto de Prueba        : {len(test_df):,} muestras")
    print(f"[Data AG News] Clases: {AG_NEWS_CLASSES}")

    return train_df, val_df, test_df

def create_nlp_dataloaders(tokenizer, train_df, val_df, test_df, batch_size=64, max_length=128, num_workers=4):
    """
    Crea DataLoaders de PyTorch optimizados para GPU con pin_memory y num_workers.
    """
    train_ds = TextClassificationDataset(train_df["text"], train_df["label"], tokenizer, max_length=max_length)
    val_ds = TextClassificationDataset(val_df["text"], val_df["label"], tokenizer, max_length=max_length)
    test_ds = TextClassificationDataset(test_df["text"], test_df["label"], tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, test_loader
