import os
import time
import torch
import torch.nn as nn
from tqdm.auto import tqdm

class Trainer:
    """
    Entrenador genérico de PyTorch optimizado para GPU local (CUDA).
    Soporta Mixed Precision (AMP FP16), Gradient Scaling, Schedulers y Checkpointing.
    """
    def __init__(
        self,
        model,
        optimizer,
        criterion=None,
        scheduler=None,
        device=None,
        use_amp=True,
        checkpoint_dir="checkpoints",
        model_name="best_model"
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.scheduler = scheduler
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_times": []
        }
        
    def _train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch in pbar:
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[0].to(self.device, non_blocking=True), batch[1].to(self.device, non_blocking=True)
            elif isinstance(batch, dict):
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k != "label"}
                targets = batch["label"].to(self.device, non_blocking=True)
            else:
                raise ValueError("Formato de batch no soportado.")
                
            self.optimizer.zero_grad(set_to_none=True) # set_to_none=True ahorra memoria y acelera
            
            # Forward pass con Automatic Mixed Precision (AMP)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                else:
                    logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                
            # Backward pass escalado
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.1f}%"})
            
        return total_loss / total, correct / total

    def _eval_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0].to(self.device, non_blocking=True), batch[1].to(self.device, non_blocking=True)
                elif isinstance(batch, dict):
                    inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k != "label"}
                    targets = batch["label"].to(self.device, non_blocking=True)
                else:
                    continue
                    
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    if isinstance(inputs, dict):
                        outputs = self.model(**inputs)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    else:
                        logits = self.model(inputs)
                    loss = self.criterion(logits, targets)
                    
                total_loss += loss.item() * targets.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs=10, early_stopping_patience=None):
        best_val_acc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(self.checkpoint_dir, f"{self.model_name}.pt")
        
        print(f"\n[Trainer] Iniciando entrenamiento por {epochs} épocas...")
        print(f"[Trainer] Dispositivo: {self.device} | Mixed Precision (AMP): {self.use_amp}\n")
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._eval_epoch(val_loader)
            epoch_time = time.time() - t0
            
            if self.scheduler:
                self.scheduler.step()
                
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_times"].append(epoch_time)
            
            lr_current = self.optimizer.param_groups[0]['lr']
            print(f"Época [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | LR: {lr_current:.6f} | "
                  f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}%")
            
            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  --> ¡Nuevo mejor modelo guardado en {best_model_path}! (Val Acc: {val_acc*100:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    print(f"\n[Early Stopping] Detención temprana alcanzada tras {early_stopping_patience} épocas sin mejora.")
                    break
                    
        # Cargar los mejores pesos encontrados
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print(f"[Trainer] Cargados los mejores pesos de {best_model_path}")
            
        return self.history

    def predict(self, dataloader):
        """Genera predicciones y probabilidades para todo el dataset."""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch[0].to(self.device, non_blocking=True), batch[1]
                elif isinstance(batch, dict):
                    inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k != "label"}
                    targets = batch["label"]
                else:
                    continue
                    
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    if isinstance(inputs, dict):
                        outputs = self.model(**inputs)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    else:
                        logits = self.model(inputs)
                    probs = torch.softmax(logits, dim=-1)
                    preds = logits.argmax(dim=-1)
                    
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy() if torch.is_tensor(targets) else targets)
                
        return all_preds, all_probs, all_targets
