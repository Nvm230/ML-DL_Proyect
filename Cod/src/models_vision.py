import torch
import torch.nn as nn
from torchvision import models

class FoodVisionBaseline(nn.Module):
    """
    Método 1: Baseline basado en ResNet-18 pre-entrenado en ImageNet.
    
    Espacio de Hipótesis:
        f(x) = Softmax(W * GAP(ResNet18_backbone(x)) + b)
    Objetivo:
        Cross-Entropy Loss optimizado con SGD con momentum o Adam.
    """
    def __init__(self, num_classes=101, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("[FoodVisionBaseline] Backbone ResNet-18 congelado. Solo el cabezal se entrenará.")
            
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
        
    def extract_features(self, x):
        """Extrae la representación del pooling global (512 dimensiones) para t-SNE/PCA."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

class FoodVisionAdvanced(nn.Module):
    """
    Método 2: Transfer Learning con EfficientNet-B0 (arquitectura moderna con compound scaling).
    
    Espacio de Hipótesis:
        f(x) = Softmax(MLP(MBConv_blocks(x)))
    Objetivo:
        Cross-Entropy Loss con regularización label smoothing y optimizador AdamW con Cosine Annealing.
    """
    def __init__(self, num_classes=101, pretrained=True, dropout=0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
        
    def extract_features(self, x):
        """Extrae la representación previa al clasificador (1280 dimensiones)."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

class SimpleGradCAM:
    """
    Implementación ligera de Grad-CAM para interpretabilidad visual (Section 5.5).
    Genera mapas de activación sobre la última capa convolucional.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)
        
    def _save_activations(self, module, input, output):
        self.activations = output.detach()
        
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        with torch.enable_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=-1).item()
                
            self.model.zero_grad()
            loss = output[0, target_class]
            loss.backward()
            
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = torch.relu(cam)
            # Interpolar a la resolución de la imagen de entrada (ej. 224x224)
            cam = torch.nn.functional.interpolate(
                cam,
                size=(input_tensor.shape[2], input_tensor.shape[3]),
                mode='bilinear',
                align_corners=False
            )
            cam = cam.squeeze().detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam
