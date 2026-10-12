import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ClassicalNLPBaseline:
    """
    Método 1: Pipeline Clásico de Machine Learning para Procesamiento de Lenguaje Natural.
    Combina representación TF-IDF con un Clasificador Lineal (LinearSVC o Logistic Regression).
    
    Espacio de Hipótesis:
        f(x) = argmax_c (w_c^T phi(x) + b_c), donde phi(x) in R^V es el vector TF-IDF.
    Objetivo de Optimización:
        Minimización de Pérdida Hinge (SVM) o Pérdida Logística Multiclase con regularización L2.
    """
    def __init__(self, classifier_type="svm", max_features=25000, ngram_range=(1, 2), C=1.0):
        self.classifier_type = classifier_type
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words='english'
        )
        if classifier_type == "svm":
            self.model = LinearSVC(C=C, random_state=42, max_iter=2000)
        elif classifier_type == "logistic":
            self.model = LogisticRegression(C=C, max_iter=1000, random_state=42, solver='lbfgs')
        else:
            self.model = SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, random_state=42)
            
    def fit(self, train_texts, train_labels):
        print(f"[Classical NLP] Extrayendo características TF-IDF (vocabulario máx: {self.vectorizer.max_features})...")
        t0 = time.time()
        X_train = self.vectorizer.fit_transform(train_texts)
        tfidf_time = time.time() - t0
        print(f"[Classical NLP] Vectorización completada en {tfidf_time:.2f}s. Dimensión: {X_train.shape}")
        
        print(f"[Classical NLP] Entrenando clasificador ({self.classifier_type.upper()})...")
        t0 = time.time()
        self.model.fit(X_train, train_labels)
        train_time = time.time() - t0
        print(f"[Classical NLP] Entrenamiento finalizado en {train_time:.2f}s.")
        return {"tfidf_time": tfidf_time, "train_time": train_time}
        
    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        preds = self.model.predict(X)
        probs = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            dec = self.model.decision_function(X)
            # Softmax sobre decision scores para obtener probabilidades aproximadas
            exp_dec = np.exp(dec - np.max(dec, axis=1, keepdims=True))
            probs = exp_dec / np.sum(exp_dec, axis=1, keepdims=True)
        return preds, probs

class TransformerClassifier(nn.Module):
    """
    Método 2: Deep Learning con Transfer Learning utilizando una arquitectura Transformer (DistilBERT).
    
    Espacio de Hipótesis:
        Representación contextualizada profunda h = Transformer(x), seguida de un cabezal de clasificación:
        logits = W_c * h_[CLS] + b_c
    Objetivo de Optimización:
        Cross-Entropy Loss entrenada con optimizador AdamW con weight decay y warm-up.
    """
    def __init__(self, model_name="distilbert-base-uncased", num_labels=4, freeze_backbone=False):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        if freeze_backbone:
            # Congelar capas base para ablación de Transfer Learning (Feature Extractor puro)
            if hasattr(self.transformer, "distilbert"):
                for param in self.transformer.distilbert.parameters():
                    param.requires_grad = False
            elif hasattr(self.transformer, "bert"):
                for param in self.transformer.bert.parameters():
                    param.requires_grad = False
            print(f"[Transformer] Backbone congelado ({model_name}). Solo el cabezal es entrenable.")
            
    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return outputs.logits

    def extract_features(self, inputs):
        """Extrae la representación latente (penúltima capa) para análisis t-SNE / PCA / interpretabilidad."""
        if hasattr(self.transformer, "distilbert"):
            out = self.transformer.distilbert(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None)
            )
            # Retorna el embedding del token [CLS] (posición 0)
            return out.last_hidden_state[:, 0, :]
        else:
            return self.transformer(**inputs).logits
            
def get_nlp_tokenizer(model_name="distilbert-base-uncased"):
    """Instancia el tokenizador pre-entrenado correspondiente."""
    return AutoTokenizer.from_pretrained(model_name)
