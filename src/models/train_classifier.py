"""
Script de entrenamiento del clasificador PLS/non-PLS.

Entrada: data/processed/{train,dev}.parquet
Salida: models/pls_classifier/*, data/processed/classified.parquet

Objetivo: Detectar si un texto ya es PLS (evitar generación innecesaria)

Modelos soportados:
- Baseline: TF-IDF + Logistic Regression / SVM
- Contextual: DistilBERT, BioBERT (recomendado)

Técnicas anti-desbalance:
- class_weight/pos_weight
- Focal loss
- Threshold tuning para F1_macro
"""

import yaml
import pandas as pd
import logging
import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params():
    """Carga parámetros desde params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


class PLSDataset(torch.utils.data.Dataset):
    """Dataset para clasificación PLS/non-PLS"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


def prepare_data(train_df, dev_df):
    """Prepara datos para clasificación"""
    # TODO: Implementar lógica de etiquetas
    # Positivos: label=pls o len_pls>0
    # Negativos: label=non_pls y len_pls==0
    
    logger.info(f"Train: {len(train_df)} registros")
    logger.info(f"Dev: {len(dev_df)} registros")
    
    return train_df, dev_df


def train_classifier(params, train_dataset, dev_dataset):
    """Entrena el clasificador"""
    model_name = params["classifier"]["model"]
    
    logger.info(f"Cargando modelo: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # PLS / non-PLS
    )
    
    # TODO: Implementar focal loss si está configurado
    
    training_args = TrainingArguments(
        output_dir="models/pls_classifier",
        learning_rate=params["classifier"]["lr"],
        num_train_epochs=params["classifier"]["epochs"],
        per_device_train_batch_size=params["classifier"]["batch_size"],
        per_device_eval_batch_size=params["classifier"]["batch_size"],
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="models/pls_classifier/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    
    # TODO: Implementar métricas personalizadas
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    
    logger.info("Iniciando entrenamiento...")
    trainer.train()
    
    return model, tokenizer


def evaluate_classifier(model, tokenizer, test_df):
    """Evalúa el clasificador"""
    # TODO: Implementar evaluación
    # - Matriz de confusión por fuente
    # - Threshold tuning para F1_macro
    # - Reportes detallados
    
    logger.info("Evaluando clasificador...")
    pass


def main():
    """Pipeline principal de entrenamiento"""
    params = load_params()
    
    logger.info("Iniciando entrenamiento del clasificador...")
    
    # Cargar datos
    train_df = pd.read_parquet("data/processed/train.parquet")
    dev_df = pd.read_parquet("data/processed/dev.parquet")
    
    # Preparar datos
    train_df, dev_df = prepare_data(train_df, dev_df)
    
    # TODO: Crear datasets PyTorch
    # train_dataset = PLSDataset(...)
    # dev_dataset = PLSDataset(...)
    
    # Entrenar
    # model, tokenizer = train_classifier(params, train_dataset, dev_dataset)
    
    # Guardar modelo
    # model.save_pretrained("models/pls_classifier")
    # tokenizer.save_pretrained("models/pls_classifier")
    
    logger.info("Clasificador entrenado y guardado")


if __name__ == "__main__":
    main()


