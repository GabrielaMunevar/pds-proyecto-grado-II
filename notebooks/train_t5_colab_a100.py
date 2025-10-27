"""
Entrenamiento T5 Optimizado para Google Colab A100
===================================================

Este script está optimizado para ejecutarse en Google Colab con GPU A100.
Divide el código en celdas según los comentarios # CELDA N:
"""

import torch
import numpy as np
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import warnings
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, DataLoader
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CELDA 1: Setup y configuración + Mount Drive
# ============================================================================
print("=" * 80)
print("CONFIGURACIÓN DE ENTRENAMIENTO T5")
print("=" * 80)

# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')
print("\nDrive montado exitosamente!")

# Verificar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")

# Configuración optimizada para A100
config = {
    'model_name': 't5-base',  # t5-base (220M) o t5-large (770M)
    'batch_size': 32,  # Aumentado para A100
    'gradient_accumulation_steps': 2,  # Batch efectivo: 64
    'learning_rate': 3e-5,
    'weight_decay': 0.01,
    'epochs': 5,
    'warmup_ratio': 0.1,
    'max_length_source': 512,  # Tokens entrada
    'max_length_target': 256,  # Tokens salida
    'save_steps': 500,
    'eval_steps': 250,
    'logging_steps': 50,
    'fp16': True,  # Mixed precision para A100
    'bf16': False,  # Si A100 soporta bf16, usar esto es mejor
    'dataloader_num_workers': 4,
    'dataloader_pin_memory': True,
    'gradient_checkpointing': True,  # Ahorra memoria
    'save_total_limit': 3,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'greater_is_better': False,
    'early_stopping_patience': 3,
    'seed': 42,
    'output_dir': '/content/models/t5_generator',
    'data_file': '/content/drive/MyDrive/T5_Training/synthetic_pairs.jsonl'  # Ruta desde Drive
}

print(f"\nConfiguración:")
for k, v in config.items():
    print(f"  {k}: {v}")

# ============================================================================
# CELDA 2: Cargar datos sintéticos
# ============================================================================
print("\n" + "=" * 80)
print("CARGANDO DATOS")
print("=" * 80)

def load_synthetic_pairs(data_path: str):
    """Cargar pares sintéticos desde JSONL"""
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs

# Cargar datos desde Drive
train_pairs = load_synthetic_pairs(config['data_file'])

# Dividir en train/eval (80/20)
np.random.seed(config['seed'])
shuffled_indices = np.random.permutation(len(train_pairs))
split_idx = int(len(train_pairs) * 0.8)

train_data = [train_pairs[i] for i in shuffled_indices[:split_idx]]
eval_data = [train_pairs[i] for i in shuffled_indices[split_idx:]]

print(f"Total pares: {len(train_pairs)}")
print(f"Train: {len(train_data)}")
print(f"Eval: {len(eval_data)}")
print(f"\nEjemplo train:")
print(f"  Técnico: {train_data[0]['texto_tecnico'][:100]}...")
print(f"  Simple: {train_data[0]['texto_simple'][:100]}...")

# ============================================================================
# CELDA 3: Preparar Dataset
# ============================================================================
print("\n" + "=" * 80)
print("PREPARANDO DATASET")
print("=" * 80)

class PLSDataset(Dataset):
    """Dataset para pares PLS (técnico → simple)"""
    
    def __init__(self, pairs, tokenizer, max_length_source, max_length_target):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length_source = max_length_source
        self.max_length_target = max_length_target
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        source = pair.get('input_text', pair['texto_tecnico'])  # Usar input_text si existe (con prefijo)
        target = pair['texto_simple']
        
        # Tokenizar
        source_encoded = self.tokenizer(
            source,
            max_length=self.max_length_source,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoded = self.tokenizer(
            target,
            max_length=self.max_length_target,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoded['input_ids'].squeeze(),
            'attention_mask': source_encoded['attention_mask'].squeeze(),
            'labels': target_encoded['input_ids'].squeeze()
        }

# Inicializar tokenizer
tokenizer = T5Tokenizer.from_pretrained(config['model_name'])
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

# Crear Preset
prepend_text = "simplify: "  # Prefijo para la tarea

# Agregar prefijo a todos los textos técnicos (no modificar el original)
for pair in train_data:
    pair['input_text'] = prepend_text + pair['texto_tecnico']
for pair in eval_data:
    pair['input_text'] = prepend_text + pair['texto_tecnico']

# Crear datasets
train_dataset = PLSDataset(train_data, tokenizer, config['max_length_source'], config['max_length_target'])
eval_dataset = PLSDataset(eval_data, tokenizer, config['max_length_source'], config['max_length_target'])

print(f"Train dataset: {len(train_dataset)} ejemplos")
print(f"Eval dataset: {len(eval_dataset)} ejemplos")

# ============================================================================
# CELDA 4: Cargar y configurar modelo
# ============================================================================
print("\n" + "=" * 80)
print("CARGANDO MODELO T5")
print("=" * 80)

model = T5ForConditionalGeneration.from_pretrained(config['model_name'])

# Configuración optimizada para A100
model.config.gradient_checkpointing = True
model.gradient_checkpointing_enable()  # Ahorra ~40% memoria

print(f"Modelo: {config['model_name']}")
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Mover a GPU
model = model.to(device)
print(f"Modelo cargado en {device}")

# ============================================================================
# CELDA 5: Configurar TrainingArguments
# ============================================================================
print("\n" + "=" * 80)
print("CONFIGURANDO ARGUMENTOS DE ENTRENAMIENTO")
print("=" * 80)

# Calcular steps totales
total_steps = len(train_dataset) // (config['batch_size'] * config['gradient_accumulation_steps']) * config['epochs']
warmup_steps = int(total_steps * config['warmup_ratio'])

print(f"Steps totales: {total_steps}")
print(f"Warmup steps: {warmup_steps}")

# Training arguments optimizado para A100
training_args = TrainingArguments(
    output_dir=config['output_dir'],
    num_train_epochs=config['epochs'],
    per_device_train_batch_size=config['batch_size'],
    per_device_eval_batch_size=config['batch_size'] * 2,  # Eval puede ser más grande
    gradient_accumulation_steps=config['gradient_accumulation_steps'],
    learning_rate=config['learning_rate'],
    weight_decay=config['weight_decay'],
    warmup_ratio=config['warmup_ratio'],
    
    # Optimización para A100
    fp16=config['fp16'],
    bf16=config['bf16'],
    dataloader_num_workers=config['dataloader_num_workers'],
    dataloader_pin_memory=config['dataloader_pin_memory'],
    gradient_checkpointing=config['gradient_checkpointing'],
    
    # Logging y checkpoints
    logging_strategy='steps',
    logging_steps=config['logging_steps'],
    eval_strategy='steps',
    eval_steps=config['eval_steps'],
    save_strategy='steps',
    save_steps=config['save_steps'],
    save_total_limit=config['save_total_limit'],
    
    # Early stopping
    load_best_model_at_end=config['load_best_model_at_end'],
    metric_for_best_model=config['metric_for_best_model'],
    greater_is_better=config['greater_is_better'],
    
    # Reproducibilidad
    seed=config['seed'],
    
    # Reportes
    report_to='tensorboard',  # Logging a TensorBoard
    logging_dir=f"{config['output_dir']}/logs",
    
    # Otros optimizadores
    remove_unused_columns=False,
    prediction_loss_only=True,
)

print("Training arguments configurados")

# ============================================================================
# CELDA 6: Definir función de compute_metrics (opcional)
# ============================================================================
print("\n" + "=" * 80)
print("CONFIGURANDO MÉTRICAS")
print("=" * 80)

def compute_metrics(eval_pred):
    """Calcular métricas durante evaluación"""
    predictions, labels = eval_pred
    
    # Decodificar predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calcular ROUGE (simple)
    rouge_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        # Overlap simple
        pred_tokens = set(pred.lower().split())
        label_tokens = set(label.lower().split())
        if len(label_tokens) > 0:
            overlap = len(pred_tokens & label_tokens) / len(label_tokens)
            rouge_scores.append(overlap)
    
    rouge_avg = np.mean(rouge_scores) if rouge_scores else 0.0
    
    return {
        'rouge_overlap': rouge_avg
    }

print("Métricas configuradas")

# ============================================================================
# CELDA 7: Crear Trainer y entrenar
# ============================================================================
print("\n" + "=" * 80)
print("INICIANDO ENTRENAMIENTO")
print("=" * 80)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience'])]
)

# Entrenar
train_result = trainer.train()

print("\n" + "=" * 80)
print("ENTRENAMIENTO COMPLETADO")
print("=" * 80)
print(f"Loss final: {train_result.training_loss:.4f}")
print(f"Steps totales: {train_result.global_step}")

# ============================================================================
# CELDA 8: Evaluación final
# ============================================================================
print("\n" + "=" * 80)
print("EVALUACIÓN FINAL")
print("=" * 80)

eval_results = trainer.evaluate()
print(f"Eval Loss: {eval_results['eval_loss']:.4f}")

# ============================================================================
# CELDA 9: Guardar modelo final
# ============================================================================
print("\n" + "=" * 80)
print("GUARDANDO MODELO FINAL")
print("=" * 80)

# Guardar modelo
model.save_pretrained(f"{config['output_dir']}/final_model")
tokenizer.save_pretrained(f"{config['output_dir']}/final_tokenizer")

print(f"Modelo guardado en: {config['output_dir']}/final_model")
print(f"Tokenizer guardado en: {config['output_dir']}/final_tokenizer")

# Guardar métricas
metrics = {
    'training_loss': train_result.training_loss,
    'eval_loss': eval_results['eval_loss'],
    'total_steps': train_result.global_step,
    'epochs': config['epochs'],
    'model_name': config['model_name'],
    'batch_size': config['batch_size'],
    'gradient_accumulation_steps': config['gradient_accumulation_steps']
}

with open(f"{config['output_dir']}/final_model/metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print("Métricas guardadas en metrics.json")

# ============================================================================
# CELDA 10: Copiar modelo a Google Drive
# ============================================================================
print("\n" + "=" * 80)
print("COPIANDO MODELO A GOOGLE DRIVE")
print("=" * 80)

import shutil
from pathlib import Path

# Crear directorio en Drive
drive_model_dir = '/content/drive/MyDrive/T5_Training/models/final_model'
Path(drive_model_dir).mkdir(parents=True, exist_ok=True)

print(f"\nCopiando modelo a: {drive_model_dir}")

# Copiar modelo
shutil.copytree(
    f"{config['output_dir']}/final_model",
    f"{drive_model_dir}/model",
    dirs_exist_ok=True
)

# Copiar tokenizer
shutil.copytree(
    f"{config['output_dir']}/final_tokenizer",
    f"{drive_model_dir}/tokenizer",
    dirs_exist_ok=True
)

# Copiar métricas
shutil.copy(
    f"{config['output_dir']}/final_model/metrics.json",
    f"{drive_model_dir}/metrics.json"
)

print("✅ Modelo copiado exitosamente a Google Drive!")
print(f"\nUbicación en Drive:")
print(f"  {drive_model_dir}/")
print(f"  - model/ (modelo)")
print(f"  - tokenizer/ (tokenizer)")
print(f"  - metrics.json (métricas)")

# ============================================================================
# CELDA 11: Test rápido del modelo
# ============================================================================
print("\n" + "=" * 80)
print("TEST RÁPIDO DEL MODELO")
print("=" * 80)

# Texto de prueba
test_text = "simplify: This randomized controlled trial evaluated the efficacy of metformin in patients with type 2 diabetes mellitus."

# Tokenizar
inputs = tokenizer(test_text, return_tensors='pt', max_length=512, truncation=True).to(device)

# Generar
with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=256,
        num_beams=4,
        early_stopping=True,
        length_penalty=1.1,
        do_sample=False
    )

# Decodificar
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Entrada:")
print(f"  {test_text}")
print("\nSalida generada:")
print(f"  {generated_text}")

print("\n" + "=" * 80)
print("SCRIPT COMPLETADO EXITOSAMENTE")
print("=" * 80)

