#!/usr/bin/env python3
"""
Entrenar Generador T5 para PLS
Entrena un modelo T5 para generar resúmenes en lenguaje sencillo.

Uso:
    python src/models/train_t5_generator.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings

# Transformers imports
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset
import torch

warnings.filterwarnings('ignore')

def load_synthetic_pairs():
    """Carga los pares sintéticos creados."""
    print("Cargando pares sintéticos...")
    
    pairs_file = Path('data/processed/synthetic_pairs/synthetic_pairs.jsonl')
    
    if not pairs_file.exists():
        print("No se encontraron pares sintéticos. Ejecutar primero:")
        print("   python src/data/create_synthetic_pairs.py")
        return None
    
    pairs = []
    with open(pairs_file, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    
    print(f"Pares cargados: {len(pairs)}")
    
    # Filtrar pares válidos
    valid_pairs = []
    for pair in pairs:
        if (len(pair['texto_tecnico'].split()) >= 20 and 
            len(pair['texto_simple'].split()) >= 10):
            valid_pairs.append(pair)
    
    print(f"Pares válidos: {len(valid_pairs)}")
    
    return valid_pairs

def tokenize_function(examples, tokenizer, max_length=256):
    """Función de tokenización para T5."""
    # T5 usa un formato de prompt: "simplify: <texto_técnico>"
    inputs = [f"simplify: {text}" for text in examples['texto_tecnico']]
    
    # Tokenizar inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    # Tokenizar targets (texto simple)
    labels = tokenizer(
        examples['texto_simple'],
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    # En T5, labels que no se deben usar para loss se ponen en -100
    labels = labels['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100
    
    model_inputs['labels'] = labels
    return model_inputs

def train_t5_generator(pairs):
    """Entrena generador T5."""
    print("\n=== ENTRENANDO GENERADOR T5 ===")
    
    # Tokenizer y modelo (usando T5-SMALL para velocidad)
    print("Cargando T5-SMALL (más rápido que T5-BASE)...")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    # Usar MUESTRA MUY PEQUEÑA para testing rápido
    print("Usando muestra ULTRA pequeña para test rápido...")
    
    # Tomar solo 100 pares para testing ultra-rápido
    train_sample = [p for p in pairs if p['split'] == 'train'][:80]
    val_sample = [p for p in pairs if p['split'] == 'test'][:20]
    
    print(f"Train sample: {len(train_sample)}, Val sample: {len(val_sample)}")
    
    # Crear datasets directamente
    train_dataset = Dataset.from_list(train_sample)
    val_dataset = Dataset.from_list(val_sample)
    
    # Tokenizar datasets
    print("Tokenizando datasets...")
    
    # Crear función wrapper para tokenize_function
    def tokenize_wrapper(examples):
        return tokenize_function(examples, tokenizer, max_length=128)
    
    train_dataset = train_dataset.map(
        tokenize_wrapper,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        tokenize_wrapper,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    print("¡Datos preparados!")
    
    # Argumentos de entrenamiento ULTRA-rápido para testing
    training_args = TrainingArguments(
        output_dir='models/t5_generator',
        num_train_epochs=1,
        per_device_train_batch_size=4,  # Batch más pequeño para CPU
        per_device_eval_batch_size=4,
        warmup_steps=2,  # Mínimo warmup
        weight_decay=0.01,
        logging_dir='logs',
        logging_steps=2,
        eval_strategy="no",
        save_strategy="no",
        report_to=None,
        dataloader_pin_memory=False,
        fp16=False,  # Sin FP16 en CPU (causa errores)
        gradient_accumulation_steps=4,
        max_steps=10,  # SOLO 10 pasos para verificar que funciona
        dataloader_num_workers=0,  # Sin paralelización
    )
    
    # Trainer (sin EarlyStopping para simplificar)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Entrenar
    print("\nIniciando entrenamiento de T5...")
    trainer.train()
    
    # Evaluar
    print("\nEvaluando modelo...")
    eval_results = trainer.evaluate()
    
    print(f"\n=== RESULTADOS T5 ===")
    print(f"Eval Loss: {eval_results['eval_loss']:.4f}")
    
    # Guardar modelo
    model_dir = Path('models/t5_generator')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(str(model_dir / 'model'))
    tokenizer.save_pretrained(str(model_dir / 'tokenizer'))
    
    # Guardar métricas
    metrics = {
        'model': 't5_generator',
        'eval_loss': float(eval_results['eval_loss']),
        'train_samples': len(train_dataset),
        'eval_samples': len(val_dataset),
        'epochs': 3
    }
    
    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nModelo guardado en: {model_dir}")
    
    return trainer, tokenizer, metrics

def test_generator(trainer, tokenizer):
    """Prueba el generador con algunos ejemplos."""
    print("\n=== PROBANDO GENERADOR T5 ===")
    
    # Ejemplos de prueba
    test_texts = [
        "simplify: This randomized controlled trial evaluated the efficacy of metformin in patients with type 2 diabetes mellitus.",
        "simplify: The study demonstrated significant improvement in glycemic control among participants receiving the intervention.",
        "simplify: Adverse events were reported in 15% of patients in the treatment group compared to 8% in the control group."
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nEjemplo {i+1}:")
        print(f"Input: {text}")
        
        # Tokenizar input
        inputs = tokenizer(text, return_tensors='pt', max_length=256, truncation=True)
        
        # Generar
        with torch.no_grad():
            outputs = trainer.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        
        # Decodificar output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")

def main():
    # Cargar pares sintéticos
    pairs = load_synthetic_pairs()
    
    if pairs is None:
        return
    
    # Entrenar T5
    trainer, tokenizer, metrics = train_t5_generator(pairs)
    
    # Probar generador
    test_generator(trainer, tokenizer)
    
    print("\nGenerador T5 entrenado exitosamente!")
    print("\nPróximo paso: Comparar T5 vs BART")

if __name__ == "__main__":
    main()
