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

# Importar utilidades y configuración
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.text_chunking import split_into_chunks
from utils.length_analysis import analyze_lengths_for_training
from config import apply_prompt, get_prompt

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


def expand_pairs_with_chunking(pairs, tokenizer, max_tokens=400):
    """
    Expande pares largos en múltiples pares de chunks para entrenamiento.
    
    Si un documento técnico es muy largo, lo divide en chunks y crea
    múltiples ejemplos de entrenamiento. Cada chunk se empareja con
    el texto simple completo (si es corto) o con chunks del texto simple.
    
    Args:
        pairs: Lista de pares (texto_tecnico, texto_simple)
        tokenizer: Tokenizer para contar tokens
        max_tokens: Máximo de tokens por chunk
    
    Returns:
        Lista expandida de pares
    """
    expanded_pairs = []
    total_chunks = 0
    
    for pair in pairs:
        texto_tecnico = pair['texto_tecnico']
        texto_simple = pair['texto_simple']
        
        # Contar tokens del texto técnico
        tech_tokens = len(tokenizer.encode(texto_tecnico, add_special_tokens=False))
        simple_tokens = len(tokenizer.encode(texto_simple, add_special_tokens=False))
        
        # Si el texto técnico es corto, usar par original
        if tech_tokens <= max_tokens:
            expanded_pairs.append(pair)
        else:
            # Dividir texto técnico en chunks
            tech_chunks = split_into_chunks(
                texto_tecnico,
                tokenizer=tokenizer.encode,
                max_tokens=max_tokens,
                overlap=50
            )
            
            # Si el texto simple también es largo, dividirlo
            if simple_tokens > max_tokens * 2:
                simple_chunks = split_into_chunks(
                    texto_simple,
                    tokenizer=tokenizer.encode,
                    max_tokens=max_tokens * 2,  # Texto simple puede ser más largo
                    overlap=50
                )
                # Emparejar chunks técnicos con chunks simples (mismo índice o circular)
                for i, tech_chunk in enumerate(tech_chunks):
                    simple_chunk = simple_chunks[min(i, len(simple_chunks) - 1)]
                    expanded_pairs.append({
                        'texto_tecnico': tech_chunk,
                        'texto_simple': simple_chunk,
                        'split': pair.get('split', 'train'),
                        'source': pair.get('source', 'unknown'),
                        'is_chunk': True,
                        'chunk_idx': i,
                        'total_chunks': len(tech_chunks)
                    })
            else:
                # Usar texto simple completo para cada chunk técnico
                for i, tech_chunk in enumerate(tech_chunks):
                    expanded_pairs.append({
                        'texto_tecnico': tech_chunk,
                        'texto_simple': texto_simple,
                        'split': pair.get('split', 'train'),
                        'source': pair.get('source', 'unknown'),
                        'is_chunk': True,
                        'chunk_idx': i,
                        'total_chunks': len(tech_chunks)
                    })
            
            total_chunks += len(tech_chunks)
    
    print(f"Pares expandidos: {len(expanded_pairs)} (originales: {len(pairs)}, chunks creados: {total_chunks})")
    return expanded_pairs

def tokenize_function(examples, tokenizer, max_length=256):
    """Función de tokenización para T5."""
    # Usar prompt estándar centralizado
    inputs = [apply_prompt(text) for text in examples['texto_tecnico']]
    
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
    
    # ANÁLISIS DE LONGITUDES ANTES DEL ENTRENAMIENTO
    print("\n" + "="*80)
    print("ANÁLISIS DE LONGITUDES DE DOCUMENTOS")
    print("="*80)
    
    # Analizar longitudes originales (antes de chunking)
    length_analysis = analyze_lengths_for_training(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length_source=400,  # Considerando el prompt
        save_report=True,
        output_dir=Path('models/t5_generator')
    )
    
    # Aplicar chunking a documentos largos
    print("\n" + "="*80)
    print("APLICANDO CHUNKING A DOCUMENTOS LARGOS")
    print("="*80)
    print("  (Documentos > 400 tokens se dividirán en chunks con overlap)")
    
    # Expandir pares con chunking
    expanded_pairs = expand_pairs_with_chunking(pairs, tokenizer, max_tokens=400)
    
    # Analizar longitudes después del chunking
    print("\n" + "="*80)
    print("ANÁLISIS DESPUÉS DEL CHUNKING")
    print("="*80)
    length_analysis_after = analyze_lengths_for_training(
        pairs=expanded_pairs,
        tokenizer=tokenizer,
        max_length_source=400,
        save_report=False  # No guardar, solo mostrar
    )
    
    print(f"\n📊 COMPARACIÓN:")
    print(f"  Documentos originales: {len(pairs)}")
    print(f"  Documentos después de chunking: {len(expanded_pairs)}")
    print(f"  Incremento: {len(expanded_pairs) - len(pairs)} documentos ({((len(expanded_pairs) - len(pairs)) / len(pairs) * 100):.1f}%)")
    print(f"  Documentos truncados antes: {length_analysis['technical']['truncation']['num_truncated']}")
    print(f"  Documentos truncados después: {length_analysis_after['technical']['truncation']['num_truncated']}")
    
    # Usar MUESTRA MUY PEQUEÑA para testing rápido
    print("Usando muestra ULTRA pequeña para test rápido...")
    
    # Tomar solo 100 pares para testing ultra-rápido
    train_sample = [p for p in expanded_pairs if p['split'] == 'train'][:80]
    val_sample = [p for p in expanded_pairs if p['split'] == 'test'][:20]
    
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
    
    # Ejemplos de prueba (usando prompt estándar)
    test_texts = [
        apply_prompt("This randomized controlled trial evaluated the efficacy of metformin in patients with type 2 diabetes mellitus."),
        apply_prompt("The study demonstrated significant improvement in glycemic control among participants receiving the intervention."),
        apply_prompt("Adverse events were reported in 15% of patients in the treatment group compared to 8% in the control group.")
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
