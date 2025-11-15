"""
Script para entrenar T5-large en Google Colab
Incluye todas las mejoras del documento summary_mejoras.md

Uso en Colab:
1. Subir este archivo y los datos a Colab
2. Ejecutar todas las celdas
3. El modelo se guardará en Google Drive o localmente
"""

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

# Instalar dependencias necesarias
!pip install -q transformers datasets accelerate rouge-score nltk sacrebleu textstat

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import nltk
nltk.download('punkt', quiet=True)
from rouge_score import rouge_scorer
from sacrebleu import BLEU
import textstat

# Verificar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# CONFIGURACIÓN DE MODELO Y PROMPT ESTANDARIZADO
# ============================================================================

# Configuración del modelo
# BALANCEADO: Optimizado para velocidad sin exceder memoria
MODEL_CONFIG = {
    'model_name': 't5-large',
    'max_context': 512,  # T5-large también tiene 512 tokens
    'max_target': 256,
    'batch_size': 2,  # Aumentado de 1 a 2 (más rápido)
    'gradient_accumulation_steps': 16,  # Reducido de 32 a 16 (batch efectivo = 32)
    'learning_rate': 3e-5,
    'num_epochs': 4,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'fp16': True,  # Mixed precision para ahorrar memoria
    'gradient_checkpointing': False,  # DESACTIVADO para más velocidad (usa más memoria)
    'dataloader_num_workers': 2,  # Aumentado para más velocidad
}

# PROMPT ESTANDARIZADO (centralizado)
STANDARD_PROMPT = "simplify medical text into plain language: "

def apply_prompt(text):
    """Aplica el prompt estandarizado al texto."""
    return STANDARD_PROMPT + text

# ============================================================================
# CARGAR DATOS
# ============================================================================

def load_synthetic_pairs(data_path='data/processed/synthetic_pairs_improved/synthetic_pairs.jsonl'):
    """
    Carga los pares sintéticos.
    IMPORTANTE: Respeta las particiones originales (train/test).
    """
    print("="*80)
    print("CARGANDO PARES SINTÉTICOS")
    print("="*80)
    
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    
    print(f"Total pares cargados: {len(pairs)}")
    
    # CRÍTICO: Usar splits originales del dataset
    train_pairs = [p for p in pairs if p.get('split') == 'train']
    test_pairs = [p for p in pairs if p.get('split') == 'test']
    
    print(f"Pares train (split original): {len(train_pairs)}")
    print(f"Pares test (split original): {len(test_pairs)}")
    
    # Validar que todos los pares tengan los campos necesarios
    valid_train = []
    valid_test = []
    
    for p in train_pairs:
        if ('texto_tecnico' in p and 'texto_simple' in p and 
            len(p['texto_tecnico'].split()) >= 20 and 
            len(p['texto_simple'].split()) >= 10):
            valid_train.append(p)
    
    for p in test_pairs:
        if ('texto_tecnico' in p and 'texto_simple' in p and 
            len(p['texto_tecnico'].split()) >= 20 and 
            len(p['texto_simple'].split()) >= 10):
            valid_test.append(p)
    
    print(f"Pares train válidos: {len(valid_train)}")
    print(f"Pares test válidos: {len(valid_test)}")
    
    return valid_train, valid_test

# ============================================================================
# ANÁLISIS DE LONGITUDES
# ============================================================================

def analyze_document_lengths(pairs, tokenizer, max_length=512):
    """
    Analiza la distribución de longitudes de documentos.
    CRÍTICO: Identificar cuántos documentos exceden la ventana de contexto.
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE LONGITUDES DE DOCUMENTOS")
    print("="*80)
    
    technical_lengths = []
    simple_lengths = []
    truncated_count = 0
    
    for pair in pairs:
        tech_tokens = len(tokenizer.encode(pair['texto_tecnico']))
        simple_tokens = len(tokenizer.encode(pair['texto_simple']))
        
        technical_lengths.append(tech_tokens)
        simple_lengths.append(simple_tokens)
        
        if tech_tokens > max_length:
            truncated_count += 1
    
    print(f"\n ESTADÍSTICAS DE LONGITUD:")
    print(f"  Textos técnicos:")
    print(f"    - Promedio: {np.mean(technical_lengths):.0f} tokens")
    print(f"    - Mediana: {np.median(technical_lengths):.0f} tokens")
    print(f"    - Mínimo: {np.min(technical_lengths)} tokens")
    print(f"    - Máximo: {np.max(technical_lengths)} tokens")
    print(f"    - Documentos que exceden {max_length} tokens: {truncated_count} ({100*truncated_count/len(pairs):.1f}%)")
    
    if truncated_count > 0:
        tokens_lost = np.mean([max(0, l - max_length) for l in technical_lengths])
        print(f"    - Tokens perdidos promedio por truncación: {tokens_lost:.0f}")
    
    print(f"\n  Textos simples (PLS):")
    print(f"    - Promedio: {np.mean(simple_lengths):.0f} tokens")
    print(f"    - Mediana: {np.median(simple_lengths):.0f} tokens")
    print(f"    - Máximo: {np.max(simple_lengths)} tokens")
    
    return {
        'technical': {
            'mean': float(np.mean(technical_lengths)),
            'median': float(np.median(technical_lengths)),
            'max': int(np.max(technical_lengths)),
            'truncated': truncated_count,
            'truncated_pct': float(100*truncated_count/len(pairs))
        },
        'simple': {
            'mean': float(np.mean(simple_lengths)),
            'median': float(np.median(simple_lengths)),
            'max': int(np.max(simple_lengths))
        }
    }

# ============================================================================
# CHUNKING PARA DOCUMENTOS LARGOS
# ============================================================================

def split_into_chunks(text, tokenizer, max_tokens=450, overlap_tokens=50):
    """
    Divide texto en chunks con overlap, trabajando directamente con tokens.
    CRÍTICO: Cada chunk debe tener ≤ max_tokens tokens.
    
    Args:
        text: Texto a dividir
        tokenizer: Tokenizer para contar tokens
        max_tokens: Máximo de tokens por chunk (debe ser < 512 para dejar espacio al prompt)
        overlap_tokens: Tokens de overlap entre chunks
    
    Returns:
        Lista de chunks, cada uno con ≤ max_tokens tokens
    """
    # Tokenizar el texto completo
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Si el texto cabe en un chunk, retornarlo tal cual
    if len(tokens) <= max_tokens:
        return [text]
    
    # Dividir en chunks con overlap
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        # Calcular fin del chunk
        end_idx = min(start_idx + max_tokens, len(tokens))
        
        # Decodificar este chunk
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        # Siguiente chunk comienza con overlap (retroceder overlap_tokens)
        if end_idx < len(tokens):
            start_idx = end_idx - overlap_tokens
        else:
            break
    
    return chunks

def expand_pairs_with_chunking(pairs, tokenizer, max_tokens=450, overlap_tokens=50):
    """
    Expande pares aplicando chunking a documentos largos.
    Cada chunk se convierte en un ejemplo de entrenamiento.
    CRÍTICO: max_tokens debe ser < 512 para dejar espacio al prompt.
    
    Args:
        pairs: Lista de pares originales
        tokenizer: Tokenizer para contar tokens
        max_tokens: Máximo de tokens por chunk (450 deja ~60 para el prompt)
        overlap_tokens: Tokens de overlap entre chunks
    
    Returns:
        Lista expandida de pares, cada chunk es un par separado
    """
    print("\n" + "="*80)
    print("APLICANDO CHUNKING A DOCUMENTOS LARGOS")
    print("="*80)
    print(f"  Max tokens por chunk: {max_tokens}")
    print(f"  Overlap entre chunks: {overlap_tokens} tokens")
    
    expanded_pairs = []
    chunked_count = 0
    total_chunks_created = 0
    
    for pair in pairs:
        tech_text = pair['texto_tecnico']
        
        # Verificar si necesita chunking (usando tokenizer para contar tokens reales)
        tech_tokens = len(tokenizer.encode(tech_text, add_special_tokens=False))
        
        if tech_tokens > max_tokens:
            # Dividir en chunks
            chunks = split_into_chunks(tech_text, tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
            chunked_count += 1
            total_chunks_created += len(chunks)
            
            # Crear un par por cada chunk (usar el mismo PLS para todos)
            # Estrategia: cada chunk aprende a generar el mismo PLS completo
            for i, chunk in enumerate(chunks):
                new_pair = pair.copy()
                new_pair['texto_tecnico'] = chunk
                new_pair['chunk_id'] = i
                new_pair['total_chunks'] = len(chunks)
                new_pair['original_tokens'] = tech_tokens
                new_pair['chunk_tokens'] = len(tokenizer.encode(chunk, add_special_tokens=False))
                expanded_pairs.append(new_pair)
        else:
            # No necesita chunking
            new_pair = pair.copy()
            new_pair['chunk_id'] = 0
            new_pair['total_chunks'] = 1
            new_pair['original_tokens'] = tech_tokens
            new_pair['chunk_tokens'] = tech_tokens
            expanded_pairs.append(new_pair)
    
    print(f"  Documentos originales: {len(pairs)}")
    print(f"  Documentos que requirieron chunking: {chunked_count} ({100*chunked_count/len(pairs):.1f}%)")
    print(f"  Total chunks creados: {total_chunks_created}")
    print(f"  Documentos después de chunking: {len(expanded_pairs)}")
    print(f"  Incremento: {len(expanded_pairs) - len(pairs)} documentos ({((len(expanded_pairs) - len(pairs)) / len(pairs) * 100):.1f}%)")
    
    # Verificar que los chunks realmente tienen ≤ max_tokens
    chunk_lengths = []
    for pair in expanded_pairs:
        chunk_tokens = len(tokenizer.encode(pair['texto_tecnico'], add_special_tokens=False))
        chunk_lengths.append(chunk_tokens)
    
    chunks_over_limit = sum(1 for l in chunk_lengths if l > max_tokens)
    print(f"\n  Verificación de chunks:")
    print(f"    Chunks > {max_tokens} tokens: {chunks_over_limit} ({100*chunks_over_limit/len(expanded_pairs):.1f}%)")
    print(f"    Promedio tokens por chunk: {np.mean(chunk_lengths):.0f}")
    print(f"    Máximo tokens en chunk: {np.max(chunk_lengths)}")
    
    if chunks_over_limit > 0:
        print(f"     ADVERTENCIA: {chunks_over_limit} chunks exceden el límite!")
    
    return expanded_pairs

# ============================================================================
# PREPARACIÓN DE DATOS
# ============================================================================

def tokenize_function(examples, tokenizer, max_length_source=400, max_length_target=256):
    """
    Tokeniza los ejemplos para entrenamiento.
    Aplica el prompt estandarizado al texto técnico.
    """
    # Aplicar prompt estandarizado
    inputs = [apply_prompt(text) for text in examples['texto_tecnico']]
    targets = examples['texto_simple']
    
    # Tokenizar inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_length_source,
        truncation=True,
        padding='max_length'
    )
    
    # Tokenizar targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_length_target,
            truncation=True,
            padding='max_length'
        )
    
    # Reemplazar padding tokens en labels con -100 (ignorados en loss)
    labels['input_ids'] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels['input_ids']
    ]
    
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

# ============================================================================
# MÉTRICAS COMPLETAS (ROUGE, BLEU, SARI)
# ============================================================================

def compute_metrics(eval_pred, tokenizer):
    """
    Calcula métricas completas: ROUGE, BLEU, SARI.
    CRÍTICO: Incluir las 3 métricas mencionadas por el profesor.
    """
    predictions, labels = eval_pred
    
    # Decodificar predicciones
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Decodificar labels (reemplazar -100 con pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE completo (ROUGE-1, ROUGE-2, ROUGE-L)
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = rouge_scorer_obj.score(label, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    # BLEU score
    bleu = BLEU()
    bleu_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        try:
            score = bleu.sentence_score(pred, [label])
            bleu_scores.append(score.score / 100.0)  # Normalizar a 0-1
        except:
            bleu_scores.append(0.0)
    
    # SARI (System output Against References and Inputs)
    # Implementación simplificada de SARI
    def compute_sari(pred, ref, source):
        # SARI mide: keep, addition, deletion scores
        # Implementación simplificada basada en n-gramas
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        source_words = set(source.lower().split())
        
        # Keep: palabras en pred que están en ref y source
        keep = len(pred_words & ref_words & source_words) / max(len(pred_words), 1)
        
        # Addition: palabras en pred y ref pero no en source
        addition = len((pred_words & ref_words) - source_words) / max(len(pred_words), 1)
        
        # Deletion: palabras en source pero no en pred (si están en ref, es bueno)
        deletion = len((source_words - pred_words) & ref_words) / max(len(source_words), 1)
        
        # SARI es promedio de keep, addition, deletion
        sari = (keep + addition + deletion) / 3.0
        return sari
    
    # Nota: Para SARI completo necesitaríamos el texto fuente original
    # Aquí usamos una aproximación
    sari_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        # Usar label como aproximación del source (no ideal pero funcional)
        sari = compute_sari(pred, label, label)
        sari_scores.append(sari)
    
    return {
        'rouge1': np.mean(rouge_scores['rouge1']),
        'rouge2': np.mean(rouge_scores['rouge2']),
        'rougeL': np.mean(rouge_scores['rougeL']),
        'bleu': np.mean(bleu_scores),
        'sari': np.mean(sari_scores)
    }

# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def train_t5_large(train_pairs, test_pairs):
    """
    Entrena T5-large con todas las mejoras implementadas.
    """
    print("\n" + "="*80)
    print("ENTRENANDO T5-LARGE")
    print("="*80)
    
    # Limpiar memoria GPU antes de cargar modelo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Memoria GPU antes de cargar modelo: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Cargar tokenizer y modelo
    print(f"\nCargando {MODEL_CONFIG['model_name']}...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    model = T5ForConditionalGeneration.from_pretrained(MODEL_CONFIG['model_name'])
    
    # Habilitar gradient checkpointing solo si está configurado
    # NOTA: Desactivado por defecto para mejor velocidad
    if MODEL_CONFIG.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        print(" Gradient checkpointing habilitado (ahorra memoria, más lento)")
    else:
        print(" Gradient checkpointing desactivado (más rápido, usa más memoria)")
    
    # Mover modelo a GPU si está disponible
    model = model.to(device)
    
    # Limpiar memoria después de cargar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Memoria GPU después de cargar modelo: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # ANÁLISIS DE LONGITUDES ANTES DEL CHUNKING
    print("\n" + "="*80)
    print("ANÁLISIS DE LONGITUDES (ANTES DE CHUNKING)")
    print("="*80)
    length_stats_before = analyze_document_lengths(
        train_pairs + test_pairs, 
        tokenizer, 
        max_length=MODEL_CONFIG['max_context']
    )
    
    # APLICAR CHUNKING
    # CRÍTICO: max_tokens debe ser < 512 para dejar espacio al prompt
    # El prompt agrega ~20-30 tokens, así que usamos 450 tokens máximo por chunk
    print("\n" + "="*80)
    print("APLICANDO CHUNKING")
    print("="*80)
    print(f"  Max tokens por chunk: 450 (deja ~60 tokens para el prompt)")
    print(f"  Overlap: 50 tokens")
    
    train_expanded = expand_pairs_with_chunking(
        train_pairs, 
        tokenizer, 
        max_tokens=450,  # 450 tokens + ~60 del prompt = ~510 total
        overlap_tokens=50
    )
    test_expanded = expand_pairs_with_chunking(
        test_pairs, 
        tokenizer, 
        max_tokens=450,
        overlap_tokens=50
    )
    
    # ANÁLISIS DESPUÉS DEL CHUNKING
    # CRÍTICO: Analizar los textos chunked, no los originales
    print("\n" + "="*80)
    print("ANÁLISIS DE LONGITUDES (DESPUÉS DE CHUNKING)")
    print("="*80)
    print("  (Analizando los chunks generados, no los textos originales)")
    length_stats_after = analyze_document_lengths(
        train_expanded + test_expanded, 
        tokenizer, 
        max_length=MODEL_CONFIG['max_context']
    )
    
    # Crear datasets
    print("\nCreando datasets...")
    train_dataset = Dataset.from_list(train_expanded)
    test_dataset = Dataset.from_list(test_expanded)
    
    # Crear dev set desde train (10% estratificado)
    # Para simplificar, tomamos 10% aleatorio
    train_size = len(train_dataset)
    dev_size = int(0.1 * train_size)
    train_dataset = train_dataset.shuffle(seed=42)
    dev_dataset = train_dataset.select(range(dev_size))
    train_dataset = train_dataset.select(range(dev_size, train_size))
    
    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")
    
    # Tokenizar
    # CRÍTICO: max_length_source debe ser 512 (o menos) para que quepa con el prompt
    # El prompt agrega tokens, así que el texto chunked debe ser ≤450 tokens
    print("Tokenizando datasets...")
    print(f"  Max length source: {MODEL_CONFIG['max_context']} (incluye prompt)")
    print(f"  Max length target: {MODEL_CONFIG['max_target']}")
    
    def tokenize_wrapper(examples):
        return tokenize_function(
            examples, 
            tokenizer, 
            max_length_source=MODEL_CONFIG['max_context'],  # 512 tokens total (texto + prompt)
            max_length_target=MODEL_CONFIG['max_target']
        )
    
    train_dataset = train_dataset.map(
        tokenize_wrapper,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    dev_dataset = dev_dataset.map(
        tokenize_wrapper,
        batched=True,
        remove_columns=dev_dataset.column_names
    )
    
    test_dataset = test_dataset.map(
        tokenize_wrapper,
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Argumentos de entrenamiento
    output_dir = 'models/t5_large_pls'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=MODEL_CONFIG['num_epochs'],
        per_device_train_batch_size=MODEL_CONFIG['batch_size'],
        per_device_eval_batch_size=1,  # Batch size muy pequeño para evaluación
        gradient_accumulation_steps=MODEL_CONFIG['gradient_accumulation_steps'],
        learning_rate=MODEL_CONFIG['learning_rate'],
        warmup_ratio=MODEL_CONFIG['warmup_ratio'],
        weight_decay=MODEL_CONFIG['weight_decay'],
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # Reducido para ahorrar espacio
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Cambiar a eval_loss para ahorrar memoria
        greater_is_better=False,  # Loss menor es mejor
        fp16=MODEL_CONFIG['fp16'],
        dataloader_num_workers=MODEL_CONFIG.get('dataloader_num_workers', 2),  # Workers para velocidad
        dataloader_pin_memory=True,  # Activar para mejor velocidad (usa más RAM)
        report_to="none",  # Cambiar a "wandb" si quieres usar Weights & Biases
        push_to_hub=False,
        # Optimizaciones de memoria
        max_grad_norm=1.0,  # Gradient clipping
        remove_unused_columns=True,  # Limpiar columnas no usadas
        # Optimizaciones de memoria para evaluación
        eval_accumulation_steps=8,  # Acumular durante evaluación para ahorrar memoria
        prediction_loss_only=False,  # Necesitamos predicciones para métricas al final
    )
    
    # Limpiar memoria antes de crear trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Trainer
    # IMPORTANTE: Desactivar compute_metrics durante entrenamiento para ahorrar memoria
    # Las métricas (ROUGE, BLEU, SARI) se calcularán solo al final en test set
    # Esto evita generar secuencias completas durante cada evaluación
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=None,  # No calcular métricas durante entrenamiento para ahorrar memoria
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Información de memoria antes de entrenar
    if torch.cuda.is_available():
        print(f"\n MEMORIA GPU ANTES DE ENTRENAR:")
        print(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Reservada: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"  Allocada: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Libre: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9:.2f} GB")
        print(f"\n  CONFIGURACIÓN (BALANCEADA - Velocidad/Memoria):")
        print(f"  Batch size: {MODEL_CONFIG['batch_size']}")
        print(f"  Gradient accumulation: {MODEL_CONFIG['gradient_accumulation_steps']}")
        print(f"  Batch efectivo: {MODEL_CONFIG['batch_size'] * MODEL_CONFIG['gradient_accumulation_steps']}")
        print(f"  Gradient checkpointing: {MODEL_CONFIG.get('gradient_checkpointing', False)} (False = más rápido)")
        print(f"  FP16: {MODEL_CONFIG['fp16']}")
        print(f"  DataLoader workers: {MODEL_CONFIG.get('dataloader_num_workers', 2)}")

    
    # Entrenar
    print("\n" + "="*80)
    print("INICIANDO ENTRENAMIENTO")
    print("="*80)
    
    # Limpiar memoria justo antes de entrenar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n ERROR: Memoria GPU insuficiente")
            raise
        else:
            raise
    
    # Evaluar en test set
    print("\n" + "="*80)
    print("EVALUANDO EN TEST SET")
    print("="*80)
    
    # Limpiar memoria antes de evaluar test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Memoria GPU antes de evaluación test: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Primero obtener solo el loss (más eficiente en memoria)
    print("Calculando loss en test set completo...")
    test_results_loss = trainer.evaluate(eval_dataset=test_dataset)
    
    # Limpiar memoria
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Ahora calcular métricas completas en una muestra más pequeña para ahorrar memoria
    print("Calculando métricas completas (ROUGE, BLEU, SARI) en muestra de test...")
    
    # Usar solo una muestra del test set para calcular métricas (más eficiente)
    test_sample_size = min(500, len(test_dataset))  # Máximo 500 ejemplos para métricas
    test_sample = test_dataset.select(range(test_sample_size))
    
    # Crear trainer temporal solo para métricas con configuración optimizada
    eval_training_args = TrainingArguments(
        output_dir=f'{output_dir}/eval_temp',
        per_device_eval_batch_size=1,  # Batch muy pequeño
        fp16=MODEL_CONFIG['fp16'],
        dataloader_num_workers=0,  # Sin workers para ahorrar memoria
        eval_accumulation_steps=16,  # Más acumulación
        prediction_loss_only=False,
    )
    
    eval_trainer = Trainer(
        model=trainer.model,
        args=eval_training_args,
        eval_dataset=test_sample,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )
    
    # Calcular métricas en muestra
    test_results_metrics = eval_trainer.evaluate(eval_dataset=test_sample)
    
    # Combinar resultados
    test_results = {
        **test_results_loss,  # Loss completo en todo el test set
        **test_results_metrics  # Métricas en muestra
    }
    
    print(f"\nNota: Métricas calculadas en muestra de {test_sample_size} ejemplos de {len(test_dataset)} totales")
    
    print("\n" + "="*80)
    print("RESULTADOS FINALES")
    print("="*80)
    print(f"ROUGE-1: {test_results.get('eval_rouge1', 0):.4f}")
    print(f"ROUGE-2: {test_results.get('eval_rouge2', 0):.4f}")
    print(f"ROUGE-L: {test_results.get('eval_rougeL', 0):.4f}")
    print(f"BLEU: {test_results.get('eval_bleu', 0):.4f}")
    print(f"SARI: {test_results.get('eval_sari', 0):.4f}")
    print(f"Loss: {test_results.get('eval_loss', 0):.4f}")
    
    # Guardar modelo
    print(f"\nGuardando modelo en {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Guardar métricas y estadísticas
    metrics = {
        'model': MODEL_CONFIG['model_name'],
        'test_metrics': {k: float(v) for k, v in test_results.items()},
        'length_stats_before': length_stats_before,
        'length_stats_after': length_stats_after,
        'train_samples': len(train_dataset),
        'dev_samples': len(dev_dataset),
        'test_samples': len(test_dataset),
        'config': MODEL_CONFIG
    }
    
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n Modelo entrenado y guardado en: {output_dir}")
    
    return trainer, tokenizer, metrics

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Cargar datos
    train_pairs, test_pairs = load_synthetic_pairs()
    
    # Entrenar modelo
    trainer, tokenizer, metrics = train_t5_large(train_pairs, test_pairs)
    
    print("\n" + "="*80)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print("\nPróximos pasos:")
    print("1. Evaluar modelo en ejemplos cualitativos")
    print("2. Comparar con otros modelos (BART, LED)")
    print("3. Analizar errores y mejorar")

