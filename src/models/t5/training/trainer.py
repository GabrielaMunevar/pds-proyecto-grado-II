"""
Entrenador del modelo T5.

Este módulo contiene funciones para entrenar el modelo T5 para generar PLS.
"""

import json
from pathlib import Path
import warnings
import sys

# Transformers imports
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import torch

# Importar utilidades y configuración
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from models.t5.config import (
    apply_prompt, 
    get_prompt, 
    get_t5_config,
    get_t5_training_config,
    get_t5_chunking_config
)
from models.t5.chunking.chunker import expand_pairs_with_chunking, expand_pairs_with_chunking_by_tokens
from models.t5.length_analysis.analyzer import analyze_lengths_before_after_chunking
from models.t5.length_analysis.colab_analyzer import analyze_document_lengths
from models.t5.evaluation.metrics import compute_metrics

warnings.filterwarnings('ignore')


def load_synthetic_pairs(pairs_file: str = 'data/processed/synthetic_pairs/synthetic_pairs.jsonl'):
    """
    Carga los pares sintéticos creados.
    
    Args:
        pairs_file: Ruta al archivo de pares sintéticos
    
    Returns:
        Lista de pares válidos o None si no se encuentra el archivo
    """
    print("Cargando pares sintéticos...")
    
    pairs_path = Path(pairs_file)
    
    if not pairs_path.exists():
        print("No se encontraron pares sintéticos. Ejecutar primero:")
        print("   python src/data/create_synthetic_pairs.py")
        return None
    
    pairs = []
    with open(pairs_path, 'r', encoding='utf-8') as f:
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


def tokenize_function(examples, tokenizer, max_length_source=400, max_length_target=256):
    """
    Tokeniza los ejemplos para entrenamiento.
    Aplica el prompt estandarizado al texto técnico.
    
    Esta es la implementación exacta del Colab que usa as_target_tokenizer.
    
    Args:
        examples: Ejemplos con 'texto_tecnico' y 'texto_simple'
        tokenizer: Tokenizer T5
        max_length_source: Longitud máxima del input (incluye prompt)
        max_length_target: Longitud máxima del output
    
    Returns:
        Diccionario con inputs tokenizados y labels
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
    
    # Tokenizar targets usando as_target_tokenizer (método del Colab)
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


def train_t5_generator(
    pairs,
    model_name: str = 't5-base',
    output_dir: str = 'models/t5_generator',
    max_length_source: int = None,
    max_length_target: int = None,
    num_epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    use_chunking: bool = True
):
    """
    Entrena generador T5.
    
    Args:
        pairs: Lista de pares (texto_tecnico, texto_simple)
        model_name: Nombre del modelo base (t5-base, t5-small, etc.)
        output_dir: Directorio donde guardar el modelo
        max_length_source: Longitud máxima del input
        max_length_target: Longitud máxima del output
        num_epochs: Número de épocas
        batch_size: Tamaño del batch
        learning_rate: Tasa de aprendizaje
        use_chunking: Si True, aplica chunking a documentos largos
    
    Returns:
        Tupla (trainer, tokenizer, metrics)
    """
    print("\n=== ENTRENANDO GENERADOR T5 ===")
    
    # Obtener configuración de T5
    t5_config = get_t5_config(model_name)
    training_config = get_t5_training_config()
    chunking_config = get_t5_chunking_config()
    
    # Usar valores por defecto de la configuración si no se especifican
    if max_length_source is None:
        max_length_source = t5_config['max_length_source']
    if max_length_target is None:
        max_length_target = t5_config['max_length_target']
    if num_epochs is None:
        num_epochs = 3  # Valor por defecto
    if batch_size is None:
        batch_size = training_config.get('batch_size', 8)
    if learning_rate is None:
        learning_rate = training_config.get('learning_rate', 3e-4)
    
    # Tokenizer y modelo
    print(f"Cargando {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ANÁLISIS DE LONGITUDES ANTES DEL ENTRENAMIENTO
    if use_chunking:
        print("\n" + "="*80)
        print("ANÁLISIS DE LONGITUDES DE DOCUMENTOS")
        print("="*80)
        
        # Analizar longitudes originales (antes de chunking)
        from utils.length_analysis import analyze_lengths_for_training
        
        length_analysis = analyze_lengths_for_training(
            pairs=pairs,
            tokenizer=tokenizer,
            max_length_source=max_length_source,
            save_report=True,
            output_dir=output_path / 'length_analysis' / 'before_chunking'
        )
        
        # Aplicar chunking a documentos largos
        print("\n" + "="*80)
        print("APLICANDO CHUNKING A DOCUMENTOS LARGOS")
        print("="*80)
        print(f"  (Documentos > {max_length_source} tokens se dividirán en chunks con overlap)")
        
        # Expandir pares con chunking usando configuración
        expanded_pairs = expand_pairs_with_chunking(
            pairs, 
            tokenizer, 
            max_tokens=chunking_config['max_tokens']
        )
        
        # Analizar longitudes después del chunking
        length_analysis_after = analyze_lengths_for_training(
            pairs=expanded_pairs,
            tokenizer=tokenizer,
            max_length_source=max_length_source,
            save_report=True,
            output_dir=output_path / 'length_analysis' / 'after_chunking'
        )
        
        print(f"\n COMPARACIÓN:")
        print(f"  Documentos originales: {len(pairs)}")
        print(f"  Documentos después de chunking: {len(expanded_pairs)}")
        print(f"  Incremento: {len(expanded_pairs) - len(pairs)} documentos ({((len(expanded_pairs) - len(pairs)) / len(pairs) * 100):.1f}%)")
        print(f"  Documentos truncados antes: {length_analysis['technical']['truncation']['num_truncated']}")
        print(f"  Documentos truncados después: {length_analysis_after['technical']['truncation']['num_truncated']}")
        
        pairs = expanded_pairs
    
    # Separar train y validation
    train_pairs = [p for p in pairs if p.get('split', 'train') == 'train']
    val_pairs = [p for p in pairs if p.get('split', 'test') == 'test']
    
    print(f"\nTrain pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    
    # Crear datasets
    train_dataset = Dataset.from_list(train_pairs)
    val_dataset = Dataset.from_list(val_pairs)
    
    # Tokenizar datasets
    print("Tokenizando datasets...")
    
    def tokenize_wrapper(examples):
        return tokenize_function(examples, tokenizer, max_length=max_length_target)
    
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
    
    # Argumentos de entrenamiento usando configuración
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=training_config.get('warmup_steps', 500),
        weight_decay=training_config.get('weight_decay', 0.01),
        learning_rate=learning_rate,
        logging_dir=str(output_path / 'logs'),
        logging_steps=100,
        eval_strategy=training_config.get('eval_strategy', 'epoch'),
        save_strategy=training_config.get('save_strategy', 'epoch'),
        save_total_limit=training_config.get('save_total_limit', 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        fp16=training_config.get('fp16', False) and torch.cuda.is_available(),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 2),
    )
    
    # Trainer
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
    trainer.save_model(str(output_path / 'model'))
    tokenizer.save_pretrained(str(output_path / 'tokenizer'))
    
    # Guardar métricas
    metrics = {
        'model': model_name,
        'training_loss': float(eval_results.get('train_loss', 0.0)),
        'eval_loss': float(eval_results['eval_loss']),
        'train_samples': len(train_dataset),
        'eval_samples': len(val_dataset),
        'epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'max_length_source': max_length_source,
        'max_length_target': max_length_target,
        'use_chunking': use_chunking
    }
    
    metrics_path = output_path / 'training' / 'metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nModelo guardado en: {output_path}")
    
    return trainer, tokenizer, metrics


def train_t5_large(
    train_pairs: list,
    test_pairs: list,
    model_name: str = 't5-large',
    output_dir: str = 'models/t5_large_pls',
    use_chunking_by_tokens: bool = True,
    max_tokens_chunk: int = 450,
    overlap_tokens: int = 50
):
    """
    Entrena T5-large con todas las mejoras implementadas (versión exacta del Colab).
    
    Esta función replica exactamente la funcionalidad de train_t5_large del Colab,
    pero organizada en módulos.
    
    Args:
        train_pairs: Lista de pares de entrenamiento
        test_pairs: Lista de pares de test
        model_name: Nombre del modelo (t5-large, t5-base, etc.)
        output_dir: Directorio donde guardar el modelo
        use_chunking_by_tokens: Si True, usa chunking por tokens (método del Colab)
        max_tokens_chunk: Máximo de tokens por chunk (450 deja ~60 para el prompt)
        overlap_tokens: Tokens de overlap entre chunks
    
    Returns:
        Tupla (trainer, tokenizer, metrics)
    """
    print("\n" + "="*80)
    print("ENTRENANDO T5-LARGE (VERSIÓN COLAB)")
    print("="*80)
    
    # Obtener configuración
    t5_config = get_t5_config(model_name)
    training_config = get_t5_training_config()
    
    # Limpiar memoria GPU antes de cargar modelo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Memoria GPU antes de cargar modelo: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Cargar tokenizer y modelo
    print(f"\nCargando {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Mover modelo a GPU si está disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        max_length=t5_config['max_context']
    )
    
    # APLICAR CHUNKING
    if use_chunking_by_tokens:
        print("\n" + "="*80)
        print("APLICANDO CHUNKING (POR TOKENS)")
        print("="*80)
        print(f"  Max tokens por chunk: {max_tokens_chunk} (deja ~60 tokens para el prompt)")
        print(f"  Overlap: {overlap_tokens} tokens")
        
        train_expanded = expand_pairs_with_chunking_by_tokens(
            train_pairs, 
            tokenizer, 
            max_tokens=max_tokens_chunk,
            overlap_tokens=overlap_tokens
        )
        test_expanded = expand_pairs_with_chunking_by_tokens(
            test_pairs, 
            tokenizer, 
            max_tokens=max_tokens_chunk,
            overlap_tokens=overlap_tokens
        )
    else:
        train_expanded = train_pairs
        test_expanded = test_pairs
    
    # ANÁLISIS DESPUÉS DEL CHUNKING
    print("\n" + "="*80)
    print("ANÁLISIS DE LONGITUDES (DESPUÉS DE CHUNKING)")
    print("="*80)
    print("  (Analizando los chunks generados, no los textos originales)")
    length_stats_after = analyze_document_lengths(
        train_expanded + test_expanded, 
        tokenizer, 
        max_length=t5_config['max_context']
    )
    
    # Crear datasets
    print("\nCreando datasets...")
    train_dataset = Dataset.from_list(train_expanded)
    test_dataset = Dataset.from_list(test_expanded)
    
    # Crear dev set desde train (10% estratificado)
    train_size = len(train_dataset)
    dev_size = int(0.1 * train_size)
    train_dataset = train_dataset.shuffle(seed=42)
    dev_dataset = train_dataset.select(range(dev_size))
    train_dataset = train_dataset.select(range(dev_size, train_size))
    
    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")
    
    # Tokenizar
    print("Tokenizando datasets...")
    print(f"  Max length source: {t5_config['max_context']} (incluye prompt)")
    print(f"  Max length target: {t5_config['max_length_target']}")
    
    def tokenize_wrapper(examples):
        return tokenize_function(
            examples, 
            tokenizer, 
            max_length_source=t5_config['max_context'],
            max_length_target=t5_config['max_length_target']
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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=training_config.get('num_epochs', 4),
        per_device_train_batch_size=training_config.get('batch_size', 2),
        per_device_eval_batch_size=training_config.get('eval_batch_size', 1),  # Batch size muy pequeño para evaluación
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 16),
        learning_rate=training_config.get('learning_rate', 3e-5),
        warmup_ratio=0.1,
        weight_decay=training_config.get('weight_decay', 0.01),
        logging_dir=str(output_path / 'logs'),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Usar loss en vez de métricas para ahorrar memoria
        greater_is_better=False,
        fp16=training_config.get('fp16', True) and torch.cuda.is_available(),
        dataloader_num_workers=training_config.get('dataloader_num_workers', 2),
        dataloader_pin_memory=True,
        report_to="none",
        push_to_hub=False,
        max_grad_norm=1.0,
        remove_unused_columns=True,
        # Optimizaciones de memoria para evaluación
        eval_accumulation_steps=training_config.get('eval_accumulation_steps', 8),  # Acumular durante eval
        prediction_loss_only=False,  # Necesitamos las predicciones para métricas
    )
    
    # Limpiar memoria antes de crear trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Para ahorrar memoria durante entrenamiento, NO calcular métricas en cada epoch
    # Solo calcular loss y calcular métricas completas al final en test set
    # Esto evita generar secuencias completas durante cada evaluación
    use_metrics_during_eval = False  # Desactivar para ahorrar memoria
    
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
    
    # Entrenar
    print("\n" + "="*80)
    print("INICIANDO ENTRENAMIENTO")
    print("="*80)
    
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
    print("Calculando loss en test set...")
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
        output_dir=str(output_path / 'eval_temp'),
        per_device_eval_batch_size=1,  # Batch muy pequeño
        fp16=training_config.get('fp16', True) and torch.cuda.is_available(),
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
    print(f"\nGuardando modelo en {output_path}...")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_path))
    
    # Guardar métricas y estadísticas
    metrics = {
        'model': model_name,
        'test_metrics': {k: float(v) for k, v in test_results.items()},
        'length_stats_before': length_stats_before,
        'length_stats_after': length_stats_after,
        'train_samples': len(train_dataset),
        'dev_samples': len(dev_dataset),
        'test_samples': len(test_dataset),
        'config': {
            'model_name': model_name,
            'max_context': t5_config['max_context'],
            'max_target': t5_config['max_length_target'],
            'use_chunking_by_tokens': use_chunking_by_tokens,
            'max_tokens_chunk': max_tokens_chunk,
            'overlap_tokens': overlap_tokens
        }
    }
    
    metrics_path = output_path / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n Modelo entrenado y guardado en: {output_path}")
    
    return trainer, tokenizer, metrics

