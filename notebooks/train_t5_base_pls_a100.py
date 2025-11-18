"""
Script de entrenamiento T5-BASE para Plain Language Summaries (PLS) médicos
Con chunking semántico LangChain optimizado para GPU A100

Autor: Proyecto de Grado - Maestría
Fecha: 2025
GPU: A100 (40GB VRAM)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
# matplotlib y seaborn solo se usan en evaluación (evaluate_t5_base_pls_a100.py)
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# 1. INSTALACIÓN DE DEPENDENCIAS
# ============================================================================

def check_and_install_dependencies():
    """Verifica e instala dependencias si es necesario"""
    print("="*80)
    print("📦 VERIFICANDO DEPENDENCIAS")
    print("="*80)
    
    # Si el usuario quiere saltar la verificación (para cuando ya reinició)
    if os.environ.get('SKIP_INSTALL') == '1':
        print("✅ Saltando instalación (SKIP_INSTALL=1)\n")
        return False
    
    # Verificación simple: intentar importar transformers sin subprocess
    # Si ya pasó un reinicio, debería funcionar sin conflictos
    try:
        # Solo verificar que el módulo existe, sin ejecutar nada complejo
        import importlib.util
        spec = importlib.util.find_spec("transformers")
        if spec is not None:
            print("✅ Dependencias ya instaladas, saltando instalación\n")
            print("💡 Si esto causó errores, ejecuta antes del script:")
            print("   import os; os.environ['SKIP_INSTALL'] = '1'")
            return False
    except Exception as e:
        print(f"   (Verificación: {e})")
        pass
    
    print("📥 Instalando dependencias...\n")
    
    # Fix pyarrow incompatibility (común en Colab)
    print("  • Reinstalando pyarrow...")
    os.system("pip install -q --force-reinstall pyarrow")
    
    # Instalar en orden correcto con versiones compatibles
    print("  • Instalando transformers, datasets, accelerate...")
    os.system("pip install -q --upgrade 'transformers>=4.30.0' 'datasets>=2.14.0' 'accelerate>=0.20.0'")
    
    print("  • Instalando LangChain...")
    os.system("pip install -q 'langchain>=0.1.0' 'langchain-text-splitters>=0.0.1'")
    
    print("  • Instalando métricas...")
    os.system("pip install -q 'rouge-score>=0.1.2' 'sacrebleu>=2.3.0' 'bert-score>=0.3.13'")
    
    print("  • Instalando utilidades...")
    os.system("pip install -q 'textstat>=0.7.3' 'sentencepiece>=0.1.99'")
    
    print("\n✅ Instalación completada")
    print("⚠️  IMPORTANTE: Debes REINICIAR el runtime ahora\n")
    return True  # Necesita reinicio

# Verificar e instalar
needs_restart = check_and_install_dependencies()

if needs_restart:
    print("="*80)
    print("🔄 ACCIÓN REQUERIDA")
    print("="*80)
    print("1. Ve a: Runtime → Restart runtime")
    print("2. Vuelve a ejecutar: %run train_t5_base_pls_a100.py")
    print("   (El script detectará que ya está instalado y continuará)")
    print("="*80 + "\n")
    import sys
    sys.exit(0)

# ============================================================================
# 2. IMPORTACIONES
# ============================================================================

print("📚 Importando librerías...")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

# LangChain - intentar nueva ubicación primero, luego antigua
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from rouge_score import rouge_scorer
# sacrebleu, bert_score, textstat solo se usan en evaluación (evaluate_t5_base_pls_a100.py)
import nltk

# Descargar recursos NLTK
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Configurar matplotlib
# Estilos de matplotlib/seaborn solo se usan en evaluación

print("✅ Importaciones completadas\n")

# ============================================================================
# 3. CONFIGURACIÓN GLOBAL
# ============================================================================

class Config:
    """Configuración del experimento"""
    # Rutas - se configurarán dinámicamente en main()
    CSV_PATH = None  # Se detectará automáticamente
    DRIVE_BASE = '/content/drive/MyDrive/PLS_Project'
    MODEL_DIR = None
    RESULTS_DIR = None
    PLOTS_DIR = None
    
    # Modelo
    MODEL_NAME = 't5-base'
    TASK_PREFIX = 'simplify medical text: '
    
    # Tokenización
    MAX_INPUT_LENGTH = 512
    MAX_TARGET_LENGTH = 256
    
    # Chunking semántico
    CHUNK_SIZE = 400  # tokens
    CHUNK_OVERLAP = 50  # tokens
    SEPARATORS = ["\n\n", "\n", ". ", " "]
    
    # Training (optimizado para A100)
    NUM_EPOCHS = 3
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 16  # A100 puede manejar esto
    GRAD_ACCUM_STEPS = 2  # Effective batch = 32
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Evaluación
    EVAL_STEPS = 200
    SAVE_STEPS = 200
    SAVE_TOTAL_LIMIT = 3
    
    # Generación
    NUM_BEAMS = 4
    
    # Semilla
    SEED = 42

config = Config()

def find_csv_file():
    """Busca el archivo CSV en múltiples ubicaciones posibles"""
    print("="*80)
    print("🔍 BUSCANDO ARCHIVO CSV")
    print("="*80)
    
    # Posibles ubicaciones
    possible_paths = [
        '/content/drive/MyDrive/PLS_Project/data/pls_10k_final_aprobados.csv',
        '/content/drive/MyDrive/PLS_Project/pls_10k_final_aprobados.csv',
        '/content/drive/My Drive/PLS_Project/data/pls_10k_final_aprobados.csv',
        '/content/drive/My Drive/PLS_Project/pls_10k_final_aprobados.csv',
        '/content/pls_10k_final_aprobados.csv',  # En local
    ]
    
    # Buscar en ubicaciones conocidas
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Archivo encontrado: {path}\n")
            return path
    
    # Si no se encuentra, buscar en todo Drive
    print("⏳ Buscando en todo Drive (esto puede tomar unos segundos)...")
    try:
        import subprocess
        result = subprocess.run(
            ['find', '/content/drive/MyDrive', '-name', 'pls_10k_final_aprobados.csv', '-type', 'f'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.stdout.strip():
            found_path = result.stdout.strip().split('\n')[0]
            print(f"✅ Archivo encontrado: {found_path}\n")
            return found_path
    except:
        pass
    
    # Si no se encuentra, ofrecer subir archivo
    print("❌ No se encontró el archivo CSV en Drive\n")
    print("="*80)
    print("💡 SOLUCIONES:")
    print("="*80)
    print("\n📤 OPCIÓN 1: Subir archivo a Colab (MÁS RÁPIDO)")
    print("   Ejecuta en una celda:")
    print("   from google.colab import files")
    print("   uploaded = files.upload()")
    print("   # Selecciona: pls_10k_final_aprobados.csv")
    print("   # Luego vuelve a ejecutar el script")
    
    print("\n📁 OPCIÓN 2: Especificar ruta manualmente")
    print("   Si sabes dónde está el archivo, ejecuta:")
    print("   import os")
    print("   os.environ['CSV_PATH'] = '/ruta/completa/al/archivo.csv'")
    print("   # Luego vuelve a ejecutar el script")
    
    print("\n📋 OPCIÓN 3: Listar contenido de Drive")
    print("   Ejecuta para ver la estructura:")
    print("   !ls -R /content/drive/MyDrive/PLS_Project/")
    print("\n" + "="*80)
    
    return None

def setup_config():
    """Configura las rutas de Config"""
    # Buscar CSV
    csv_path = os.environ.get('CSV_PATH')  # Primero verificar si se especificó manualmente
    
    if not csv_path:
        csv_path = find_csv_file()
    
    if not csv_path:
        print("\n❌ No se puede continuar sin el archivo CSV")
        import sys
        sys.exit(1)
    
    # Configurar rutas
    config.CSV_PATH = csv_path
    config.MODEL_DIR = f'{config.DRIVE_BASE}/models/t5_pls/final'
    config.RESULTS_DIR = f'{config.DRIVE_BASE}/results'
    config.PLOTS_DIR = f'{config.RESULTS_DIR}/plots'
    
    # Crear directorios
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    # Mostrar configuración
    print("="*80)
    print("⚙️  CONFIGURACIÓN")
    print("="*80)
    print(f"CSV: {config.CSV_PATH}")
    print(f"Modelo: {config.MODEL_NAME}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Batch size: {config.BATCH_SIZE} (effective: {config.BATCH_SIZE * config.GRAD_ACCUM_STEPS})")
    print(f"Max input: {config.MAX_INPUT_LENGTH} tokens")
    print(f"Max output: {config.MAX_TARGET_LENGTH} tokens")
    print(f"Chunking: {config.CHUNK_SIZE} tokens (overlap: {config.CHUNK_OVERLAP})")
    print("="*80 + "\n")

# ============================================================================
# 4. CARGAR Y PREPARAR DATOS
# ============================================================================

def cargar_datos(csv_path: str) -> pd.DataFrame:
    """Carga y valida el dataset"""
    print("="*80)
    print("📂 CARGA DE DATOS")
    print("="*80)
    
    df = pd.read_csv(csv_path)
    print(f"✅ Dataset cargado: {len(df):,} filas")
    
    # Renombrar columnas
    df = df.rename(columns={
        'texto_original': 'input',
        'pls_generado': 'target'
    })
    
    # Limpiar
    df = df.dropna(subset=['input', 'target'])
    df = df[df['input'].str.strip() != '']
    df = df[df['target'].str.strip() != '']
    
    print(f"✅ Después de limpieza: {len(df):,} filas")
    
    # Estadísticas
    df['input_words'] = df['input'].str.split().str.len()
    df['target_words'] = df['target'].str.split().str.len()
    
    print(f"\n📊 Estadísticas:")
    print(f"  Input words: {df['input_words'].mean():.1f} ± {df['input_words'].std():.1f}")
    print(f"  Target words: {df['target_words'].mean():.1f} ± {df['target_words'].std():.1f}")
    print(f"  Compression ratio: {(df['target_words']/df['input_words']).mean():.2f}\n")
    
    return df

# ============================================================================
# 5. CHUNKING SEMÁNTICO
# ============================================================================

def setup_chunking(tokenizer):
    """Configura el text splitter semántico"""
    print("="*80)
    print("✂️  CONFIGURACIÓN CHUNKING SEMÁNTICO")
    print("="*80)
    
    # Función para contar tokens
    def length_function(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))
    
    # Configurar splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=length_function,
        separators=config.SEPARATORS,
        is_separator_regex=False
    )
    
    print(f"✅ Text Splitter configurado")
    print(f"  - Chunk size: {config.CHUNK_SIZE} tokens")
    print(f"  - Overlap: {config.CHUNK_OVERLAP} tokens")
    print(f"  - Separators: {config.SEPARATORS}")
    print()
    
    return text_splitter, length_function

def aplicar_chunking(texto: str, target: str, text_splitter, length_function) -> List[Tuple[str, str]]:
    """
    Aplica chunking semántico a un texto si excede 512 tokens.
    
    Returns:
        Lista de tuplas (chunk, target)
    """
    # Agregar task prefix
    texto_con_prefix = config.TASK_PREFIX + texto
    num_tokens = length_function(texto_con_prefix)
    
    # Si cabe en 512 tokens, no hacer chunking
    if num_tokens <= config.MAX_INPUT_LENGTH:
        return [(texto, target)]
    
    # Aplicar chunking
    chunks = text_splitter.split_text(texto)
    
    # Cada chunk genera el MISMO target
    return [(chunk, target) for chunk in chunks]

def preparar_dataset_con_chunking(df: pd.DataFrame, text_splitter, length_function):
    """Prepara dataset aplicando chunking donde sea necesario"""
    print("="*80)
    print("🔄 APLICANDO CHUNKING")
    print("="*80)
    
    all_examples = []
    textos_con_chunking = 0
    total_chunks = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesando textos"):
        chunks = aplicar_chunking(row['input'], row['target'], text_splitter, length_function)
        all_examples.extend(chunks)
        
        if len(chunks) > 1:
            textos_con_chunking += 1
            total_chunks += len(chunks)
    
    # Estadísticas
    print(f"\n📊 Estadísticas de Chunking:")
    print(f"  - Textos originales: {len(df):,}")
    print(f"  - Ejemplos de entrenamiento: {len(all_examples):,}")
    print(f"  - Textos con chunking: {textos_con_chunking:,} ({textos_con_chunking/len(df)*100:.1f}%)")
    print(f"  - Chunks promedio (cuando se aplica): {total_chunks/textos_con_chunking if textos_con_chunking > 0 else 0:.2f}")
    print()
    
    # Convertir a DataFrame
    result_df = pd.DataFrame(all_examples, columns=['input', 'target'])
    return result_df

# ============================================================================
# 6. SPLIT Y TOKENIZACIÓN
# ============================================================================

def split_dataset(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1):
    """Split dataset en train/val/test"""
    print("="*80)
    print("✂️  SPLIT DATASET")
    print("="*80)
    
    # Identificar textos únicos para evitar leakage
    unique_inputs = df['input'].unique()
    np.random.seed(config.SEED)
    np.random.shuffle(unique_inputs)
    
    # Calcular índices de split
    n = len(unique_inputs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_inputs = set(unique_inputs[:train_end])
    val_inputs = set(unique_inputs[train_end:val_end])
    test_inputs = set(unique_inputs[val_end:])
    
    # Crear splits
    train_df = df[df['input'].isin(train_inputs)].copy()
    val_df = df[df['input'].isin(val_inputs)].copy()
    test_df = df[df['input'].isin(test_inputs)].copy()
    
    print(f"✅ Split completado:")
    print(f"  - Train: {len(train_df):,} ejemplos ({len(train_inputs):,} textos únicos)")
    print(f"  - Val: {len(val_df):,} ejemplos ({len(val_inputs):,} textos únicos)")
    print(f"  - Test: {len(test_df):,} ejemplos ({len(test_inputs):,} textos únicos)")
    print()
    
    return train_df, val_df, test_df

def tokenize_dataset(df: pd.DataFrame, tokenizer, desc="Tokenizando"):
    """Tokeniza un dataset"""
    
    def tokenize_function(examples):
        # Agregar task prefix
        inputs = [config.TASK_PREFIX + text for text in examples['input']]
        
        model_inputs = tokenizer(
            inputs,
            max_length=config.MAX_INPUT_LENGTH,
            truncation=True,
            padding=False  # Padding dinámico con DataCollator
        )
        
        # Tokenizar targets
        labels = tokenizer(
            examples['target'],
            max_length=config.MAX_TARGET_LENGTH,
            truncation=True,
            padding=False
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    # Convertir a Dataset de Hugging Face
    dataset = Dataset.from_pandas(df[['input', 'target']])
    
    # Tokenizar
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc=desc
    )
    
    return tokenized

# ============================================================================
# 7. MÉTRICAS
# ============================================================================

def compute_metrics_simple(eval_pred, tokenizer):
    """
    Función de métricas simple que solo usa ROUGE
    (evita problemas con evaluate library)
    """
    predictions, labels = eval_pred
    
    # Decodificar predicciones
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Convertir a numpy arrays si no lo son
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Reemplazar -100 en labels con pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Asegurarse de que todos los valores están en rango válido
    # Clip predictions a rango de vocab
    vocab_size = len(tokenizer)
    predictions = np.clip(predictions, 0, vocab_size - 1)
    labels = np.clip(labels, 0, vocab_size - 1)
    
    # Convertir a int32 para evitar overflow
    predictions = predictions.astype(np.int32)
    labels = labels.astype(np.int32)
    
    # Decodificar (con manejo de errores)
    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        print(f"⚠️  Error al decodificar: {e}")
        # Retornar métricas dummy en caso de error
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
        }
    
    # Calcular ROUGE
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        # Saltar textos vacíos
        if not pred.strip() or not label.strip():
            continue
        scores = rouge_scorer_obj.score(label, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    # Evitar división por cero
    if len(rouge1_scores) == 0:
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
        }
    
    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores),
    }

# ============================================================================
# 8. ENTRENAMIENTO
# ============================================================================

def train_model(train_dataset, val_dataset, tokenizer, model):
    """Entrena el modelo T5"""
    print("="*80)
    print("🚀 ENTRENAMIENTO")
    print("="*80)
    
    # Configurar argumentos de entrenamiento
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',
        num_train_epochs=config.NUM_EPOCHS,
        
        # Batch sizes (A100 optimizado)
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM_STEPS,
        
        # Optimización
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        
        # Evaluación (nombres actualizados en transformers 4.x)
        eval_strategy='steps',  # antes: evaluation_strategy
        eval_steps=config.EVAL_STEPS,
        save_strategy='steps',
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model='rougeL',
        greater_is_better=True,
        
        # Generación
        predict_with_generate=True,
        generation_max_length=config.MAX_TARGET_LENGTH,
        generation_num_beams=config.NUM_BEAMS,
        
        # Eficiencia (A100)
        bf16=True,  # BF16 para A100
        dataloader_num_workers=2,
        
        # Logging
        logging_dir='./logs',
        logging_steps=50,
        report_to=['tensorboard'],
        
        # Otros
        seed=config.SEED,
        push_to_hub=False,
    )
    
    print(f"✅ Training arguments configurados")
    print(f"  - Épocas: {config.NUM_EPOCHS}")
    print(f"  - Batch size efectivo: {config.BATCH_SIZE * config.GRAD_ACCUM_STEPS}")
    print(f"  - Learning rate: {config.LEARNING_RATE}")
    print(f"  - BF16: True (A100)")
    print(f"\n💾 CHECKPOINTS AUTOMÁTICOS:")
    print(f"  - Guardando cada {config.SAVE_STEPS} steps en: ./results/")
    print(f"  - Se mantienen los últimos {config.SAVE_TOTAL_LIMIT} checkpoints")
    print(f"  - Si se interrumpe, puedes reanudar desde el último checkpoint")
    print()
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors='pt'
    )
    
    # Crear función de métricas con tokenizer
    def compute_metrics_fn(eval_pred):
        return compute_metrics_simple(eval_pred, tokenizer)
    
    # Inicializar Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    
    print(f"✅ Seq2SeqTrainer inicializado")
    print(f"  - Train examples: {len(train_dataset):,}")
    print(f"  - Eval examples: {len(val_dataset):,}")
    print()
    
    # Entrenar
    print("🚀 Iniciando entrenamiento...")
    print("⏱️  Tiempo estimado: ~45-90 minutos en A100 GPU")
    print("💡 Con batch size 16 y BF16 precision")
    print("\n⚠️  IMPORTANTE: El modelo se guardará AUTOMÁTICAMENTE:")
    print("   1. Cada 200 steps → checkpoints en ./results/checkpoint-XXX/")
    print("   2. Al finalizar → modelo completo en Drive")
    print("   3. Antes de evaluar → copia de seguridad adicional\n")
    
    try:
        train_result = trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido manualmente")
        print("💾 Los checkpoints están guardados en: ./results/")
        print("   Para continuar desde el último checkpoint, ejecuta:")
        print("   trainer.train(resume_from_checkpoint=True)")
        raise
    except Exception as e:
        print(f"\n❌ ERROR durante entrenamiento: {e}")
        print("💾 Los checkpoints están guardados en: ./results/")
        raise
    
    print("\n✅ Entrenamiento completado!")
    print(f"  - Training loss: {train_result.training_loss:.4f}")
    print(f"  - Total steps: {train_result.global_step}")
    print(f"  - Total time: {train_result.metrics['train_runtime']:.2f}s ({train_result.metrics['train_runtime']/60:.1f} min)")
    print(f"  - Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
    print()
    
    # Guardar historial
    training_history = {
        'train_loss': train_result.training_loss,
        'train_runtime': train_result.metrics['train_runtime'],
        'train_samples_per_second': train_result.metrics['train_samples_per_second'],
        'global_step': train_result.global_step,
    }
    
    with open(f'{config.RESULTS_DIR}/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"💾 Historial guardado en: {config.RESULTS_DIR}/training_history.json\n")
    
    return trainer, training_history

# ============================================================================
# 9. EXPORTAR MODELO
# ============================================================================

def export_model(trainer, tokenizer):
    """Exporta modelo a Google Drive"""
    print("="*80)
    print("💾 EXPORTANDO MODELO A DRIVE")
    print("="*80)
    
    try:
        # Guardar modelo
        print("📝 Guardando modelo...")
        trainer.model.save_pretrained(config.MODEL_DIR)
        tokenizer.save_pretrained(config.MODEL_DIR)
        
        print(f"✅ Modelo guardado en: {config.MODEL_DIR}")
        print(f"  - model.safetensors / pytorch_model.bin")
        print(f"  - config.json")
        print(f"  - tokenizer files")
        
        # También copiar checkpoints a Drive como respaldo adicional
        checkpoints_dir = f'{config.DRIVE_BASE}/checkpoints_backup'
        print(f"\n📦 Copiando checkpoints a Drive como respaldo...")
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Copiar el último checkpoint
        if os.path.exists('./results'):
            checkpoints = [d for d in os.listdir('./results') if d.startswith('checkpoint-')]
            if checkpoints:
                # Ordenar por número de checkpoint
                checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))
                last_checkpoint = checkpoints_sorted[-1]
                last_checkpoint_path = f'./results/{last_checkpoint}'
                
                print(f"   Copiando {last_checkpoint} a Drive...")
                os.system(f"cp -r {last_checkpoint_path} {checkpoints_dir}/")
                print(f"   ✅ Checkpoint respaldado en: {checkpoints_dir}/{last_checkpoint}")
        
        print()
        
    except Exception as e:
        print(f"⚠️  Error al guardar en Drive: {e}")
        print(f"   Intentando guardar localmente en ./results/")
        
        # Fallback: guardar en local
        local_dir = './results/final_model'
        os.makedirs(local_dir, exist_ok=True)
        trainer.model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)
        print(f"   ✅ Modelo guardado localmente en: {local_dir}")
        print(f"   Puedes copiarlo a Drive manualmente después")
        print()

# ============================================================================
# 11. FUNCIÓN DE INFERENCIA
# ============================================================================

def generar_pls(texto_tecnico: str, model, tokenizer, text_splitter, 
                max_length=256, num_beams=4, device='cuda'):
    """
    Genera PLS desde texto técnico.
    Maneja automáticamente chunking si texto excede 512 tokens.
    
    Args:
        texto_tecnico: Texto técnico de entrada
        model: Modelo T5 entrenado
        tokenizer: Tokenizer
        text_splitter: RecursiveCharacterTextSplitter configurado
        max_length: Longitud máxima de output
        num_beams: Número de beams para generación
        device: Dispositivo (cuda/cpu)
    
    Returns:
        str: PLS generado
    """
    # Verificar longitud
    texto_con_prefix = config.TASK_PREFIX + texto_tecnico
    tokens = tokenizer.encode(texto_con_prefix, add_special_tokens=False)
    
    # CASO 1: Texto cabe en 512 tokens
    if len(tokens) <= config.MAX_INPUT_LENGTH:
        inputs = tokenizer(
            texto_con_prefix,
            max_length=config.MAX_INPUT_LENGTH,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        pls = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return pls
    
    # CASO 2: Texto requiere chunking
    else:
        chunks = text_splitter.split_text(texto_tecnico)
        
        chunk_outputs = []
        for chunk in chunks:
            chunk_con_prefix = config.TASK_PREFIX + chunk
            inputs = tokenizer(
                chunk_con_prefix,
                max_length=config.MAX_INPUT_LENGTH,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            chunk_pls = tokenizer.decode(outputs[0], skip_special_tokens=True)
            chunk_outputs.append(chunk_pls)
        
        # Fusionar outputs (tomar el primero o combinar)
        # Para simplificación, generalmente el primer chunk es más representativo
        pls_final = chunk_outputs[0]
        
        return pls_final

# ============================================================================
# 12. MAIN
# ============================================================================

def main():
    """Función principal"""
    print("\n" + "="*80)
    print("🚀 T5-BASE TRAINING FOR MEDICAL PLS")
    print("    Con Chunking Semántico LangChain")
    print("    Optimizado para GPU A100")
    print("="*80 + "\n")
    
    # Montar Google Drive (si es necesario)
    print("📂 Verificando Google Drive...")
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            drive.mount('/content/drive', force_remount=False)
            print("✅ Drive montado\n")
        else:
            print("✅ Drive ya está montado\n")
    except Exception as e:
        print(f"⚠️  No se pudo montar Drive: {e}")
        print("   Se intentará usar archivo local\n")
    
    # Configurar rutas y buscar archivo CSV
    setup_config()
    
    # 1. Cargar datos
    df = cargar_datos(config.CSV_PATH)
    
    # 2. Cargar modelo y tokenizer
    print("="*80)
    print("🤖 CARGANDO MODELO Y TOKENIZER")
    print("="*80)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)
    print(f"✅ {config.MODEL_NAME} cargado")
    print(f"  - Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 3. Setup chunking
    text_splitter, length_function = setup_chunking(tokenizer)
    
    # 4. Split ANTES de chunking (evitar data leakage)
    print("="*80)
    print("⚠️  IMPORTANTE: Split ANTES de chunking (evitar data leakage)")
    print("="*80)
    
    # Identificar textos únicos
    unique_df = df.drop_duplicates(subset=['input']).copy()
    
    # Split por textos únicos
    from sklearn.model_selection import train_test_split
    
    train_unique, temp = train_test_split(unique_df, test_size=0.2, random_state=config.SEED)
    val_unique, test_unique = train_test_split(temp, test_size=0.5, random_state=config.SEED)
    
    print(f"✅ Split por textos únicos:")
    print(f"  - Train: {len(train_unique):,}")
    print(f"  - Val: {len(val_unique):,}")
    print(f"  - Test: {len(test_unique):,}\n")
    
    # 5. Aplicar chunking a train y val (para entrenamiento)
    train_df = preparar_dataset_con_chunking(train_unique, text_splitter, length_function)
    val_df = preparar_dataset_con_chunking(val_unique, text_splitter, length_function)
    
    # IMPORTANTE: Guardar test set SIN chunking (documentos completos) para evaluación
    # El test set se guarda tal cual para evaluación a nivel documento
    test_df_complete = test_unique.copy()  # Test sin chunking
    test_df_complete = test_df_complete.rename(columns={
        'texto_original': 'input',
        'pls_generado': 'target'
    })
    
    # Guardar test set completo para evaluación posterior
    test_csv_path = f'{config.RESULTS_DIR}/test_set_complete.csv'
    test_df_complete[['input', 'target']].to_csv(test_csv_path, index=False)
    print(f"💾 Test set completo guardado: {test_csv_path}")
    print(f"   - {len(test_df_complete):,} documentos completos (sin chunking)")
    print(f"   - Para evaluación: usar evaluate_t5_base_pls_a100.py\n")
    
    # Para entrenamiento, también necesitamos test con chunking (opcional, para validación rápida)
    test_df = preparar_dataset_con_chunking(test_unique, text_splitter, length_function)
    
    # 6. Tokenizar
    print("="*80)
    print("🔤 TOKENIZACIÓN")
    print("="*80)
    train_dataset = tokenize_dataset(train_df, tokenizer, "Tokenizando train")
    val_dataset = tokenize_dataset(val_df, tokenizer, "Tokenizando val")
    test_dataset = tokenize_dataset(test_df, tokenizer, "Tokenizando test")
    print("✅ Tokenización completada\n")
    
    # 7. Entrenar
    trainer, training_history = train_model(train_dataset, val_dataset, tokenizer, model)
    
    # 8. GUARDAR MODELO
    print("="*80)
    print("💾 GUARDANDO MODELO")
    print("="*80)
    export_model(trainer, tokenizer)
    print("✅ Modelo guardado")
    print(f"📁 Ubicación: {config.MODEL_DIR}\n")
    
    # 9. Ejemplo de inferencia rápida
    print("="*80)
    print("🧪 EJEMPLO DE INFERENCIA RÁPIDA")
    print("="*80)
    ejemplo = test_df_complete.iloc[0]['input']
    pls_generado = generar_pls(ejemplo, trainer.model, tokenizer, text_splitter)
    print(f"INPUT: {ejemplo[:200]}...")
    print(f"\nGENERATED PLS: {pls_generado[:200]}...")
    print("\n" + "="*80)
    
    print("\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
    print(f"🤖 Modelo guardado en: {config.MODEL_DIR}")
    print(f"📊 Test set completo guardado en: {config.RESULTS_DIR}/test_set_complete.csv")
    print(f"\n📝 PRÓXIMOS PASOS:")
    print(f"   Para evaluar el modelo con todas las métricas, ejecuta:")
    print(f"   %run evaluate_t5_base_pls_a100.py")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()

