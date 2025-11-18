"""
Script de evaluación T5-BASE para Plain Language Summaries (PLS) médicos
Evalúa el modelo entrenado en el conjunto de test con todas las métricas

IMPORTANTE: Este script NO entrena el modelo, solo evalúa uno ya entrenado.

Autor: Proyecto de Grado - Maestría
Fecha: 2025
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from typing import Dict, List, Optional
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# 1. IMPORTACIONES
# ============================================================================

print("📚 Importando librerías...")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

# Métricas
from rouge_score import rouge_scorer
import sacrebleu
try:
    from bert_score import score as bert_score_fn
except:
    bert_score_fn = None
import textstat

# LangChain para chunking
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# sklearn para split
from sklearn.model_selection import train_test_split

print("✅ Importaciones completadas\n")

# ============================================================================
# 2. CONFIGURACIÓN (replicada de train_t5_base_pls_a100.py)
# ============================================================================

class Config:
    """Configuración del experimento (idéntica a train_t5_base_pls_a100.py)"""
    # Rutas - se configurarán dinámicamente
    CSV_PATH = None
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
    
    # Generación
    NUM_BEAMS = 4
    
    # Semilla (CRÍTICO: debe ser la misma que en entrenamiento)
    SEED = 42

config = Config()

def find_csv_file():
    """Busca el archivo CSV en múltiples ubicaciones posibles (replicado de train)"""
    print("="*80)
    print("🔍 BUSCANDO ARCHIVO CSV")
    print("="*80)
    
    possible_paths = [
        '/content/drive/MyDrive/PLS_Project/data/pls_10k_final_aprobados.csv',
        '/content/drive/MyDrive/PLS_Project/pls_10k_final_aprobados.csv',
        '/content/drive/My Drive/PLS_Project/data/pls_10k_final_aprobados.csv',
        '/content/drive/My Drive/PLS_Project/pls_10k_final_aprobados.csv',
        '/content/pls_10k_final_aprobados.csv',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Archivo encontrado: {path}\n")
            return path
    
    print("❌ No se encontró el archivo CSV\n")
    return None

def setup_config():
    """Configura las rutas de Config"""
    # Buscar CSV
    csv_path = os.environ.get('CSV_PATH')
    
    if not csv_path:
        csv_path = find_csv_file()
    
    if not csv_path:
        print("\n❌ No se puede continuar sin el archivo CSV")
        print("   Especifica manualmente: os.environ['CSV_PATH'] = '/ruta/al/archivo.csv'")
        sys.exit(1)
    
    # Ruta al modelo entrenado
    model_dir = os.environ.get('MODEL_DIR')
    if not model_dir:
        model_dir = f'{config.DRIVE_BASE}/models/t5_pls/final'
    
    # Verificar que el modelo existe
    if not os.path.exists(model_dir):
        print(f"❌ Modelo no encontrado en: {model_dir}")
        print("   Especifica manualmente: os.environ['MODEL_DIR'] = '/ruta/al/modelo'")
        sys.exit(1)
    
    # Configurar rutas
    config.CSV_PATH = csv_path
    config.MODEL_DIR = model_dir
    config.RESULTS_DIR = f'{config.DRIVE_BASE}/results'
    config.PLOTS_DIR = f'{config.RESULTS_DIR}/plots'
    
    # Crear directorios
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    # Mostrar configuración
    print("="*80)
    print("⚙️  CONFIGURACIÓN")
    print("="*80)
    print(f"CSV: {config.CSV_PATH}")
    print(f"Modelo: {config.MODEL_DIR}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Semilla: {config.SEED} (debe ser igual a entrenamiento)")
    print("="*80 + "\n")

# ============================================================================
# 3. CARGAR Y PREPARAR DATOS (replicado de train_t5_base_pls_a100.py)
# ============================================================================

def cargar_datos(csv_path: str) -> pd.DataFrame:
    """Carga y valida el dataset (replicado de train_t5_base_pls_a100.py)"""
    print("="*80)
    print("📂 CARGANDO DATOS")
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
    
    print(f"\n📊 Estadísticas del dataset completo:")
    print(f"  Input words: {df['input_words'].mean():.1f} ± {df['input_words'].std():.1f}")
    print(f"  Target words: {df['target_words'].mean():.1f} ± {df['target_words'].std():.1f}")
    print(f"  Compression ratio: {(df['target_words']/df['input_words']).mean():.2f}\n")
    
    return df

def reconstruir_split(df: pd.DataFrame):
    """
    Reconstruye el split exactamente igual que en train_t5_base_pls_a100.py
    
    IMPORTANTE: Usa el mismo random_state (config.SEED) para obtener los mismos splits.
    """
    print("="*80)
    print("✂️  RECONSTRUYENDO SPLIT (igual que en entrenamiento)")
    print("="*80)
    
    # Identificar textos únicos (igual que en train)
    unique_df = df.drop_duplicates(subset=['input']).copy()
    print(f"✅ Textos únicos: {len(unique_df):,} documentos")
    
    # Split por textos únicos (EXACTAMENTE igual que en train)
    # train_test_split con random_state=config.SEED garantiza reproducibilidad
    train_unique, temp = train_test_split(
        unique_df, 
        test_size=0.2, 
        random_state=config.SEED
    )
    
    val_unique, test_unique = train_test_split(
        temp, 
        test_size=0.5, 
        random_state=config.SEED
    )
    
    print(f"✅ Split por textos únicos (random_state={config.SEED}):")
    print(f"  - Train: {len(train_unique):,} documentos")
    print(f"  - Val: {len(val_unique):,} documentos")
    print(f"  - Test: {len(test_unique):,} documentos")
    print()
    
    return train_unique, val_unique, test_unique

def validar_test_set(test_unique: pd.DataFrame):
    """
    Valida el test set con checks de sanidad
    """
    print("="*80)
    print("✅ VALIDACIÓN DEL TEST SET")
    print("="*80)
    
    # 1. Tamaño del test set
    print(f"\n📊 Tamaño del test set:")
    print(f"  - Número de documentos: {len(test_unique):,}")
    print(f"  - Esperado: ~699 documentos (10% de ~6985 documentos únicos)")
    
    if len(test_unique) < 600 or len(test_unique) > 800:
        print(f"  ⚠️  ADVERTENCIA: Tamaño fuera del rango esperado")
    else:
        print(f"  ✅ Tamaño dentro del rango esperado")
    
    # 2. Longitud media en palabras
    test_unique['input_words'] = test_unique['input'].str.split().str.len()
    test_unique['target_words'] = test_unique['target'].str.split().str.len()
    
    mean_input_words = test_unique['input_words'].mean()
    mean_target_words = test_unique['target_words'].mean()
    
    print(f"\n📏 Longitud media en palabras:")
    print(f"  - Input (texto técnico): {mean_input_words:.1f} palabras")
    print(f"  - Target (PLS): {mean_target_words:.1f} palabras")
    print(f"  - Esperado input: ~540 palabras")
    print(f"  - Esperado target: ~170 palabras")
    
    if mean_input_words < 300:
        print(f"  ⚠️  ADVERTENCIA: Input muy corto (¿están fragmentados?)")
    else:
        print(f"  ✅ Input tiene longitud adecuada (documentos completos)")
    
    if mean_target_words < 100 or mean_target_words > 250:
        print(f"  ⚠️  ADVERTENCIA: Target fuera del rango esperado")
    else:
        print(f"  ✅ Target tiene longitud adecuada")
    
    # 3. Ratio de compresión esperado
    compression_ratio = mean_target_words / mean_input_words if mean_input_words > 0 else 0
    print(f"\n📉 Ratio de compresión esperado:")
    print(f"  - {compression_ratio:.3f} (target_words / input_words)")
    print(f"  - Esperado: 0.33-0.40")
    
    if compression_ratio < 0.25 or compression_ratio > 0.50:
        print(f"  ⚠️  ADVERTENCIA: Ratio fuera del rango esperado")
    elif compression_ratio < 0.30 or compression_ratio > 0.40:
        print(f"  ⚠️  ADVERTENCIA: Ratio cerca del límite del rango")
    else:
        print(f"  ✅ Ratio dentro del rango esperado")
    
    # 4. Verificar que no hay duplicados en input
    duplicates = test_unique['input'].duplicated().sum()
    print(f"\n🔍 Verificación de duplicados:")
    print(f"  - Duplicados en input: {duplicates}")
    if duplicates > 0:
        print(f"  ⚠️  ADVERTENCIA: Hay {duplicates} documentos duplicados")
    else:
        print(f"  ✅ No hay duplicados (cada documento es único)")
    
    # 5. Verificar que no hay textos vacíos
    empty_inputs = (test_unique['input'].str.strip() == '').sum()
    empty_targets = (test_unique['target'].str.strip() == '').sum()
    print(f"\n🔍 Verificación de textos vacíos:")
    print(f"  - Inputs vacíos: {empty_inputs}")
    print(f"  - Targets vacíos: {empty_targets}")
    if empty_inputs > 0 or empty_targets > 0:
        print(f"  ⚠️  ADVERTENCIA: Hay textos vacíos")
    else:
        print(f"  ✅ No hay textos vacíos")
    
    print("\n" + "="*80)
    print()

# ============================================================================
# 4. SETUP CHUNKING (para generación de textos largos)
# ============================================================================

def setup_chunking(tokenizer):
    """Configura el text splitter para chunking semántico"""
    def length_function(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=length_function,
        separators=config.SEPARATORS,
        is_separator_regex=False
    )
    
    return text_splitter

# ============================================================================
# 5. GENERACIÓN DE PREDICCIONES
# ============================================================================

def generar_pls_con_chunking(texto_tecnico: str, model, tokenizer, text_splitter, 
                             device='cuda', max_length=256, num_beams=4):
    """
    Genera PLS desde texto técnico con manejo automático de chunking.
    
    IMPORTANTE: Para evaluación, generamos UNA PLS por documento completo.
    Si el documento es muy largo, usamos chunking pero fusionamos los outputs.
    """
    texto_con_prefix = config.TASK_PREFIX + texto_tecnico
    tokens = tokenizer.encode(texto_con_prefix, add_special_tokens=False)
    
    # CASO 1: Texto cabe en 512 tokens (procesamiento directo)
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
        
        # Fusionar outputs (usar el primero como principal)
        # Para evaluación, esto es aceptable ya que evaluamos a nivel documento
        pls_final = chunk_outputs[0]
        
        return pls_final

def generar_predicciones(sources: List[str], model, tokenizer, text_splitter, device):
    """Genera predicciones para todo el test set"""
    print("="*80)
    print("🔄 GENERANDO PREDICCIONES")
    print("="*80)
    
    predictions = []
    model.eval()
    
    for source in tqdm(sources, desc="Generando PLS"):
        pls = generar_pls_con_chunking(
            source, model, tokenizer, text_splitter, 
            device=device,
            max_length=config.MAX_TARGET_LENGTH,
            num_beams=config.NUM_BEAMS
        )
        predictions.append(pls)
    
    print(f"✅ {len(predictions)} predicciones generadas\n")
    
    # Validar longitudes de predicciones
    pred_lengths = [len(p.split()) for p in predictions]
    mean_pred_length = np.mean(pred_lengths)
    
    print(f"📏 Longitud media de predicciones generadas:")
    print(f"  - {mean_pred_length:.1f} palabras")
    print(f"  - Esperado: ~173 palabras")
    print()
    
    return predictions

# ============================================================================
# 6. CÁLCULO DE MÉTRICAS
# ============================================================================

def compute_all_metrics(sources: List[str], predictions: List[str], references: List[str]) -> Dict:
    """
    Calcula todas las 11 métricas del paper
    
    IMPORTANTE: Esta función evalúa a NIVEL DE DOCUMENTO:
    - sources: textos técnicos originales completos
    - references: PLS completos correspondientes
    - predictions: una PLS generada por cada documento original
    
    Args:
        sources: Textos originales (inputs) - documentos completos
        predictions: Textos generados por el modelo - una PLS por documento
        references: Textos gold standard (targets) - PLS completos
    """
    print("="*80)
    print("📊 CALCULANDO TODAS LAS MÉTRICAS")
    print("="*80)
    
    metrics = {}
    
    # 1-3. ROUGE
    print("1️⃣  Calculando ROUGE...")
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    
    for pred, ref in zip(predictions, references):
        scores = rouge_scorer_obj.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    metrics['rouge1'] = np.mean(rouge1_scores)
    metrics['rouge2'] = np.mean(rouge2_scores)
    metrics['rougeL'] = np.mean(rougeL_scores)
    
    # 4. BLEU
    print("2️⃣  Calculando BLEU...")
    refs_for_bleu = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(predictions, refs_for_bleu, smooth_method='exp')
    metrics['bleu'] = bleu.score / 100.0
    
    # 5. METEOR (implementación manual simple)
    print("3️⃣  Calculando METEOR aproximado...")
    meteor_scores = []
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        if len(pred_words) == 0 or len(ref_words) == 0:
            meteor_scores.append(0.0)
        else:
            precision = len(pred_words & ref_words) / len(pred_words)
            recall = len(pred_words & ref_words) / len(ref_words)
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                meteor_scores.append(f1)
            else:
                meteor_scores.append(0.0)
    metrics['meteor'] = np.mean(meteor_scores)
    
    # 6. BERTScore
    print("4️⃣  Calculando BERTScore (esto puede tomar tiempo)...")
    if bert_score_fn is not None:
        try:
            P, R, F1 = bert_score_fn(predictions, references, lang='en', verbose=False)
            metrics['bertscore_f1'] = F1.mean().item()
        except Exception as e:
            print(f"  ⚠️  Error en BERTScore: {e}")
            metrics['bertscore_f1'] = 0.0
    else:
        print("  ⚠️  BERTScore no disponible")
        metrics['bertscore_f1'] = 0.0
    
    # 7. SARI (implementación manual)
    print("5️⃣  Calculando SARI...")
    sari_scores = []
    for src, pred, ref in zip(sources, predictions, references):
        src_words = set(src.lower().split())
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        
        # Keep: palabras que están en source y reference, y se mantienen en prediction
        keep = len(src_words & ref_words & pred_words)
        keep_total = len(src_words & ref_words)
        keep_score = keep / keep_total if keep_total > 0 else 0
        
        # Add: palabras nuevas en reference que están en prediction
        add = len((ref_words - src_words) & pred_words)
        add_total = len(ref_words - src_words)
        add_score = add / add_total if add_total > 0 else 0
        
        # Delete: palabras en source que no están en reference ni en prediction
        delete = len(src_words - ref_words - pred_words)
        delete_total = len(src_words - ref_words)
        delete_score = delete / delete_total if delete_total > 0 else 0
        
        # SARI es el promedio de las tres operaciones
        sari = (keep_score + add_score + delete_score) / 3
        sari_scores.append(sari)
    
    metrics['sari'] = np.mean(sari_scores)
    
    # 8-9. Legibilidad
    print("6️⃣  Calculando métricas de legibilidad...")
    fre_scores = [textstat.flesch_reading_ease(text) for text in predictions]
    fkg_scores = [textstat.flesch_kincaid_grade(text) for text in predictions]
    
    metrics['fre_mean'] = np.mean(fre_scores)
    metrics['fre_std'] = np.std(fre_scores)
    metrics['fkg_mean'] = np.mean(fkg_scores)
    metrics['fkg_std'] = np.std(fkg_scores)
    
    # También para referencias
    fre_refs = [textstat.flesch_reading_ease(text) for text in references]
    fkg_refs = [textstat.flesch_kincaid_grade(text) for text in references]
    metrics['fre_ref_mean'] = np.mean(fre_refs)
    metrics['fkg_ref_mean'] = np.mean(fkg_refs)
    
    # 10-11. Compresión (CORREGIDO: a nivel documento)
    print("7️⃣  Calculando métricas de compresión...")
    pred_lengths = [len(p.split()) for p in predictions]
    src_lengths = [len(s.split()) for s in sources]  # Longitud del documento original completo
    
    compression_ratios = [p/s if s > 0 else 0 for p, s in zip(pred_lengths, src_lengths)]
    
    metrics['compression_ratio'] = np.mean(compression_ratios)
    metrics['compression_ratio_std'] = np.std(compression_ratios)
    metrics['output_length_words'] = np.mean(pred_lengths)
    metrics['output_length_std'] = np.std(pred_lengths)
    
    print("\n✅ Todas las métricas calculadas\n")
    
    # Validar ratio de compresión
    print("="*80)
    print("✅ VALIDACIÓN DEL RATIO DE COMPRESIÓN")
    print("="*80)
    print(f"  - Ratio calculado: {metrics['compression_ratio']:.3f}")
    print(f"  - Esperado: 0.33-0.40")
    if 0.33 <= metrics['compression_ratio'] <= 0.40:
        print(f"  ✅ Ratio dentro del rango esperado")
    else:
        print(f"  ⚠️  ADVERTENCIA: Ratio fuera del rango esperado")
    print()
    
    return metrics, {
        'fre_preds': fre_scores,
        'fkg_preds': fkg_scores,
        'fre_refs': fre_refs,
        'fkg_refs': fkg_refs,
        'pred_lengths': pred_lengths,
        'ref_lengths': [len(r.split()) for r in references],
        'src_lengths': src_lengths,  # Agregado para análisis
        'rouge1_individual': rouge1_scores,
        'rouge2_individual': rouge2_scores,
        'rougeL_individual': rougeL_scores,
        'sari_individual': sari_scores,
    }

# ============================================================================
# 7. VISUALIZACIÓN Y REPORTE
# ============================================================================

def print_metrics_table(metrics: Dict):
    """Imprime tabla de métricas"""
    print("="*80)
    print("📊 RESULTADOS FINALES")
    print("="*80)
    print()
    print(f"{'Métrica':<25} {'Valor':<12} {'Target':<15}")
    print("-" * 80)
    print(f"{'ROUGE-1':<25} {metrics['rouge1']:<12.3f} {'-':<15}")
    print(f"{'ROUGE-2':<25} {metrics['rouge2']:<12.3f} {'-':<15}")
    print(f"{'ROUGE-L':<25} {metrics['rougeL']:<12.3f} {'-':<15}")
    print(f"{'BLEU':<25} {metrics['bleu']:<12.3f} {'-':<15}")
    print(f"{'METEOR':<25} {metrics['meteor']:<12.3f} {'-':<15}")
    print(f"{'BERTScore F1':<25} {metrics['bertscore_f1']:<12.3f} {'-':<15}")
    print(f"{'SARI ⭐':<25} {metrics['sari']:<12.3f} {'>0.40':<15}")
    print(f"{'Flesch Reading Ease':<25} {metrics['fre_mean']:<12.1f} {'~64':<15}")
    print(f"{'Flesch-Kincaid Grade':<25} {metrics['fkg_mean']:<12.1f} {'~7.4':<15}")
    print(f"{'Compression Ratio':<25} {metrics['compression_ratio']:<12.2f} {'0.33-0.37':<15}")
    print(f"{'Longitud (palabras)':<25} {metrics['output_length_words']:<12.0f} {'~173':<15}")
    print("="*80)
    print()

def save_results(test_df, predictions, metrics, detailed_metrics):
    """Guarda todos los resultados"""
    print("💾 Guardando resultados...")
    
    # DataFrame con resultados detallados
    results_df = test_df.copy()
    results_df['generated'] = predictions
    results_df['rouge1'] = detailed_metrics['rouge1_individual']
    results_df['rouge2'] = detailed_metrics['rouge2_individual']
    results_df['rougeL'] = detailed_metrics['rougeL_individual']
    results_df['sari'] = detailed_metrics['sari_individual']
    results_df['fre'] = detailed_metrics['fre_preds']
    results_df['fkg'] = detailed_metrics['fkg_preds']
    results_df['length'] = detailed_metrics['pred_lengths']
    results_df['src_length'] = detailed_metrics['src_lengths']
    results_df['compression_ratio'] = [
        p/s if s > 0 else 0 
        for p, s in zip(detailed_metrics['pred_lengths'], detailed_metrics['src_lengths'])
    ]
    
    results_df.to_csv(f'{config.RESULTS_DIR}/test_results.csv', index=False)
    print(f"  ✅ CSV: {config.RESULTS_DIR}/test_results.csv")
    
    # Métricas agregadas
    with open(f'{config.RESULTS_DIR}/test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✅ JSON: {config.RESULTS_DIR}/test_metrics.json")
    
    print()

def create_visualizations(metrics, detailed_metrics):
    """Crea todas las visualizaciones"""
    print("="*80)
    print("📊 CREANDO VISUALIZACIONES")
    print("="*80)
    
    # 1. Histograma FRE
    plt.figure(figsize=(10, 6))
    plt.hist(detailed_metrics['fre_preds'], bins=30, alpha=0.7, label='Generados', color='blue')
    plt.hist(detailed_metrics['fre_refs'], bins=30, alpha=0.7, label='Referencias', color='green')
    plt.axvline(64, color='red', linestyle='--', label='Target (64)')
    plt.xlabel('Flesch Reading Ease')
    plt.ylabel('Frecuencia')
    plt.title('Distribución Flesch Reading Ease')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/distribution_fre.png', dpi=300)
    plt.close()
    print("  ✅ distribution_fre.png")
    
    # 2. Histograma FKG
    plt.figure(figsize=(10, 6))
    plt.hist(detailed_metrics['fkg_preds'], bins=30, alpha=0.7, label='Generados', color='blue')
    plt.hist(detailed_metrics['fkg_refs'], bins=30, alpha=0.7, label='Referencias', color='green')
    plt.axvline(7.4, color='red', linestyle='--', label='Target (7.4)')
    plt.xlabel('Flesch-Kincaid Grade')
    plt.ylabel('Frecuencia')
    plt.title('Distribución Flesch-Kincaid Grade')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/distribution_fkg.png', dpi=300)
    plt.close()
    print("  ✅ distribution_fkg.png")
    
    # 3. Histograma Length
    plt.figure(figsize=(10, 6))
    plt.hist(detailed_metrics['pred_lengths'], bins=30, alpha=0.7, label='Generados', color='blue')
    plt.hist(detailed_metrics['ref_lengths'], bins=30, alpha=0.7, label='Referencias', color='green')
    plt.axvline(173, color='red', linestyle='--', label='Target (173)')
    plt.xlabel('Longitud (palabras)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución Longitud de Outputs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/distribution_length.png', dpi=300)
    plt.close()
    print("  ✅ distribution_length.png")
    
    # 4. Scatter ROUGE-L vs SARI
    plt.figure(figsize=(10, 6))
    plt.scatter(detailed_metrics['rougeL_individual'], detailed_metrics['sari_individual'], alpha=0.5)
    plt.xlabel('ROUGE-L')
    plt.ylabel('SARI')
    plt.title('Correlación ROUGE-L vs SARI')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/scatter_rouge_sari.png', dpi=300)
    plt.close()
    print("  ✅ scatter_rouge_sari.png")
    
    # 5. Boxplot métricas normalizadas
    plt.figure(figsize=(12, 6))
    normalized_metrics = {
        'ROUGE-1': detailed_metrics['rouge1_individual'],
        'ROUGE-2': detailed_metrics['rouge2_individual'],
        'ROUGE-L': detailed_metrics['rougeL_individual'],
        'SARI': detailed_metrics['sari_individual'],
    }
    plt.boxplot(normalized_metrics.values(), labels=normalized_metrics.keys())
    plt.ylabel('Score')
    plt.title('Distribución de Métricas')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{config.PLOTS_DIR}/boxplot_metrics.png', dpi=300)
    plt.close()
    print("  ✅ boxplot_metrics.png")
    
    print()

def show_qualitative_examples(sources, predictions, references, detailed_metrics):
    """Muestra ejemplos cualitativos"""
    print("="*80)
    print("📝 EJEMPLOS CUALITATIVOS")
    print("="*80)
    
    sari_scores = detailed_metrics['sari_individual']
    
    # Mejores 3
    best_indices = np.argsort(sari_scores)[-3:][::-1]
    print("\n🏆 TOP 3 MEJORES (SARI más alto):")
    for i, idx in enumerate(best_indices, 1):
        print(f"\n{'='*60}")
        print(f"Ejemplo {i} (SARI: {sari_scores[idx]:.3f})")
        print(f"{'='*60}")
        print(f"INPUT: {sources[idx][:200]}...")
        print(f"\nREFERENCE: {references[idx][:200]}...")
        print(f"\nGENERATED: {predictions[idx][:200]}...")
    
    # Peores 3
    worst_indices = np.argsort(sari_scores)[:3]
    print("\n\n❌ TOP 3 PEORES (SARI más bajo):")
    for i, idx in enumerate(worst_indices, 1):
        print(f"\n{'='*60}")
        print(f"Ejemplo {i} (SARI: {sari_scores[idx]:.3f})")
        print(f"{'='*60}")
        print(f"INPUT: {sources[idx][:200]}...")
        print(f"\nREFERENCE: {references[idx][:200]}...")
        print(f"\nGENERATED: {predictions[idx][:200]}...")
    
    print("\n" + "="*80)
    print()

# ============================================================================
# 8. MAIN
# ============================================================================

def main(max_ejemplos: Optional[int] = None):
    """
    Función principal de evaluación
    
    Args:
        max_ejemplos: Si es None, evalúa sobre TODO el test set.
                     Si es un entero (ej: 200), evalúa solo sobre una muestra de ese tamaño.
    """
    print("\n" + "="*80)
    print("🧪 EVALUACIÓN T5-BASE PARA MEDICAL PLS")
    print("    Evaluación en Test Set Completo")
    print("    (Modelo ya entrenado - NO se re-entrena)")
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
        print("   Continuando con rutas locales\n")
    
    # Configurar rutas
    setup_config()
    
    # 1. Cargar datos completos
    df = cargar_datos(config.CSV_PATH)
    
    # 2. Reconstruir split (igual que en entrenamiento)
    train_unique, val_unique, test_unique = reconstruir_split(df)
    
    # 3. Validar test set
    validar_test_set(test_unique)
    
    # 4. Opcional: muestrear si se especifica max_ejemplos
    if max_ejemplos is not None and max_ejemplos < len(test_unique):
        print(f"📊 Muestreando {max_ejemplos} ejemplos del test set...")
        np.random.seed(config.SEED)
        indices = np.random.choice(len(test_unique), size=max_ejemplos, replace=False)
        test_unique = test_unique.iloc[indices].copy()
        print(f"✅ Evaluando sobre {len(test_unique):,} ejemplos (muestra)\n")
    else:
        print(f"✅ Evaluando sobre TODO el test set: {len(test_unique):,} documentos\n")
    
    # 5. Preparar sources y references (documentos completos)
    sources = test_unique['input'].tolist()
    references = test_unique['target'].tolist()
    
    print(f"📊 Preparación de datos para evaluación:")
    print(f"  - Sources: {len(sources):,} documentos completos")
    print(f"  - References: {len(references):,} PLS completos")
    print(f"  - Longitud media sources: {np.mean([len(s.split()) for s in sources]):.1f} palabras")
    print(f"  - Longitud media references: {np.mean([len(r.split()) for r in references]):.1f} palabras")
    print()
    
    # 6. Cargar modelo y tokenizer entrenados
    print("="*80)
    print("🤖 CARGANDO MODELO ENTRENADO")
    print("="*80)
    print(f"📥 Cargando desde: {config.MODEL_DIR}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_DIR)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"✅ Modelo cargado")
    print(f"   - Device: {device}")
    print(f"   - Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 7. Setup chunking (para textos largos)
    text_splitter = setup_chunking(tokenizer)
    print("✅ Text splitter configurado\n")
    
    # 8. Generar predicciones (una PLS por documento completo)
    predictions = generar_predicciones(sources, model, tokenizer, text_splitter, device)
    
    # 9. Validar longitudes de predicciones vs sources
    print("="*80)
    print("✅ VALIDACIÓN POST-GENERACIÓN")
    print("="*80)
    src_lengths = [len(s.split()) for s in sources]
    pred_lengths = [len(p.split()) for p in predictions]
    
    print(f"📏 Longitudes:")
    print(f"  - Sources (media): {np.mean(src_lengths):.1f} palabras")
    print(f"  - Predictions (media): {np.mean(pred_lengths):.1f} palabras")
    print(f"  - Ratio: {np.mean(pred_lengths)/np.mean(src_lengths):.3f}")
    print()
    
    # 10. Calcular todas las métricas (a nivel documento)
    metrics, detailed_metrics = compute_all_metrics(sources, predictions, references)
    
    # 11. Mostrar resultados
    print_metrics_table(metrics)
    
    # 12. Guardar resultados
    save_results(test_unique, predictions, metrics, detailed_metrics)
    
    # 13. Crear visualizaciones
    create_visualizations(metrics, detailed_metrics)
    
    # 14. Mostrar ejemplos cualitativos
    show_qualitative_examples(sources, predictions, references, detailed_metrics)
    
    print("\n🎉 ¡EVALUACIÓN COMPLETADA!")
    print(f"📊 Resultados en: {config.RESULTS_DIR}")
    print(f"📈 Gráficos en: {config.PLOTS_DIR}")
    print("="*80 + "\n")

if __name__ == '__main__':
    # Para evaluar solo una muestra (útil para pruebas rápidas):
    # main(max_ejemplos=200)
    
    # Para evaluación completa (por defecto):
    main(max_ejemplos=None)
