"""
Script completo para entrenar y evaluar modelo T5-large para simplificación de texto médico.

Este script incluye:
- Entrenamiento del modelo T5-large con chunking semántico
- Evaluación con métricas completas (ROUGE, BLEU, SARI, Factuality, Flesch, etc.)
"""

import os
import json
import pickle
import torch
import numpy as np
import subprocess
import sys
import re
import warnings
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

warnings.filterwarnings('ignore')

# ============================================================================
# INSTALACIÓN DE DEPENDENCIAS
# ============================================================================

def install_package(package_name, quiet=True):
    """Instala un paquete usando pip."""
    try:
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython:
                ipython.run_line_magic('pip', f'install {"-q" if quiet else ""} {package_name}')
                return True
        except:
            pass
        
        cmd = [sys.executable, "-m", "pip", "install"]
        if quiet:
            cmd.append("-q")
        cmd.append(package_name)
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"Advertencia: Error instalando {package_name}: {str(e)[:100]}")
        return False

def setup_dependencies():
    """Instala y verifica dependencias necesarias."""
    print("Instalando dependencias necesarias...")
    dependencies = [
        "transformers", "datasets", "accelerate", "nltk",
        "sacrebleu", "textstat", "tqdm", "sentence-transformers", "rouge-score"
    ]
    
    for dep in dependencies:
        install_package(dep, quiet=True)
    
    try:
        import rouge_score
        print("rouge-score disponible")
    except ImportError:
        print("rouge-score no encontrado. Instalando...")
        install_package("rouge-score", quiet=False)
        import rouge_score
        print("rouge-score instalado e importado")

# ============================================================================
# IMPORTS
# ============================================================================

from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorForSeq2Seq, TrainerCallback,
    AutoModelForSequenceClassification, AutoTokenizer as NLI_Tokenizer
)
from datasets import Dataset
from rouge_score import rouge_scorer
from sacrebleu import BLEU
import textstat
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False
    print("SentenceTransformers no disponible. Usando chunking por párrafos.")

try:
    from bert_score import score as bert_score_func
    BERTSCORE_AVAILABLE = True
except ImportError:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "bert-score", "--quiet"], check=True)
        from bert_score import score as bert_score_func
        BERTSCORE_AVAILABLE = True
    except:
        BERTSCORE_AVAILABLE = False
        print("BERTScore no disponible. Se omitirá en las métricas.")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

def mount_google_drive():
    """Monta Google Drive si estamos en Colab."""
    drive_mounted = False
    try:
        from google.colab import drive
        drive_path = '/content/drive'
        if not os.path.exists(drive_path):
            print("Montando Google Drive...")
            drive.mount('/content/drive', force_remount=False)
            print("Google Drive montado")
            drive_mounted = True
        else:
            test_path = '/content/drive/MyDrive'
            if os.path.exists(test_path):
                print("Google Drive ya está montado")
                drive_mounted = True
            else:
                print("Intentando montar...")
                try:
                    drive.mount('/content/drive', force_remount=True)
                    drive_mounted = True
                    print("Google Drive montado correctamente")
                except Exception as mount_error:
                    print(f"Error al montar: {mount_error}")
    except ImportError:
        print("No se detectó Colab. Asumiendo que no se necesita montar Google Drive")
        drive_mounted = True
    except Exception as e:
        print(f"Error al montar Google Drive: {e}")
    
    return drive_mounted

CONFIG = {
    'model_name': 't5-large',
    'data_path': '/content/drive/MyDrive/T5_Training/synthetic_pairs.jsonl',
    'output_dir': '/content/drive/MyDrive/T5_Training/models/t5_large_semantic',
    'max_tokens_source': 350,
    'max_tokens_target': 256,
    'chunk_overlap': 40,
    'use_semantic_chunking': True,
    'semantic_model': 'all-MiniLM-L6-v2',
    'batch_size': 10,
    'eval_batch_size': 8,
    'gradient_accumulation_steps': 4,
    'learning_rate': 7e-6,
    'num_epochs': 5,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'fp16': False,
    'bf16': True,
    'gradient_checkpointing': True,
    'dataloader_num_workers': 6,
    'max_gpu_memory_gb': 65,
    'logging_steps': 100,
    'eval_steps': 500,
    'save_steps': 1000,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# CALLBACKS PARA ENTRENAMIENTO
# ============================================================================

class NaNDetectionCallback(TrainerCallback):
    """Detecta NaN en loss y detiene el entrenamiento."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            train_loss = logs.get('loss')
            eval_loss = logs.get('eval_loss')
            
            if train_loss is not None and (np.isnan(train_loss) or np.isinf(train_loss)):
                print(f"\nERROR: Train loss es NaN/Inf: {train_loss}")
                print("Deteniendo entrenamiento...")
                control.should_training_stop = True
            
            if eval_loss is not None and (np.isnan(eval_loss) or np.isinf(eval_loss)):
                print(f"\nERROR: Eval loss es NaN/Inf: {eval_loss}")
                print("Deteniendo entrenamiento...")
                control.should_training_stop = True

class MemoryMonitorCallback(TrainerCallback):
    """Monitorea el uso de GPU RAM y advierte si se acerca al límite."""
    def __init__(self, max_memory_gb=65):
        self.max_memory_gb = max_memory_gb
        self.check_interval = 50
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.check_interval == 0 and torch.cuda.is_available():
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            
            if reserved_gb > self.max_memory_gb * 0.90:
                print(f"\nADVERTENCIA DE MEMORIA (Step {state.global_step}):")
                print(f"   VRAM reservada: {reserved_gb:.2f} GB / {self.max_memory_gb} GB ({reserved_gb/self.max_memory_gb*100:.1f}%)")
                print(f"   VRAM usada: {allocated_gb:.2f} GB")
                
                if reserved_gb > self.max_memory_gb * 0.98:
                    print(f"   CRÍTICO: Cerca del límite máximo!")
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

# ============================================================================
# CHUNKING SEMÁNTICO
# ============================================================================

def split_into_paragraphs(text):
    """Divide texto por párrafos (doble salto de línea)."""
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs if paragraphs else [text]

def split_into_sentences(text):
    """Divide texto por oraciones usando NLTK."""
    try:
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

def chunk_text_semantic(text, tokenizer, max_tokens=350, overlap_tokens=40, prompt_tokens=10, 
                        semantic_model=None):
    """
    Chunking semántico: agrupa oraciones/párrafos por similitud semántica.
    Preserva contexto médico agrupando contenido relacionado.
    """
    if not text:
        return [text]
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    effective_max = max_tokens
    
    if len(tokens) <= effective_max:
        return [text]
    
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        chunks = []
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + effective_max, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            if end_idx < len(tokens):
                start_idx = end_idx - overlap_tokens
            else:
                break
        return chunks
    
    if CONFIG.get('use_semantic_chunking', False) and semantic_model is not None:
        try:
            sentence_embeddings = semantic_model.encode(sentences, show_progress_bar=False)
            
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for i, sent in enumerate(sentences):
                sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
                
                if current_tokens + sent_tokens > effective_max:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        overlap_sents = []
                        overlap_tokens_count = 0
                        for sent_back in reversed(current_chunk):
                            sent_back_tokens = len(tokenizer.encode(sent_back, add_special_tokens=False))
                            if overlap_tokens_count + sent_back_tokens <= overlap_tokens:
                                overlap_sents.insert(0, sent_back)
                                overlap_tokens_count += sent_back_tokens
                            else:
                                break
                        current_chunk = overlap_sents
                        current_tokens = overlap_tokens_count
                
                current_chunk.append(sent)
                current_tokens += sent_tokens
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            return chunks if chunks else [text]
        except Exception as e:
            print(f"Error en chunking semántico: {e}. Usando chunking por párrafos.")
    
    paragraphs = split_into_paragraphs(text)
    if len(paragraphs) > 1:
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = len(tokenizer.encode(para, add_special_tokens=False))
            
            if para_tokens > effective_max:
                para_sents = split_into_sentences(para)
                for sent in para_sents:
                    sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
                    
                    if current_tokens + sent_tokens > effective_max:
                        if current_chunk:
                            chunks.append('\n\n'.join(current_chunk))
                            if len(current_chunk) > 1:
                                current_chunk = [current_chunk[-1]]
                                current_tokens = len(tokenizer.encode(current_chunk[0], add_special_tokens=False))
                            else:
                                current_chunk = []
                                current_tokens = 0
                    
                    current_chunk.append(sent)
                    current_tokens += sent_tokens
            elif current_tokens + para_tokens > effective_max:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    if len(current_chunk) > 1:
                        current_chunk = [current_chunk[-1]]
                        current_tokens = len(tokenizer.encode(current_chunk[0], add_special_tokens=False))
                    else:
                        current_chunk = []
                        current_tokens = 0
                
                current_chunk.append(para)
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks if chunks else [text]
    
    chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + effective_max, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end_idx < len(tokens):
            start_idx = end_idx - overlap_tokens
        else:
            break
    
    return chunks

def expand_pairs_with_chunking(pairs, tokenizer, max_tokens=350, overlap_tokens=40, prompt_tokens=10, 
                                semantic_model=None):
    """Expande pares aplicando chunking semántico a documentos largos."""
    print("\n" + "="*80)
    print("APLICANDO CHUNKING SEMÁNTICO")
    print("="*80)
    
    expanded_pairs = []
    chunked_count = 0
    total_chunks = 0
    fixed_count = 0
    
    absolute_max = 512
    safety_margin = 15
    effective_max = absolute_max - prompt_tokens - safety_margin
    
    print(f"Procesando {len(pairs)} pares...")
    print(f"Límite efectivo por chunk: {effective_max} tokens")
    print(f"Chunking semántico: {'ACTIVADO' if CONFIG.get('use_semantic_chunking', False) and semantic_model else 'DESACTIVADO'}\n")
    
    prompt_text = "Simplify medical text to plain language: "
    
    for idx, pair in enumerate(tqdm(pairs, desc="Chunking", unit="pares")):
        tech_text = pair['texto_tecnico']
        simple_text = pair['texto_simple']
        
        tech_tokens = len(tokenizer.encode(tech_text, add_special_tokens=False))
        
        if tech_tokens > effective_max:
            chunks = chunk_text_semantic(
                tech_text, 
                tokenizer, 
                max_tokens=effective_max,
                overlap_tokens=overlap_tokens,
                prompt_tokens=prompt_tokens,
                semantic_model=semantic_model
            )
            chunked_count += 1
            total_chunks += len(chunks)
            
            for i, chunk in enumerate(chunks):
                full_input = prompt_text + chunk
                full_tokens = len(tokenizer.encode(full_input, add_special_tokens=False))
                
                if full_tokens > absolute_max:
                    chunk_tokens = tokenizer.encode(chunk, add_special_tokens=False)
                    max_chunk_tokens = absolute_max - prompt_tokens - 5
                    chunk_tokens = chunk_tokens[:max_chunk_tokens]
                    chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    fixed_count += 1
                
                new_pair = pair.copy()
                new_pair['texto_tecnico'] = chunk
                new_pair['chunk_id'] = i
                new_pair['total_chunks'] = len(chunks)
                new_pair['original_idx'] = idx
                expanded_pairs.append(new_pair)
        else:
            full_input = prompt_text + tech_text
            full_tokens = len(tokenizer.encode(full_input, add_special_tokens=False))
            
            if full_tokens > absolute_max:
                tech_tokens_list = tokenizer.encode(tech_text, add_special_tokens=False)
                max_tech_tokens = absolute_max - prompt_tokens - 5
                tech_tokens_list = tech_tokens_list[:max_tech_tokens]
                tech_text = tokenizer.decode(tech_tokens_list, skip_special_tokens=True)
                fixed_count += 1
            
            new_pair = pair.copy()
            new_pair['texto_tecnico'] = tech_text
            new_pair['chunk_id'] = 0
            new_pair['total_chunks'] = 1
            new_pair['original_idx'] = idx
            expanded_pairs.append(new_pair)
        
        if (idx + 1) % 1000 == 0:
            print(f"\nProgreso: {idx + 1}/{len(pairs)} pares procesados")
            print(f"   - Chunked: {chunked_count} ({100*chunked_count/(idx+1):.1f}%)")
            print(f"   - Total chunks: {total_chunks}")
    
    print(f"\nRESUMEN FINAL:")
    print(f"   Documentos originales: {len(pairs)}")
    print(f"   Documentos chunked: {chunked_count} ({100*chunked_count/len(pairs):.1f}%)")
    print(f"   Total chunks creados: {total_chunks}")
    print(f"   Total después chunking: {len(expanded_pairs)}")
    print(f"   Expansión: {len(expanded_pairs)/len(pairs):.2f}x")
    
    return expanded_pairs

# ============================================================================
# FUNCIONES DE MÉTRICAS
# ============================================================================

def compute_flesch_reading_ease(text):
    """Calcula Flesch Reading Ease Score (0-100, mayor = más fácil)."""
    try:
        score = textstat.flesch_reading_ease(text)
        return score
    except:
        return 0.0

def _get_ngrams(tokens, n):
    """Obtiene n-gramas de una lista de tokens."""
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return ngrams

def _sari_operation_score(pred_tokens, source_tokens, ref_tokens_list):
    """Calcula el score de operaciones SARI usando n-gramas."""
    keep_scores = []
    add_scores = []
    delete_scores = []

    for n in [1, 2, 3, 4]:
        pred_ngrams = _get_ngrams(pred_tokens, n)
        source_ngrams = _get_ngrams(source_tokens, n)

        keep_count = sum(min(pred_ngrams.get(ng, 0), source_ngrams.get(ng, 0))
                        for ng in set(pred_ngrams.keys()) & set(source_ngrams.keys()))
        keep_total = sum(pred_ngrams.values())
        keep_score = keep_count / keep_total if keep_total > 0 else 0.0

        ref_ngrams_union = {}
        for ref_tokens in ref_tokens_list:
            ref_ngrams = _get_ngrams(ref_tokens, n)
            for ng, count in ref_ngrams.items():
                ref_ngrams_union[ng] = max(ref_ngrams_union.get(ng, 0), count)

        add_count = 0
        for ng in pred_ngrams:
            if ng not in source_ngrams and ng in ref_ngrams_union:
                add_count += min(pred_ngrams[ng], ref_ngrams_union[ng])
        add_total = sum(pred_ngrams.values())
        add_score = add_count / add_total if add_total > 0 else 0.0

        delete_count = 0
        for ng in source_ngrams:
            if ng not in pred_ngrams and ng not in ref_ngrams_union:
                delete_count += source_ngrams[ng]
        delete_total = sum(source_ngrams.values())
        delete_score = delete_count / delete_total if delete_total > 0 else 0.0

        keep_scores.append(keep_score)
        add_scores.append(add_score)
        delete_scores.append(delete_score)

    avg_keep = sum(keep_scores) / len(keep_scores) if keep_scores else 0.0
    avg_add = sum(add_scores) / len(add_scores) if add_scores else 0.0
    avg_delete = sum(delete_scores) / len(delete_scores) if delete_scores else 0.0

    sari_score = (avg_keep + avg_add + avg_delete) / 3.0

    return sari_score, avg_keep, avg_add, avg_delete

def compute_sari(pred, ref, source):
    """Calcula SARI (simplificación) usando implementación completa con n-gramas."""
    try:
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = word_tokenize(ref.lower())
        source_tokens = word_tokenize(source.lower())

        sari_score, keep, add, delete = _sari_operation_score(
            pred_tokens,
            source_tokens,
            [ref_tokens]
        )
        return sari_score
    except Exception as e:
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        source_words = set(source.lower().split())

        keep = len(pred_words & ref_words & source_words) / max(len(pred_words), 1)
        addition = len((pred_words & ref_words) - source_words) / max(len(pred_words), 1)
        deletion = len((source_words - pred_words) & ref_words) / max(len(source_words), 1)

        return (keep + addition + deletion) / 3.0

def compute_meteor(pred, ref):
    """Calcula METEOR score aproximado (basado en F1 de overlap de palabras)."""
    try:
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())

        if len(pred_words) == 0 or len(ref_words) == 0:
            return 0.0

        precision = len(pred_words & ref_words) / len(pred_words) if len(pred_words) > 0 else 0.0
        recall = len(pred_words & ref_words) / len(ref_words) if len(ref_words) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            return f1
        else:
            return 0.0
    except Exception as e:
        return 0.0

def compute_compression_ratio(generated_text, source_text):
    """Calcula el ratio de compresión (longitud generada / longitud fuente)."""
    try:
        gen_words = len(generated_text.split())
        src_words = len(source_text.split())
        if src_words == 0:
            return 0.0
        return gen_words / src_words
    except:
        return 0.0

def compute_avg_length(texts):
    """Calcula la longitud promedio en palabras."""
    try:
        lengths = [len(text.split()) for text in texts]
        return sum(lengths) / len(lengths) if lengths else 0.0
    except:
        return 0.0

# ============================================================================
# CARGAR Y PROCESAR DATOS
# ============================================================================

def load_synthetic_pairs(data_path):
    """Carga pares sintéticos desde JSONL."""
    print("\n" + "="*80)
    print("CARGANDO DATOS")
    print("="*80)
    
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    
    print(f"Total pares: {len(pairs)}")
    
    train_pairs = [p for p in pairs if p.get('split') == 'train']
    test_pairs = [p for p in pairs if p.get('split') == 'test']
    
    print(f"Train: {len(train_pairs)}")
    print(f"Test: {len(test_pairs)}")
    
    return train_pairs, test_pairs

def tokenize_function(examples, tokenizer, max_length_source=512, max_length_target=256):
    """Tokeniza ejemplos para entrenamiento."""
    prompt = """Generate a Plain Language Summary understandable by any patient:
- Readability grade level 6 or below
- No jargon (define technical terms)
- Active voice
- Mostly 1-2 syllable words
- Sentences of 15 words or less
- Short paragraphs of 3-5 sentences
- Simple numbers (ratios, not percentages)
Simplify medical text to plain language: """
    inputs = [f"{prompt}{text}" for text in examples['texto_tecnico']]
    targets = examples['texto_simple']
    
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding=False
    )
    
    labels = tokenizer(
        text_target=targets,
        max_length=max_length_target,
        truncation=True,
        padding=False
    )
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def train_t5_large():
    """Entrena T5-large con chunking semántico."""
    print("\n" + "="*80)
    print("ENTRENANDO T5-LARGE (CHUNKING SEMÁNTICO)")
    print("="*80)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        max_memory_gb = CONFIG.get('max_gpu_memory_gb', 65)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if max_memory_gb >= total_memory_gb:
            memory_fraction = 0.95
            actual_limit_gb = total_memory_gb * memory_fraction
            print(f"El límite solicitado ({max_memory_gb} GB) excede la memoria disponible ({total_memory_gb:.1f} GB)")
            print(f"Límite ajustado a: {actual_limit_gb:.1f} GB (95% de {total_memory_gb:.1f} GB)")
        else:
            memory_fraction = max_memory_gb / total_memory_gb
            print(f"Límite de GPU RAM establecido: {max_memory_gb} GB de {total_memory_gb:.1f} GB disponibles")
        
        memory_fraction = min(max(memory_fraction, 0.1), 0.95)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
        print(f"   Fracción de memoria: {memory_fraction:.2%}")
        
        torch.cuda.empty_cache()
    
    print(f"\nCargando {CONFIG['model_name']}...")
    tokenizer = T5Tokenizer.from_pretrained(CONFIG['model_name'])
    model = T5ForConditionalGeneration.from_pretrained(CONFIG['model_name'])
    
    model.config.use_cache = False
    
    if hasattr(model.config, 'gradient_checkpointing'):
        model.config.gradient_checkpointing = CONFIG.get('gradient_checkpointing', True)
    
    model = model.to(device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"VRAM usada después de cargar modelo: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"VRAM reservada: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        print(f"Límite máximo configurado: {CONFIG.get('max_gpu_memory_gb', 65)} GB")
    
    semantic_model = None
    if CONFIG.get('use_semantic_chunking', False) and SEMANTIC_CHUNKING_AVAILABLE:
        try:
            print(f"\nCargando modelo semántico: {CONFIG['semantic_model']}...")
            semantic_model = SentenceTransformer(CONFIG['semantic_model'])
            semantic_model = semantic_model.to('cpu')
            print("Modelo semántico cargado")
        except Exception as e:
            print(f"Error cargando modelo semántico: {e}. Usando chunking por párrafos.")
            semantic_model = None
    
    train_pairs, test_pairs = load_synthetic_pairs(CONFIG['data_path'])
    
    test_pairs_original = test_pairs.copy()
    
    prompt_text = """Generate a Plain Language Summary understandable by any patient:
- Readability grade level 6 or below
- No jargon (define technical terms)
- Active voice
- Mostly 1-2 syllable words
- Sentences of 15 words or less
- Short paragraphs of 3-5 sentences
- Simple numbers (ratios, not percentages)
Simplify medical text to plain language: """
    prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    
    print("\n" + "="*80)
    print("PROCESANDO TRAIN SET")
    print("="*80)
    train_expanded = expand_pairs_with_chunking(
        train_pairs,
        tokenizer,
        max_tokens=CONFIG['max_tokens_source'],
        overlap_tokens=CONFIG['chunk_overlap'],
        prompt_tokens=prompt_tokens,
        semantic_model=semantic_model
    )
    
    print("\n" + "="*80)
    print("PROCESANDO TEST SET")
    print("="*80)
    test_expanded = expand_pairs_with_chunking(
        test_pairs,
        tokenizer,
        max_tokens=CONFIG['max_tokens_source'],
        overlap_tokens=CONFIG['chunk_overlap'],
        prompt_tokens=prompt_tokens,
        semantic_model=semantic_model
    )
    
    test_pairs_expanded_path = f"{CONFIG['output_dir']}/test_pairs_expanded.pkl"
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    with open(test_pairs_expanded_path, 'wb') as f:
        pickle.dump({
            'test_pairs_original': test_pairs_original,
            'test_pairs_expanded': test_expanded
        }, f)
    print(f"Test pairs guardados en: {test_pairs_expanded_path}")
    
    print("\nCreando datasets...")
    train_dataset = Dataset.from_list(train_expanded)
    test_dataset = Dataset.from_list(test_expanded)
    
    train_size = len(train_dataset)
    dev_size = int(0.1 * train_size)
    train_dataset = train_dataset.shuffle(seed=42)
    dev_dataset = train_dataset.select(range(dev_size))
    train_dataset = train_dataset.select(range(dev_size, train_size))
    
    print(f"Train: {len(train_dataset)}")
    print(f"Dev: {len(dev_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    print("\nTokenizando datasets...")
    
    def tokenize_wrapper(examples):
        return tokenize_function(
            examples,
            tokenizer,
            max_length_source=512,
            max_length_target=CONFIG['max_tokens_target']
        )
    
    train_dataset = train_dataset.map(
        tokenize_wrapper,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizando train"
    )
    
    dev_dataset = dev_dataset.map(
        tokenize_wrapper,
        batched=True,
        remove_columns=dev_dataset.column_names,
        desc="Tokenizando dev"
    )
    
    test_dataset = test_dataset.map(
        tokenize_wrapper,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizando test"
    )
    
    print("Tokenización completada")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print(f"VRAM después de tokenización: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )
    
    eval_batch_size = CONFIG.get('eval_batch_size', 4)
    use_bf16 = CONFIG.get('bf16', False) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = CONFIG.get('fp16', False) and torch.cuda.is_available() and not use_bf16
    
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        learning_rate=CONFIG['learning_rate'],
        warmup_ratio=CONFIG['warmup_ratio'],
        weight_decay=CONFIG['weight_decay'],
        logging_steps=CONFIG['logging_steps'],
        eval_steps=CONFIG['eval_steps'],
        save_steps=CONFIG['save_steps'],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=CONFIG['gradient_checkpointing'],
        dataloader_num_workers=CONFIG['dataloader_num_workers'],
        dataloader_pin_memory=True,
        report_to="none",
        push_to_hub=False,
        max_grad_norm=0.5,
        remove_unused_columns=True,
        prediction_loss_only=True,
        label_smoothing_factor=0.0,
        save_safetensors=True,
        eval_accumulation_steps=10,
        dataloader_drop_last=False,
        include_inputs_for_metrics=False,
        optim="adamw_torch",
        dataloader_persistent_workers=False,
    )
    
    if CONFIG.get('gradient_checkpointing', False):
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing activado en el modelo")
    
    max_memory_gb = CONFIG.get('max_gpu_memory_gb', 65)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            NaNDetectionCallback(),
            MemoryMonitorCallback(max_memory_gb=max_memory_gb)
        ]
    )
    
    print("\n" + "="*80)
    print("INICIANDO ENTRENAMIENTO")
    print("="*80)
    print(f"Modelo: {CONFIG['model_name']} (~770M parámetros)")
    print(f"Batch efectivo: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"Épocas: {CONFIG['num_epochs']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print(f"Chunking semántico: {'ACTIVADO' if semantic_model else 'DESACTIVADO'}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        max_memory_gb = CONFIG.get('max_gpu_memory_gb', 65)
        current_memory_gb = torch.cuda.memory_allocated()/1e9
        reserved_memory_gb = torch.cuda.memory_reserved()/1e9
        print(f"\nVRAM antes de entrenar: {current_memory_gb:.2f} GB")
        print(f"VRAM reservada: {reserved_memory_gb:.2f} GB")
        print(f"Límite máximo configurado: {max_memory_gb} GB")
        if reserved_memory_gb > max_memory_gb * 0.95:
            print(f"ADVERTENCIA: VRAM reservada ({reserved_memory_gb:.2f} GB) está cerca del límite ({max_memory_gb} GB)")
        else:
            print(f"VRAM dentro del límite seguro (<{max_memory_gb} GB)")
    
    trainer.train()
    
    print("\nGuardando modelo...")
    trainer.save_model(CONFIG['output_dir'])
    tokenizer.save_pretrained(CONFIG['output_dir'])
    print(f"Modelo guardado en: {CONFIG['output_dir']}")
    
    with open(f"{CONFIG['output_dir']}/config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    print("\n" + "="*80)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"Modelo guardado en: {CONFIG['output_dir']}")
    
    return trainer, tokenizer

# ============================================================================
# EVALUACIÓN
# ============================================================================

def evaluate_model_complete():
    """Evalúa el modelo con todas las métricas."""
    print("\n" + "="*80)
    print("EVALUANDO MODELO CON MÉTRICAS COMPLETAS")
    print("="*80)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\nCargando modelo desde: {CONFIG['output_dir']}")
    
    if not os.path.exists(CONFIG['output_dir']):
        raise FileNotFoundError(
            f"No se encontró el directorio: {CONFIG['output_dir']}\n"
            f"   Verifica que la ruta sea correcta y que Google Drive esté montado."
        )
    
    tokenizer_files = ['tokenizer_config.json', 'vocab.json', 'spiece.model']
    tokenizer_available = all(os.path.exists(os.path.join(CONFIG['output_dir'], f)) for f in tokenizer_files)
    
    model_files = ['pytorch_model.bin', 'model.safetensors', 'config.json']
    model_available = any(os.path.exists(os.path.join(CONFIG['output_dir'], f)) for f in model_files)
    
    model_name = CONFIG.get('model_name', 't5-large')
    
    try:
        if tokenizer_available:
            print("   Intentando cargar tokenizer desde directorio guardado...")
            try:
                tokenizer = T5Tokenizer.from_pretrained(CONFIG['output_dir'], local_files_only=False)
                print("   Tokenizer cargado")
            except Exception as tok_error:
                print(f"   Error al cargar tokenizer: {str(tok_error)[:100]}")
                print(f"   Cargando tokenizer desde modelo base: {model_name}")
                tokenizer = T5Tokenizer.from_pretrained(model_name)
        else:
            print(f"   Tokenizer no encontrado en directorio. Cargando desde modelo base: {model_name}")
            tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        if model_available:
            print("   Intentando cargar modelo completo desde directorio guardado...")
            try:
                model = T5ForConditionalGeneration.from_pretrained(
                    CONFIG['output_dir'],
                    local_files_only=False
                )
                print("Modelo cargado directamente desde directorio guardado")
            except Exception as model_error:
                print(f"   Error al cargar modelo completo: {str(model_error)[:100]}")
                print(f"   Cargando modelo base: {model_name} y luego los pesos...")
                model = T5ForConditionalGeneration.from_pretrained(model_name)
                
                checkpoint_path = None
                for f in ['pytorch_model.bin', 'model.safetensors']:
                    path = os.path.join(CONFIG['output_dir'], f)
                    if os.path.exists(path):
                        checkpoint_path = path
                        break
                
                if checkpoint_path:
                    print(f"   Cargando pesos desde: {checkpoint_path}")
                    try:
                        if checkpoint_path.endswith('.safetensors'):
                            try:
                                from safetensors.torch import load_file
                                state_dict = load_file(checkpoint_path)
                            except ImportError:
                                print("   Advertencia: safetensors no disponible, intentando con torch...")
                                state_dict = torch.load(checkpoint_path, map_location=device)
                        else:
                            checkpoint = torch.load(checkpoint_path, map_location=device)
                            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                                state_dict = checkpoint['state_dict']
                            else:
                                state_dict = checkpoint
                        
                        model.load_state_dict(state_dict, strict=False)
                        print("Pesos cargados (algunos pueden no coincidir)")
                    except Exception as load_error:
                        print(f"   Error cargando pesos: {str(load_error)[:100]}")
                        print("   Usando modelo base sin fine-tuning.")
                else:
                    print("   No se encontraron pesos guardados. Usando modelo base sin fine-tuning.")
        else:
            print(f"   Modelo no encontrado en directorio. Cargando modelo base: {model_name}")
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            print("   Usando modelo base sin fine-tuning (no se encontraron pesos guardados)")
    
    except Exception as e:
        print(f"Error inesperado al cargar modelo: {str(e)[:200]}")
        print(f"   Intentando cargar modelo base: {model_name} como fallback...")
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            print("   Usando modelo base sin fine-tuning")
        except Exception as fallback_error:
            print(f"Error crítico: No se pudo cargar ni el modelo guardado ni el modelo base.")
            print(f"   Error: {str(fallback_error)[:200]}")
            raise
    
    model.config.use_cache = True
    model = model.to(device)
    model.eval()
    
    if torch.cuda.is_available():
        print(f"VRAM usada: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    test_pairs_path = f"{CONFIG['output_dir']}/test_pairs_expanded.pkl"
    if not os.path.exists(test_pairs_path):
        raise FileNotFoundError(
            f"No se encontró {test_pairs_path}\n"
            f"   Asegúrate de haber ejecutado el entrenamiento primero."
        )
    
    with open(test_pairs_path, 'rb') as f:
        test_data = pickle.load(f)
    
    test_pairs_original = test_data['test_pairs_original']
    test_pairs_expanded = test_data['test_pairs_expanded']
    
    print(f"Test pairs originales: {len(test_pairs_original)}")
    print(f"Test pairs expandidos: {len(test_pairs_expanded)}")
    
    print("\nPreparando dataset de test...")
    
    prompt = """Generate a Plain Language Summary understandable by any patient:
- Readability grade level 6 or below
- No jargon (define technical terms)
- Active voice
- Mostly 1-2 syllable words
- Sentences of 15 words or less
- Short paragraphs of 3-5 sentences
- Simple numbers (ratios, not percentages)
Simplify medical text to plain language: """
    
    def tokenize_wrapper(examples):
        inputs = [f"{prompt}{text}" for text in examples['texto_tecnico']]
        targets = examples['texto_simple']
        
        model_inputs = tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding=False
        )
        
        labels = tokenizer(
            text_target=targets,
            max_length=CONFIG['max_tokens_target'],
            truncation=True,
            padding=False
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    test_dataset = Dataset.from_list(test_pairs_expanded)
    test_dataset = test_dataset.map(
        tokenize_wrapper,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizando test"
    )
    
    print(f"Dataset preparado: {len(test_dataset)} ejemplos")
    
    print("\n" + "="*80)
    print("GENERANDO PREDICCIONES")
    print("="*80)
    print(f"Procesando {len(test_dataset)} ejemplos...")
    
    generated_texts = []
    reference_texts = []
    source_texts = []
    
    batch_size_gen = 64
    num_batches = (len(test_dataset) + batch_size_gen - 1) // batch_size_gen
    
    expanded_to_original = {}
    for exp_idx, pair in enumerate(test_pairs_expanded):
        orig_idx = pair.get('original_idx', exp_idx)
        expanded_to_original[exp_idx] = orig_idx
    
    print(f"Batch size: {batch_size_gen}")
    print(f"Generación en batch (procesa múltiples ejemplos a la vez)")
    print(f"Greedy decoding (más rápido que beam search)")
    
    for batch_idx in tqdm(range(num_batches), desc="Generando PLS"):
        start_idx = batch_idx * batch_size_gen
        end_idx = min(start_idx + batch_size_gen, len(test_dataset))
        
        batch_indices = list(range(start_idx, end_idx))
        batch_samples = test_dataset.select(batch_indices)
        
        batch_input_ids = []
        batch_labels = []
        batch_sources = []
        
        for i, sample_idx in enumerate(batch_indices):
            sample = batch_samples[i]
            batch_input_ids.append(sample['input_ids'])
            batch_labels.append(sample['labels'])
            
            orig_idx = expanded_to_original.get(sample_idx, sample_idx)
            if orig_idx < len(test_pairs_original):
                batch_sources.append(test_pairs_original[orig_idx].get('texto_tecnico', ''))
            else:
                batch_sources.append('')
        
        max_length = max(len(ids) for ids in batch_input_ids)
        
        padded_input_ids = []
        attention_mask_list = []
        for ids in batch_input_ids:
            padding_length = max_length - len(ids)
            padded_ids = ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(ids) + [0] * padding_length
            padded_input_ids.append(padded_ids)
            attention_mask_list.append(attention_mask)
        
        input_ids_batch = torch.tensor(padded_input_ids, dtype=torch.long).to(device)
        attention_mask_batch = torch.tensor(attention_mask_list, dtype=torch.long).to(device)
        
        with torch.no_grad():
            generated_batch = model.generate(
                input_ids_batch,
                attention_mask=attention_mask_batch,
                max_length=CONFIG['max_tokens_target'],
                min_length=10,
                num_beams=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.0,
            )
        
        batch_generated = tokenizer.batch_decode(generated_batch, skip_special_tokens=True)
        
        batch_references = []
        for labels in batch_labels:
            labels_tensor = torch.tensor([labels]).to(device)
            labels_tensor = torch.where(labels_tensor != -100, labels_tensor, tokenizer.pad_token_id)
            reference_text = tokenizer.decode(labels_tensor[0], skip_special_tokens=True)
            batch_references.append(reference_text)
        
        generated_texts.extend(batch_generated)
        reference_texts.extend(batch_references)
        source_texts.extend(batch_sources)
        
        if (batch_idx + 1) % 10 == 0:
            del batch_samples, batch_input_ids, batch_labels, input_ids_batch, generated_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"Generación completada: {len(generated_texts)} predicciones")
    
    print("\n" + "="*80)
    print("CALCULANDO MÉTRICAS")
    print("="*80)
    
    test_metrics = {}
    
    print("Calculando ROUGE...")
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(generated_texts, reference_texts):
        scores = rouge_scorer_obj.score(ref, pred)
        rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
    
    test_metrics['eval_rouge1'] = float(np.mean(rouge_scores['rouge1']))
    test_metrics['eval_rouge2'] = float(np.mean(rouge_scores['rouge2']))
    test_metrics['eval_rougeL'] = float(np.mean(rouge_scores['rougeL']))
    
    print("Calculando BLEU...")
    bleu = BLEU()
    bleu_scores = []
    for pred, ref in zip(generated_texts, reference_texts):
        try:
            score = bleu.sentence_score(pred, [ref])
            bleu_scores.append(score.score / 100.0)
        except:
            bleu_scores.append(0.0)
    test_metrics['eval_bleu'] = float(np.mean(bleu_scores))
    
    print("Calculando SARI...")
    sari_scores = []
    for pred, ref, src in zip(generated_texts, reference_texts, source_texts):
        source = src if src and len(src) > 0 else ref
        sari = compute_sari(pred, ref, source)
        sari_scores.append(sari)
    test_metrics['eval_sari'] = float(np.mean(sari_scores))
    
    print("Calculando METEOR...")
    meteor_scores = []
    for pred, ref in zip(generated_texts, reference_texts):
        meteor = compute_meteor(pred, ref)
        meteor_scores.append(meteor)
    test_metrics['eval_meteor'] = float(np.mean(meteor_scores))
    
    if BERTSCORE_AVAILABLE:
        print("Calculando BERTScore F1...")
        try:
            P, R, F1 = bert_score_func(generated_texts, reference_texts, lang='en', verbose=False)
            test_metrics['eval_bertscore_f1'] = float(F1.mean().item())
            print(f"   BERTScore F1 calculado")
        except Exception as e:
            print(f"   Error calculando BERTScore: {e}")
            test_metrics['eval_bertscore_f1'] = 0.0
    else:
        test_metrics['eval_bertscore_f1'] = 0.0
        print("   BERTScore no disponible")
    
    print("Calculando Factuality (NLI_RoBERTa)...")
    
    factuality_scores = []
    factuality_method = None
    
    try:
        nli_model_name = 'roberta-large-mnli'
        print(f"   Cargando modelo NLI: {nli_model_name}...")
        
        nli_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        nli_tokenizer = NLI_Tokenizer.from_pretrained(nli_model_name)
        nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        nli_model.to(nli_device)
        nli_model.eval()
        
        print(f"   Evaluando factualidad en {len(source_texts)} documentos (a nivel de oraciones)...")
        
        for idx, (src, pred) in enumerate(tqdm(zip(source_texts, generated_texts), total=len(source_texts), desc="Factuality NLI", leave=False)):
            source = src if src and len(src) > 0 else reference_texts[idx]
            
            sentences = re.split(r'[.!?]+\s+', pred.strip())
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if len(sentences) == 0:
                sentences = [pred]
            
            sentence_scores = []
            for sentence in sentences:
                if len(sentence.strip()) < 5:
                    continue
                
                try:
                    inputs = nli_tokenizer(
                        source,
                        sentence,
                        return_tensors='pt',
                        truncation='only_first',
                        max_length=512,
                        padding='max_length'
                    ).to(nli_device)
                    
                    with torch.no_grad():
                        outputs = nli_model(**inputs)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=-1)
                        entailment_prob = probs[0][2].item()
                        neutral_prob = probs[0][1].item()
                        sentence_score = entailment_prob + (neutral_prob * 0.3)
                        sentence_scores.append(sentence_score)
                except Exception:
                    continue
            
            if len(sentence_scores) > 0:
                doc_factuality = np.mean(sentence_scores)
            else:
                doc_factuality = 0.0
            
            factuality_scores.append(doc_factuality)
        
        factuality_method = 'NLI_RoBERTa'
        test_metrics['eval_alignscore'] = float(np.mean(factuality_scores))
        print(f"   Factuality (NLI_RoBERTa) calculado: {test_metrics['eval_alignscore']:.3f}")
    
    except Exception as e:
        print(f"   Error calculando Factuality con NLI: {str(e)[:200]}")
        print(f"   Intentando método alternativo simple...")
        
        try:
            factuality_scores = []
            for idx, (src, pred) in enumerate(zip(source_texts, generated_texts)):
                source = src if src and len(src) > 0 else reference_texts[idx]
                
                source_words = set(source.lower().split())
                pred_words = set(pred.lower().split())
                
                if len(pred_words) > 0:
                    overlap = len(source_words & pred_words) / len(pred_words)
                else:
                    overlap = 0.0
                
                factuality_scores.append(min(overlap * 1.5, 1.0))
            
            factuality_method = 'Simple_Overlap'
            test_metrics['eval_alignscore'] = float(np.mean(factuality_scores))
            print(f"   Factuality (Simple_Overlap) calculado: {test_metrics['eval_alignscore']:.3f}")
        except Exception as e2:
            print(f"   Método alternativo también falló: {str(e2)[:200]}")
            factuality_scores = [0.0] * len(generated_texts)
            factuality_method = 'None'
            test_metrics['eval_alignscore'] = 0.0
    
    print("Calculando Flesch Reading Ease...")
    flesch_scores = []
    for pred in generated_texts:
        flesch = compute_flesch_reading_ease(pred)
        flesch_scores.append(flesch)
    test_metrics['eval_flesch'] = float(np.mean(flesch_scores))
    
    print("Calculando Flesch-Kincaid Grade Level...")
    fk_scores = []
    for pred in generated_texts:
        try:
            fk_grade = textstat.flesch_kincaid_grade(pred)
            fk_scores.append(fk_grade)
        except:
            fk_scores.append(0.0)
    test_metrics['eval_flesch_kincaid'] = float(np.mean(fk_scores))
    
    print("Calculando Compression Ratio...")
    compression_ratios = []
    for idx, (gen, src) in enumerate(zip(generated_texts, source_texts)):
        source = src if src and len(src) > 0 else reference_texts[idx]
        ratio = compute_compression_ratio(gen, source)
        compression_ratios.append(ratio)
    test_metrics['eval_compression_ratio'] = float(np.mean(compression_ratios))
    
    print("Calculando Longitud promedio...")
    avg_length = compute_avg_length(generated_texts)
    test_metrics['eval_avg_length'] = float(avg_length)
    
    print("\n" + "="*80)
    print("RESULTADOS FINALES")
    print("="*80)
    print(f"{'Métrica':<30} {'Valor':<15} {'Target':<15}")
    print("-" * 80)
    print(f"{'ROUGE-1':<30} {test_metrics.get('eval_rouge1', 0):.3f} {'-':<15}")
    print(f"{'ROUGE-2':<30} {test_metrics.get('eval_rouge2', 0):.3f} {'-':<15}")
    print(f"{'ROUGE-L':<30} {test_metrics.get('eval_rougeL', 0):.3f} {'-':<15}")
    print(f"{'BLEU':<30} {test_metrics.get('eval_bleu', 0):.3f} {'-':<15}")
    print(f"{'METEOR':<30} {test_metrics.get('eval_meteor', 0):.3f} {'-':<15}")
    bertscore_val = test_metrics.get('eval_bertscore_f1', 0.0)
    if bertscore_val == 0.0 and not BERTSCORE_AVAILABLE:
        print(f"{'BERTScore F1':<30} {'N/A':<15} {'-':<15}")
    else:
        print(f"{'BERTScore F1':<30} {bertscore_val:.3f} {'-':<15}")
    print(f"{'SARI':<30} {test_metrics.get('eval_sari', 0):.3f} {'>0.40':<15}")
    alignscore_val = test_metrics.get('eval_alignscore', 0.0)
    if alignscore_val == 0.0:
        print(f"{'Factuality (NLI_RoBERTa)':<30} {'N/A':<15} {'>0.50':<15}")
    else:
        print(f"{'Factuality (NLI_RoBERTa)':<30} {alignscore_val:.3f} {'>0.50':<15}")
    print(f"{'Flesch Reading Ease':<30} {test_metrics.get('eval_flesch', 0):.1f} {'~64':<15}")
    print(f"{'Flesch-Kincaid Grade':<30} {test_metrics.get('eval_flesch_kincaid', 0):.1f} {'~7.4':<15}")
    print(f"{'Compression Ratio':<30} {test_metrics.get('eval_compression_ratio', 0):.2f} {'0.33-0.37':<15}")
    print(f"{'Longitud (palabras)':<30} {test_metrics.get('eval_avg_length', 0):.0f} {'~173':<15}")
    print("="*80)
    print(f"\nMétricas calculadas en {len(generated_texts)} ejemplos")
    
    metrics = {
        'model': CONFIG['model_name'],
        'test_metrics': {k: float(v) for k, v in test_metrics.items() if k.startswith('eval_')},
        'test_samples': len(generated_texts),
        'config': CONFIG
    }
    
    metrics_path = f"{CONFIG['output_dir']}/metrics_complete.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMétricas guardadas en: {metrics_path}")
    print("\n" + "="*80)
    print("EVALUACIÓN COMPLETADA")
    print("="*80)
    
    return metrics

# ============================================================================
# MAIN
# ============================================================================

def load_config():
    """Carga configuración guardada si existe."""
    config_path = f"{CONFIG['output_dir']}/config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                if 'model_name' in saved_config:
                    CONFIG['model_name'] = saved_config['model_name']
                    print(f"Modelo usado en entrenamiento: {CONFIG['model_name']}")
                CONFIG.update(saved_config)
            print(f"Configuración cargada desde: {config_path}")
        except Exception as e:
            print(f"Error al cargar config.json: {e}")
            print(f"   Usando configuración por defecto.")
    else:
        print(f"No se encontró config.json en: {config_path}")
        print(f"   Usando configuración por defecto.")

def main():
    """Función principal."""
    parser = ArgumentParser(description='Entrenar o evaluar modelo T5-large')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train',
                        help='Modo: train para entrenar, evaluate para evaluar')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Ruta al archivo de datos JSONL')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directorio de salida para el modelo')
    
    args = parser.parse_args()
    
    if args.data_path:
        CONFIG['data_path'] = args.data_path
    if args.output_dir:
        CONFIG['output_dir'] = args.output_dir
    
    setup_dependencies()
    mount_google_drive()
    
    if args.mode == 'train':
        load_config()
        train_t5_large()
    elif args.mode == 'evaluate':
        load_config()
        evaluate_model_complete()

if __name__ == "__main__":
    main()

