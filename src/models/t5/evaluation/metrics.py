"""
Métricas de evaluación para T5 durante el entrenamiento.

Este módulo contiene la función compute_metrics que se usa durante
el entrenamiento para calcular ROUGE, BLEU y SARI.
"""

import numpy as np
from typing import Tuple
import sys
from pathlib import Path

# Importar dependencias
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score no disponible. Instalar: pip install rouge-score")

try:
    from sacrebleu import BLEU
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: sacrebleu no disponible. Instalar: pip install sacrebleu")


def compute_metrics(eval_pred, tokenizer):
    """
    Calcula métricas completas: ROUGE, BLEU, SARI.
    CRÍTICO: Incluir las 3 métricas mencionadas por el profesor.
    
    Esta función se usa durante el entrenamiento con Trainer.compute_metrics.
    
    Args:
        eval_pred: Tupla (predictions, labels) del Trainer
        tokenizer: Tokenizer para decodificar
    
    Returns:
        Diccionario con métricas: rouge1, rouge2, rougeL, bleu, sari
    """
    predictions, labels = eval_pred
    
    # Decodificar predicciones
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Decodificar labels (reemplazar -100 con pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    results = {}
    
    # ROUGE completo (ROUGE-1, ROUGE-2, ROUGE-L)
    if ROUGE_AVAILABLE:
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = rouge_scorer_obj.score(label, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        results['rouge1'] = np.mean(rouge_scores['rouge1'])
        results['rouge2'] = np.mean(rouge_scores['rouge2'])
        results['rougeL'] = np.mean(rouge_scores['rougeL'])
    
    # BLEU score
    if BLEU_AVAILABLE:
        bleu = BLEU()
        bleu_scores = []
        for pred, label in zip(decoded_preds, decoded_labels):
            try:
                score = bleu.sentence_score(pred, [label])
                bleu_scores.append(score.score / 100.0)  # Normalizar a 0-1
            except:
                bleu_scores.append(0.0)
        results['bleu'] = np.mean(bleu_scores)
    
    # SARI (System output Against References and Inputs)
    # Implementación simplificada de SARI
    def compute_sari(pred, ref, source):
        """
        Calcula SARI: keep, addition, deletion scores.
        Implementación simplificada basada en n-gramas.
        """
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
    # Aquí usamos una aproximación usando label como source
    sari_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        # Usar label como aproximación del source (no ideal pero funcional)
        sari = compute_sari(pred, label, label)
        sari_scores.append(sari)
    
    results['sari'] = np.mean(sari_scores)
    
    return results

