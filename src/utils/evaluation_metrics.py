"""
Módulo completo de métricas de evaluación para simplificación de texto.

Incluye:
- ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
- BLEU
- SARI (System output Against References and Inputs)
- BERTScore (best score)
- Métricas de legibilidad (py-readability-metrics)
"""

from typing import List, Dict, Optional, Tuple
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# IMPORTS CONDICIONALES
# ============================================================================

# ROUGE
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score no disponible. Instalar: pip install rouge-score")

# BLEU
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    # Descargar datos necesarios de NLTK
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: nltk no disponible. Instalar: pip install nltk")

# SARI
try:
    # SARI se implementa manualmente ya que no hay librería estándar
    SARI_AVAILABLE = True
except:
    SARI_AVAILABLE = False

# BERTScore
try:
    from bert_score import score as bert_score_func
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert-score no disponible. Instalar: pip install bert-score")

# py-readability-metrics
try:
    from readability import Readability
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    print("Warning: py-readability-metrics no disponible. Instalar: pip install py-readability-metrics")

# ============================================================================
# ROUGE
# ============================================================================

def calculate_rouge(predictions: List[str], references: List[str]) -> Optional[Dict]:
    """
    Calcula métricas ROUGE completas (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Args:
        predictions: Lista de textos generados
        references: Lista de textos de referencia
    
    Returns:
        Diccionario con métricas ROUGE o None si no está disponible
    """
    if not ROUGE_AVAILABLE:
        return None
    
    print("Calculando ROUGE...")
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
        use_stemmer=True
    )
    
    rouge_scores = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="ROUGE"):
        scores = scorer.score(ref, pred)
        rouge_scores.append({
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'fmeasure': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'fmeasure': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'fmeasure': scores['rougeL'].fmeasure
            },
            'rougeLsum': {
                'precision': scores['rougeLsum'].precision,
                'recall': scores['rougeLsum'].recall,
                'fmeasure': scores['rougeLsum'].fmeasure
            }
        })
    
    # Calcular promedios
    n = len(rouge_scores)
    if n == 0:
        return None
    
    avg_rouge1 = {
        'precision': sum([r['rouge1']['precision'] for r in rouge_scores]) / n,
        'recall': sum([r['rouge1']['recall'] for r in rouge_scores]) / n,
        'fmeasure': sum([r['rouge1']['fmeasure'] for r in rouge_scores]) / n
    }
    
    avg_rouge2 = {
        'precision': sum([r['rouge2']['precision'] for r in rouge_scores]) / n,
        'recall': sum([r['rouge2']['recall'] for r in rouge_scores]) / n,
        'fmeasure': sum([r['rouge2']['fmeasure'] for r in rouge_scores]) / n
    }
    
    avg_rougeL = {
        'precision': sum([r['rougeL']['precision'] for r in rouge_scores]) / n,
        'recall': sum([r['rougeL']['recall'] for r in rouge_scores]) / n,
        'fmeasure': sum([r['rougeL']['fmeasure'] for r in rouge_scores]) / n
    }
    
    avg_rougeLsum = {
        'precision': sum([r['rougeLsum']['precision'] for r in rouge_scores]) / n,
        'recall': sum([r['rougeLsum']['recall'] for r in rouge_scores]) / n,
        'fmeasure': sum([r['rougeLsum']['fmeasure'] for r in rouge_scores]) / n
    }
    
    return {
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'rougeLsum': avg_rougeLsum,
        'individual_scores': rouge_scores
    }

# ============================================================================
# BLEU
# ============================================================================

def calculate_bleu(predictions: List[str], references: List[str]) -> Optional[Dict]:
    """
    Calcula métricas BLEU (BLEU-1, BLEU-2, BLEU-3, BLEU-4).
    
    Args:
        predictions: Lista de textos generados
        references: Lista de textos de referencia
    
    Returns:
        Diccionario con métricas BLEU o None si no está disponible
    """
    if not BLEU_AVAILABLE:
        return None
    
    print("Calculando BLEU...")
    smoothing = SmoothingFunction().method1
    
    bleu_scores = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="BLEU"):
        try:
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())
            
            # BLEU-1 a BLEU-4
            bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu4 = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
            
            bleu_scores.append({
                'bleu1': bleu1,
                'bleu2': bleu2,
                'bleu3': bleu3,
                'bleu4': bleu4
            })
        except Exception as e:
            # Si hay error, usar valores por defecto
            bleu_scores.append({
                'bleu1': 0.0,
                'bleu2': 0.0,
                'bleu3': 0.0,
                'bleu4': 0.0
            })
    
    n = len(bleu_scores)
    if n == 0:
        return None
    
    return {
        'bleu1': sum([b['bleu1'] for b in bleu_scores]) / n,
        'bleu2': sum([b['bleu2'] for b in bleu_scores]) / n,
        'bleu3': sum([b['bleu3'] for b in bleu_scores]) / n,
        'bleu4': sum([b['bleu4'] for b in bleu_scores]) / n,
        'individual_scores': bleu_scores
    }

# ============================================================================
# SARI
# ============================================================================

def _get_ngrams(tokens: List[str], n: int) -> Dict[str, int]:
    """Obtiene n-gramas de una lista de tokens."""
    ngrams = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    return ngrams

def _sari_operation_score(
    sys_output: List[str],
    sources: List[str],
    references: List[List[str]]
) -> float:
    """
    Calcula el score de operaciones SARI.
    
    SARI evalúa:
    - Keep: n-gramas que están en source y en output (correctos)
    - Add: n-gramas que están en output y references pero no en source
    - Delete: n-gramas que están en source pero no en output ni references
    """
    keep_scores = []
    add_scores = []
    delete_scores = []
    
    for n in [1, 2, 3, 4]:
        sys_ngrams = _get_ngrams(sys_output, n)
        source_ngrams = _get_ngrams(sources, n)
        
        # Keep: n-gramas en source y sys_output
        keep_count = sum(min(sys_ngrams.get(ng, 0), source_ngrams.get(ng, 0)) 
                        for ng in set(sys_ngrams.keys()) & set(source_ngrams.keys()))
        keep_total = sum(sys_ngrams.values())
        keep_score = keep_count / keep_total if keep_total > 0 else 0.0
        
        # Add: n-gramas en sys_output y references pero no en source
        ref_ngrams_union = {}
        for ref in references:
            ref_ngrams = _get_ngrams(ref, n)
            for ng, count in ref_ngrams.items():
                ref_ngrams_union[ng] = max(ref_ngrams_union.get(ng, 0), count)
        
        add_count = 0
        for ng in sys_ngrams:
            if ng not in source_ngrams and ng in ref_ngrams_union:
                add_count += min(sys_ngrams[ng], ref_ngrams_union[ng])
        add_total = sum(sys_ngrams.values())
        add_score = add_count / add_total if add_total > 0 else 0.0
        
        # Delete: n-gramas en source pero no en sys_output ni references
        delete_count = 0
        for ng in source_ngrams:
            if ng not in sys_ngrams and ng not in ref_ngrams_union:
                delete_count += source_ngrams[ng]
        delete_total = sum(source_ngrams.values())
        delete_score = delete_count / delete_total if delete_total > 0 else 0.0
        
        keep_scores.append(keep_score)
        add_scores.append(add_score)
        delete_scores.append(delete_score)
    
    # Promedio de n-gramas
    avg_keep = sum(keep_scores) / len(keep_scores)
    avg_add = sum(add_scores) / len(add_scores)
    avg_delete = sum(delete_scores) / len(delete_scores)
    
    # SARI es el promedio de las tres operaciones
    sari_score = (avg_keep + avg_add + avg_delete) / 3.0
    
    return sari_score, avg_keep, avg_add, avg_delete

def calculate_sari(
    predictions: List[str],
    references: List[str],
    sources: Optional[List[str]] = None
) -> Optional[Dict]:
    """
    Calcula métrica SARI (System output Against References and Inputs).
    
    SARI evalúa simplificación de texto considerando:
    - Keep: palabras que se mantienen correctamente
    - Add: palabras simples que se agregan correctamente
    - Delete: palabras complejas que se eliminan correctamente
    
    Args:
        predictions: Lista de textos generados (simplificados)
        references: Lista de textos de referencia (simplificados)
        sources: Lista de textos originales (técnicos). Si es None, se usa references como source.
    
    Returns:
        Diccionario con métrica SARI o None
    """
    if not SARI_AVAILABLE:
        return None
    
    print("Calculando SARI...")
    
    # Si no hay sources, usar references como source (para casos donde no tenemos el original)
    if sources is None:
        sources = references
    
    if len(predictions) != len(references) or len(predictions) != len(sources):
        print("Error: Las listas deben tener la misma longitud")
        return None
    
    sari_scores = []
    for pred, ref, src in tqdm(zip(predictions, references, sources), total=len(predictions), desc="SARI"):
        try:
            # Tokenizar
            if BLEU_AVAILABLE:
                pred_tokens = word_tokenize(pred.lower())
                ref_tokens = word_tokenize(ref.lower())
                src_tokens = word_tokenize(src.lower())
            else:
                pred_tokens = pred.lower().split()
                ref_tokens = ref.lower().split()
                src_tokens = src.lower().split()
            
            sari_score, keep, add, delete = _sari_operation_score(
                pred_tokens,
                src_tokens,
                [ref_tokens]
            )
            
            sari_scores.append({
                'sari': sari_score,
                'keep': keep,
                'add': add,
                'delete': delete
            })
        except Exception as e:
            sari_scores.append({
                'sari': 0.0,
                'keep': 0.0,
                'add': 0.0,
                'delete': 0.0
            })
    
    n = len(sari_scores)
    if n == 0:
        return None
    
    return {
        'sari': sum([s['sari'] for s in sari_scores]) / n,
        'keep': sum([s['keep'] for s in sari_scores]) / n,
        'add': sum([s['add'] for s in sari_scores]) / n,
        'delete': sum([s['delete'] for s in sari_scores]) / n,
        'individual_scores': sari_scores
    }

# ============================================================================
# BERTScore (Best Score)
# ============================================================================

def calculate_bertscore(predictions: List[str], references: List[str]) -> Optional[Dict]:
    """
    Calcula BERTScore (mejor métrica semántica, también conocida como "best score").
    
    BERTScore evalúa la similitud semántica usando embeddings contextuales de BERT.
    
    Args:
        predictions: Lista de textos generados
        references: Lista de textos de referencia
    
    Returns:
        Diccionario con métricas BERTScore o None
    """
    if not BERTSCORE_AVAILABLE:
        return None
    
    print("Calculando BERTScore (best score)...")
    
    try:
        P, R, F1 = bert_score_func(predictions, references, lang='en', verbose=True)
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item(),
            'precision_scores': P.tolist(),
            'recall_scores': R.tolist(),
            'f1_scores': F1.tolist()
        }
    except Exception as e:
        print(f"Error calculando BERTScore: {e}")
        return None

# ============================================================================
# MÉTRICAS DE LEGIBILIDAD (py-readability-metrics)
# ============================================================================

def calculate_readability_metrics(text: str) -> Optional[Dict]:
    """
    Calcula métricas de legibilidad usando py-readability-metrics.
    
    Args:
        text: Texto a evaluar
    
    Returns:
        Diccionario con métricas de legibilidad o None
    """
    if not READABILITY_AVAILABLE:
        return None
    
    try:
        r = Readability(text)
        metrics = {}
        
        # Flesch-Kincaid Grade Level
        try:
            fk = r.flesch_kincaid()
            metrics['flesch_kincaid'] = {
                'score': fk.score,
                'grade_level': fk.grade_level
            }
        except:
            pass
        
        # Flesch Reading Ease
        try:
            fre = r.flesch()
            metrics['flesch_reading_ease'] = fre.score
        except:
            pass
        
        # Gunning Fog Index
        try:
            gfi = r.gunning_fog()
            metrics['gunning_fog'] = {
                'score': gfi.score,
                'grade_level': gfi.grade_level
            }
        except:
            pass
        
        # Dale-Chall Readability
        try:
            dc = r.dale_chall()
            metrics['dale_chall'] = {
                'score': dc.score,
                'grade_level': dc.grade_level
            }
        except:
            pass
        
        # Automated Readability Index (ARI)
        try:
            ari = r.ari()
            metrics['ari'] = {
                'score': ari.score,
                'grade_level': ari.grade_level
            }
        except:
            pass
        
        # Coleman-Liau Index
        try:
            cli = r.coleman_liau()
            metrics['coleman_liau'] = {
                'score': cli.score,
                'grade_level': cli.grade_level
            }
        except:
            pass
        
        # SMOG Index
        try:
            smog = r.smog()
            metrics['smog'] = {
                'score': smog.score,
                'grade_level': smog.grade_level
            }
        except:
            pass
        
        # Linsear Write Formula
        try:
            lwf = r.linsear_write()
            metrics['linsear_write'] = {
                'score': lwf.score,
                'grade_level': lwf.grade_level
            }
        except:
            pass
        
        return metrics if metrics else None
        
    except Exception as e:
        return None

def calculate_readability_batch(texts: List[str]) -> Optional[Dict]:
    """
    Calcula métricas de legibilidad para un batch de textos.
    
    Args:
        texts: Lista de textos a evaluar
    
    Returns:
        Diccionario con métricas promedio o None
    """
    if not READABILITY_AVAILABLE:
        return None
    
    print("Calculando métricas de legibilidad...")
    
    all_metrics = []
    for text in tqdm(texts, desc="Readability"):
        metrics = calculate_readability_metrics(text)
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        return None
    
    # Calcular promedios
    avg_metrics = {}
    
    # Flesch-Kincaid
    fk_scores = [m.get('flesch_kincaid', {}).get('score', 0) for m in all_metrics if 'flesch_kincaid' in m]
    if fk_scores:
        avg_metrics['flesch_kincaid'] = {
            'avg_score': sum(fk_scores) / len(fk_scores),
            'avg_grade_level': sum([m.get('flesch_kincaid', {}).get('grade_level', 0) 
                                   for m in all_metrics if 'flesch_kincaid' in m]) / len(fk_scores)
        }
    
    # Flesch Reading Ease
    fre_scores = [m.get('flesch_reading_ease', 0) for m in all_metrics if 'flesch_reading_ease' in m]
    if fre_scores:
        avg_metrics['flesch_reading_ease'] = sum(fre_scores) / len(fre_scores)
    
    # Gunning Fog
    gfi_scores = [m.get('gunning_fog', {}).get('score', 0) for m in all_metrics if 'gunning_fog' in m]
    if gfi_scores:
        avg_metrics['gunning_fog'] = {
            'avg_score': sum(gfi_scores) / len(gfi_scores),
            'avg_grade_level': sum([m.get('gunning_fog', {}).get('grade_level', 0) 
                                   for m in all_metrics if 'gunning_fog' in m]) / len(gfi_scores)
        }
    
    # Dale-Chall
    dc_scores = [m.get('dale_chall', {}).get('score', 0) for m in all_metrics if 'dale_chall' in m]
    if dc_scores:
        avg_metrics['dale_chall'] = {
            'avg_score': sum(dc_scores) / len(dc_scores),
            'avg_grade_level': sum([m.get('dale_chall', {}).get('grade_level', 0) 
                                   for m in all_metrics if 'dale_chall' in m]) / len(dc_scores)
        }
    
    # ARI
    ari_scores = [m.get('ari', {}).get('score', 0) for m in all_metrics if 'ari' in m]
    if ari_scores:
        avg_metrics['ari'] = {
            'avg_score': sum(ari_scores) / len(ari_scores),
            'avg_grade_level': sum([m.get('ari', {}).get('grade_level', 0) 
                                   for m in all_metrics if 'ari' in m]) / len(ari_scores)
        }
    
    # Coleman-Liau
    cli_scores = [m.get('coleman_liau', {}).get('score', 0) for m in all_metrics if 'coleman_liau' in m]
    if cli_scores:
        avg_metrics['coleman_liau'] = {
            'avg_score': sum(cli_scores) / len(cli_scores),
            'avg_grade_level': sum([m.get('coleman_liau', {}).get('grade_level', 0) 
                                   for m in all_metrics if 'coleman_liau' in m]) / len(cli_scores)
        }
    
    # SMOG
    smog_scores = [m.get('smog', {}).get('score', 0) for m in all_metrics if 'smog' in m]
    if smog_scores:
        avg_metrics['smog'] = {
            'avg_score': sum(smog_scores) / len(smog_scores),
            'avg_grade_level': sum([m.get('smog', {}).get('grade_level', 0) 
                                   for m in all_metrics if 'smog' in m]) / len(smog_scores)
        }
    
    # Linsear Write
    lwf_scores = [m.get('linsear_write', {}).get('score', 0) for m in all_metrics if 'linsear_write' in m]
    if lwf_scores:
        avg_metrics['linsear_write'] = {
            'avg_score': sum(lwf_scores) / len(lwf_scores),
            'avg_grade_level': sum([m.get('linsear_write', {}).get('grade_level', 0) 
                                   for m in all_metrics if 'linsear_write' in m]) / len(lwf_scores)
        }
    
    avg_metrics['individual_scores'] = all_metrics
    
    return avg_metrics

# ============================================================================
# FUNCIÓN PRINCIPAL: CALCULAR TODAS LAS MÉTRICAS
# ============================================================================

def calculate_all_metrics(
    predictions: List[str],
    references: List[str],
    sources: Optional[List[str]] = None,
    include_readability: bool = True
) -> Dict:
    """
    Calcula todas las métricas disponibles.
    
    Args:
        predictions: Lista de textos generados
        references: Lista de textos de referencia
        sources: Lista de textos originales (para SARI). Si es None, se usa references.
        include_readability: Si True, incluye métricas de legibilidad
    
    Returns:
        Diccionario con todas las métricas calculadas
    """
    results = {}
    
    # ROUGE
    rouge_results = calculate_rouge(predictions, references)
    if rouge_results:
        results['rouge'] = {
            'rouge1_f': rouge_results['rouge1']['fmeasure'],
            'rouge2_f': rouge_results['rouge2']['fmeasure'],
            'rougeL_f': rouge_results['rougeL']['fmeasure'],
            'rougeLsum_f': rouge_results['rougeLsum']['fmeasure'],
            'full': rouge_results
        }
    
    # BLEU
    bleu_results = calculate_bleu(predictions, references)
    if bleu_results:
        results['bleu'] = {
            'bleu1': bleu_results['bleu1'],
            'bleu2': bleu_results['bleu2'],
            'bleu3': bleu_results['bleu3'],
            'bleu4': bleu_results['bleu4'],
            'full': bleu_results
        }
    
    # SARI
    sari_results = calculate_sari(predictions, references, sources)
    if sari_results:
        results['sari'] = {
            'sari': sari_results['sari'],
            'keep': sari_results['keep'],
            'add': sari_results['add'],
            'delete': sari_results['delete'],
            'full': sari_results
        }
    
    # BERTScore (Best Score)
    bertscore_results = calculate_bertscore(predictions, references)
    if bertscore_results:
        results['bertscore'] = {
            'precision': bertscore_results['precision'],
            'recall': bertscore_results['recall'],
            'f1': bertscore_results['f1'],
            'full': bertscore_results
        }
    
    # Readability
    if include_readability:
        readability_results = calculate_readability_batch(predictions)
        if readability_results:
            results['readability'] = readability_results
    
    return results

