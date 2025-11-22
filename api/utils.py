"""
Utilidades para la API de PLS
- Chunking semántico
- Generación de PLS
- Cálculo de métricas
"""

import torch
import numpy as np
from typing import Tuple, Dict
import textstat
from rouge_score import rouge_scorer
import sacrebleu

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
# Importar configuración centralizada
try:
    from config import (
        TASK_PREFIX,
        MAX_INPUT_LENGTH,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        SEPARATORS,
        DEFAULT_MAX_LENGTH,
        DEFAULT_NUM_BEAMS
    )
except ImportError:
    # Fallback si config.py no está disponible
    TASK_PREFIX = "simplify medical text into plain language: "
    MAX_INPUT_LENGTH = 512
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50
    SEPARATORS = ["\n\n", "\n", ". ", " "]
    DEFAULT_MAX_LENGTH = 256
    DEFAULT_NUM_BEAMS = 4

# ============================================================================
# CHUNKING SEMÁNTICO
# ============================================================================

def setup_chunking(tokenizer):
    """
    Configura el text splitter para chunking semántico
    
    Args:
        tokenizer: Tokenizer de T5
    
    Returns:
        RecursiveCharacterTextSplitter configurado
    """
    # Función para contar tokens
    def length_function(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))
    
    # Configurar splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=length_function,
        separators=SEPARATORS,
        is_separator_regex=False
    )
    
    return text_splitter

def generar_pls_con_chunking(
    texto: str,
    model,
    tokenizer,
    text_splitter,
    device: str = "cuda",
    max_length: int = 256,
    num_beams: int = 4
) -> Tuple[str, int]:
    """
    Genera PLS con manejo automático de chunking
    
    Args:
        texto: Texto técnico de entrada
        model: Modelo T5
        tokenizer: Tokenizer
        text_splitter: RecursiveCharacterTextSplitter
        device: 'cuda' o 'cpu'
        max_length: Longitud máxima del output
        num_beams: Número de beams para generación
    
    Returns:
        (pls_generado, num_chunks)
    """
    # Agregar task prefix
    texto_con_prefix = TASK_PREFIX + texto
    
    # Contar tokens
    tokens = tokenizer.encode(texto_con_prefix, add_special_tokens=False)
    
    # CASO 1: Texto cabe en 512 tokens (procesamiento directo)
    if len(tokens) <= MAX_INPUT_LENGTH:
        inputs = tokenizer(
            texto_con_prefix,
            max_length=MAX_INPUT_LENGTH,
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
        return pls, 1
    
    # CASO 2: Texto requiere chunking
    else:
        chunks = text_splitter.split_text(texto)
        
        chunk_outputs = []
        for chunk in chunks:
            chunk_con_prefix = TASK_PREFIX + chunk
            inputs = tokenizer(
                chunk_con_prefix,
                max_length=MAX_INPUT_LENGTH,
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
        pls_final = chunk_outputs[0]
        
        return pls_final, len(chunks)

# ============================================================================
# CÁLCULO DE MÉTRICAS
# ============================================================================

def calcular_metricas_basicas(source: str, prediction: str) -> Dict:
    """
    Calcula métricas básicas (sin PLS de referencia)
    
    Args:
        source: Texto original
        prediction: PLS generado
    
    Returns:
        Dict con métricas
    """
    metricas = {}
    
    # Legibilidad
    try:
        metricas['flesch_reading_ease'] = textstat.flesch_reading_ease(prediction)
        metricas['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(prediction)
    except:
        metricas['flesch_reading_ease'] = 0.0
        metricas['flesch_kincaid_grade'] = 0.0
    
    # Compresión
    source_words = len(source.split())
    pred_words = len(prediction.split())
    
    metricas['compression_ratio'] = pred_words / source_words if source_words > 0 else 0
    metricas['word_length'] = pred_words
    metricas['original_word_length'] = source_words
    
    return metricas

def calcular_todas_las_metricas(source: str, prediction: str, reference: str) -> Dict:
    """
    Calcula todas las métricas (con PLS de referencia)
    
    Args:
        source: Texto original
        prediction: PLS generado
        reference: PLS de referencia (gold standard)
    
    Returns:
        Dict con todas las métricas
    """
    metricas = {}
    
    # 1-3. ROUGE
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = rouge_scorer_obj.score(reference, prediction)
    
    metricas['rouge1'] = scores['rouge1'].fmeasure
    metricas['rouge2'] = scores['rouge2'].fmeasure
    metricas['rougeL'] = scores['rougeL'].fmeasure
    
    # 4. BLEU
    try:
        bleu = sacrebleu.sentence_bleu(prediction, [reference], smooth_method='exp')
        metricas['bleu'] = bleu.score / 100.0
    except:
        metricas['bleu'] = 0.0
    
    # 5. METEOR (aproximación con F1 de palabras)
    pred_words = set(prediction.lower().split())
    ref_words = set(reference.lower().split())
    
    if len(pred_words) > 0 and len(ref_words) > 0:
        precision = len(pred_words & ref_words) / len(pred_words)
        recall = len(pred_words & ref_words) / len(ref_words)
        if precision + recall > 0:
            metricas['meteor'] = 2 * precision * recall / (precision + recall)
        else:
            metricas['meteor'] = 0.0
    else:
        metricas['meteor'] = 0.0
    
    # 6. BERTScore (opcional, puede ser lento)
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn([prediction], [reference], lang='en', verbose=False)
        metricas['bertscore_f1'] = F1.mean().item()
    except:
        metricas['bertscore_f1'] = None
    
    # 7. SARI (simplificación)
    src_words = set(source.lower().split())
    pred_words_sari = set(prediction.lower().split())
    ref_words_sari = set(reference.lower().split())
    
    # Keep: palabras que están en source y reference, y se mantienen en prediction
    keep = len(src_words & ref_words_sari & pred_words_sari)
    keep_total = len(src_words & ref_words_sari)
    keep_score = keep / keep_total if keep_total > 0 else 0
    
    # Add: palabras nuevas en reference que están en prediction
    add = len((ref_words_sari - src_words) & pred_words_sari)
    add_total = len(ref_words_sari - src_words)
    add_score = add / add_total if add_total > 0 else 0
    
    # Delete: palabras en source que no están en reference ni en prediction
    delete = len(src_words - ref_words_sari - pred_words_sari)
    delete_total = len(src_words - ref_words_sari)
    delete_score = delete / delete_total if delete_total > 0 else 0
    
    # SARI es el promedio de las tres operaciones
    metricas['sari'] = (keep_score + add_score + delete_score) / 3
    
    # 8-9. Legibilidad
    try:
        metricas['flesch_reading_ease'] = textstat.flesch_reading_ease(prediction)
        metricas['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(prediction)
    except:
        metricas['flesch_reading_ease'] = 0.0
        metricas['flesch_kincaid_grade'] = 0.0
    
    # 10-11. Compresión
    source_words_count = len(source.split())
    pred_words_count = len(prediction.split())
    
    metricas['compression_ratio'] = pred_words_count / source_words_count if source_words_count > 0 else 0
    metricas['word_length'] = pred_words_count
    metricas['original_word_length'] = source_words_count
    
    return metricas

# ============================================================================
# CLASIFICACIÓN PLS / NON-PLS
# ============================================================================

# Variables globales para el clasificador
CLASSIFIER = None
VECTORIZER = None

def load_classifier():
    """
    Carga el modelo clasificador entrenado.
    
    Returns:
        (classifier, vectorizer) o (None, None) si no está disponible
    """
    global CLASSIFIER, VECTORIZER
    
    if CLASSIFIER is not None and VECTORIZER is not None:
        return CLASSIFIER, VECTORIZER
    
    try:
        import joblib
        from pathlib import Path
        
        model_dir = Path('models/baseline_classifier')
        
        # Intentar rutas relativas desde api/
        possible_paths = [
            model_dir,  # Desde raíz del proyecto
            Path('../models/baseline_classifier'),  # Desde api/
            Path('../../models/baseline_classifier'),  # Desde subdirectorio
        ]
        
        classifier_path = None
        vectorizer_path = None
        
        for base_path in possible_paths:
            if (base_path / 'classifier.pkl').exists() and (base_path / 'vectorizer.pkl').exists():
                classifier_path = base_path / 'classifier.pkl'
                vectorizer_path = base_path / 'vectorizer.pkl'
                break
        
        if classifier_path is None or vectorizer_path is None:
            return None, None
        
        VECTORIZER = joblib.load(vectorizer_path)
        CLASSIFIER = joblib.load(classifier_path)
        
        return CLASSIFIER, VECTORIZER
        
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return None, None

def clasificar_texto(texto: str) -> Dict:
    """
    Clasifica si un texto es PLS (Plain Language Summary) o texto técnico.
    
    Intenta usar el modelo clasificador entrenado. Si no está disponible,
    usa heurísticas basadas en métricas de legibilidad.
    
    Args:
        texto: Texto a clasificar
    
    Returns:
        Dict con:
            - is_pls: bool
            - confidence: float (0-1)
            - flesch_reading_ease: float
            - flesch_kincaid_grade: float
            - avg_word_length: float
            - technical_terms_count: int
            - reasoning: str
    """
    if not texto or len(texto.strip()) < 50:
        return {
            "is_pls": False,
            "confidence": 0.0,
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "avg_word_length": 0.0,
            "technical_terms_count": 0,
            "reasoning": "Text too short to classify (minimum 50 characters required)"
        }
    
    # Intentar cargar y usar el clasificador entrenado
    clf, vectorizer = load_classifier()
    
    if clf is not None and vectorizer is not None:
        try:
            # Usar modelo entrenado
            X = vectorizer.transform([texto])
            proba = clf.predict_proba(X)[0]
            
            # Probabilidad de ser PLS (clase 1)
            pls_prob = proba[1] if len(proba) > 1 else proba[0]
            is_pls = clf.predict(X)[0] == 'pls'
            
            # Calcular métricas adicionales para el reasoning
            try:
                fre = textstat.flesch_reading_ease(texto)
                fkg = textstat.flesch_kincaid_grade(texto)
            except:
                fre = 0.0
                fkg = 20.0
            
            words = texto.split()
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            # Contar términos técnicos
            technical_indicators = [
                'randomized', 'controlled', 'trial', 'statistically', 'significant',
                'methodology', 'intervention', 'efficacy', 'placebo', 'cohort',
                'retrospective', 'prospective', 'multivariate', 'regression',
                'confidence interval', 'p-value', 'hypothesis', 'protocol'
            ]
            text_lower = texto.lower()
            technical_count = sum(1 for term in technical_indicators if term in text_lower)
            
            # Create user-friendly reasoning
            prob_percent = pls_prob * 100
            if pls_prob >= 0.7:
                prob_desc = "high"
            elif pls_prob >= 0.5:
                prob_desc = "moderate"
            else:
                prob_desc = "low"
            
            if fre >= 60:
                readability_desc = "easy to read"
            elif fre >= 50:
                readability_desc = "moderately readable"
            elif fre >= 30:
                readability_desc = "difficult to read"
            elif fre >= 0:
                readability_desc = "very difficult to read"
            else:
                readability_desc = "extremely difficult to read"
            
            if fkg <= 8:
                grade_desc = f"grade {fkg:.0f} level (easy)"
            elif fkg <= 12:
                grade_desc = f"grade {fkg:.0f} level (moderate)"
            else:
                grade_desc = f"grade {fkg:.0f} level (difficult)"
            
            # Format FRE score appropriately
            if fre < 0:
                fre_display = f"{fre:.0f} (negative scores indicate extremely complex text)"
            else:
                fre_display = f"{fre:.0f}"
            
            reasoning = f"AI model analysis: {prob_percent:.0f}% probability this is plain language ({prob_desc} confidence). " \
                       f"Text is {readability_desc} (reading ease score: {fre_display}) and written at {grade_desc}."
            
            return {
                "is_pls": bool(is_pls),
                "confidence": round(float(pls_prob), 3),
                "flesch_reading_ease": round(fre, 1),
                "flesch_kincaid_grade": round(fkg, 1),
                "avg_word_length": round(avg_word_length, 2),
                "technical_terms_count": technical_count,
                "reasoning": reasoning
            }
        except Exception as e:
            # Si falla el modelo, continuar con heurísticas
            print(f"Error using trained classifier: {e}, falling back to heuristics")
    
    # Fallback: usar heurísticas si el modelo no está disponible
    try:
        fre = textstat.flesch_reading_ease(texto)
        fkg = textstat.flesch_kincaid_grade(texto)
    except:
        fre = 0.0
        fkg = 20.0
    
    words = texto.split()
    if len(words) > 0:
        avg_word_length = sum(len(word) for word in words) / len(words)
    else:
        avg_word_length = 0
    
    technical_indicators = [
        'randomized', 'controlled', 'trial', 'statistically', 'significant',
        'methodology', 'intervention', 'efficacy', 'placebo', 'cohort',
        'retrospective', 'prospective', 'multivariate', 'regression',
        'confidence interval', 'p-value', 'hypothesis', 'protocol'
    ]
    
    text_lower = texto.lower()
    technical_count = sum(1 for term in technical_indicators if term in text_lower)
    technical_ratio = technical_count / max(len(words), 1) * 100
    
    # Heurísticas para determinar si es PLS
    pls_score = 0.0
    reasoning_parts = []
    
    # Build user-friendly reasoning
    reasoning_parts = []
    
    if fre >= 60:
        pls_score += 0.3
        reasoning_parts.append(f"Easy to read (score: {fre:.0f})")
    elif fre >= 50:
        pls_score += 0.15
        reasoning_parts.append(f"Moderately readable (score: {fre:.0f})")
    elif fre >= 30:
        reasoning_parts.append(f"Difficult to read (score: {fre:.0f})")
    elif fre >= 0:
        reasoning_parts.append(f"Very difficult to read (score: {fre:.0f})")
    else:
        reasoning_parts.append(f"Extremely difficult to read (score: {fre:.0f}, negative scores indicate very complex text)")
    
    if fkg <= 8:
        pls_score += 0.3
        reasoning_parts.append(f"Written at grade {fkg:.0f} level (easy to understand)")
    elif fkg <= 12:
        pls_score += 0.15
        reasoning_parts.append(f"Written at grade {fkg:.0f} level (moderate difficulty)")
    else:
        reasoning_parts.append(f"Written at grade {fkg:.0f} level (difficult to understand)")
    
    if avg_word_length <= 4.5:
        pls_score += 0.2
        reasoning_parts.append(f"Uses short, simple words (avg {avg_word_length:.1f} characters)")
    elif avg_word_length <= 5.5:
        pls_score += 0.1
        reasoning_parts.append(f"Uses moderately sized words (avg {avg_word_length:.1f} characters)")
    else:
        reasoning_parts.append(f"Uses long, complex words (avg {avg_word_length:.1f} characters)")
    
    if technical_ratio < 2:
        pls_score += 0.2
        if technical_count == 0:
            reasoning_parts.append(f"No technical terms found")
        else:
            reasoning_parts.append(f"Very few technical terms ({technical_count} found)")
    elif technical_ratio < 5:
        pls_score += 0.1
        reasoning_parts.append(f"Some technical terms present ({technical_count} found)")
    else:
        reasoning_parts.append(f"Many technical terms present ({technical_count} found)")
    
    confidence = min(pls_score, 1.0)
    is_pls = confidence >= 0.5
    
    # Create natural language reasoning
    if len(reasoning_parts) > 0:
        reasoning = "Analysis based on text characteristics: " + ". ".join(reasoning_parts) + "."
    else:
        reasoning = "Analysis based on text characteristics."
    
    return {
        "is_pls": is_pls,
        "confidence": round(confidence, 3),
        "flesch_reading_ease": round(fre, 1),
        "flesch_kincaid_grade": round(fkg, 1),
        "avg_word_length": round(avg_word_length, 2),
        "technical_terms_count": technical_count,
        "reasoning": reasoning
    }

