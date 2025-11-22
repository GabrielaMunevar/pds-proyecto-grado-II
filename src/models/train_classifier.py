"""
Clasificador Baseline MEJORADO para PLS/non-PLS
TF-IDF + Logistic Regression + Features de Legibilidad

MEJORAS IMPLEMENTADAS:
 Preparación CORRECTA de datos según label (PLS→resumen, non_PLS→texto_original)
 Cross-validation 5-fold estratificado
 Evaluación detallada con métricas por clase
 Análisis de errores (FP/FN)
 Gate function mejorada con probabilidades y umbral ajustable
 ROC-AUC y métricas comprehensivas
 Eliminación de función focal_loss vacía (honestidad académica)

Uso:
    python src/models/train_classifier.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_auc_score, roc_curve, precision_recall_fscore_support,
    accuracy_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, csr_matrix
import joblib
from pathlib import Path
import json
import warnings
import re

# Importar textstat para métricas de legibilidad
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    warnings.warn("textstat no disponible. Features de legibilidad deshabilitadas.")

warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE ENGINEERING: Legibilidad y Términos Técnicos
# ============================================================================

# Diccionario de términos técnicos médicos comunes
MEDICAL_TECHNICAL_TERMS = [
    # Métodos de estudio
    'randomized controlled trial', 'rct', 'placebo', 'placebo-controlled',
    'double-blind', 'single-blind', 'cohort study', 'case-control',
    'retrospective', 'prospective', 'longitudinal', 'cross-sectional',
    'systematic review', 'meta-analysis', 'observational study',
    
    # Términos estadísticos
    'statistically significant', 'p-value', 'p <', 'p>', 'confidence interval',
    'ci', '95% ci', 'odds ratio', 'or', 'relative risk', 'rr', 'hazard ratio',
    'hr', 'multivariate', 'univariate', 'regression', 'logistic regression',
    'cox regression', 'kaplan-meier', 'survival analysis',
    
    # Términos metodológicos
    'efficacy', 'effectiveness', 'intervention', 'methodology', 'protocol',
    'endpoint', 'primary endpoint', 'secondary endpoint', 'outcome measure',
    'adverse event', 'side effect', 'contraindication', 'indication',
    
    # Términos técnicos médicos
    'pathophysiology', 'etiology', 'pathogenesis', 'diagnosis', 'prognosis',
    'morbidity', 'mortality', 'incidence', 'prevalence', 'epidemiology',
    'pathology', 'histology', 'biomarker', 'biomarkers',
    
    # Acrónimos comunes
    'rct', 'ci', 'or', 'rr', 'hr', 'pcr', 'ct', 'mri', 'ecg', 'eeg',
    
    # Patrones estadísticos
    r'\bp\s*[<>=]\s*0\.\d+',  # p-values
    r'\d+%\s*ci',  # Confidence intervals
    r'\d+\.\d+\s*\([^)]+\)',  # Estadísticas con intervalos
]

def extract_readability_features(text):
    """
    Extrae features de legibilidad que distinguen PLS de texto técnico.
    
    PLS típicamente tiene:
    - Mayor Flesch Reading Ease (más fácil de leer)
    - Menor Flesch-Kincaid Grade Level (nivel escolar más bajo)
    - Palabras más cortas
    - Oraciones más cortas
    - Menos palabras complejas
    
    Args:
        text: Texto a analizar
        
    Returns:
        dict: Diccionario con features de legibilidad
    """
    features = {}
    
    if not TEXTSTAT_AVAILABLE or not text or len(text.strip()) < 10:
        # Valores por defecto si textstat no está disponible o texto muy corto
        return {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 20.0,
            'gunning_fog': 20.0,
            'avg_word_length': 0.0,
            'avg_sentence_length': 0.0,
            'complex_words_ratio': 0.0,
            'syllables_per_word': 0.0,
        }
    
    try:
        # Métricas básicas de legibilidad
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        features['gunning_fog'] = textstat.gunning_fog(text)
        
        # Estadísticas de texto
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if words:
            features['avg_word_length'] = np.mean([len(w) for w in words])
            # Palabras complejas: >10 caracteres o >3 sílabas (aproximación)
            complex_words = [w for w in words if len(w) > 10]
            features['complex_words_ratio'] = len(complex_words) / len(words)
            
            # Sílabas promedio por palabra (aproximación)
            total_syllables = sum(textstat.syllable_count(w) for w in words[:100])  # Limitar para velocidad
            features['syllables_per_word'] = total_syllables / min(len(words), 100)
        else:
            features['avg_word_length'] = 0.0
            features['complex_words_ratio'] = 0.0
            features['syllables_per_word'] = 0.0
        
        if sentences and words:
            features['avg_sentence_length'] = len(words) / len(sentences)
        else:
            features['avg_sentence_length'] = 0.0
            
    except Exception as e:
        # En caso de error, usar valores por defecto
        warnings.warn(f"Error calculando legibilidad: {e}")
        features = {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 20.0,
            'gunning_fog': 20.0,
            'avg_word_length': 0.0,
            'avg_sentence_length': 0.0,
            'complex_words_ratio': 0.0,
            'syllables_per_word': 0.0,
        }
    
    return features

def count_technical_terms(text):
    """
    Cuenta la densidad de términos técnicos médicos en el texto.
    
    Texto técnico típicamente tiene mayor densidad de estos términos.
    
    Args:
        text: Texto a analizar
        
    Returns:
        dict: Diccionario con conteos de términos técnicos
    """
    if not text:
        return {
            'technical_terms_count': 0,
            'technical_density': 0.0,
            'statistical_patterns_count': 0,
            'acronyms_count': 0,
        }
    
    text_lower = text.lower()
    words = text.split()
    n_words = len(words) if words else 1
    
    # Contar términos técnicos exactos
    technical_count = 0
    for term in MEDICAL_TECHNICAL_TERMS:
        if isinstance(term, str) and len(term) > 2:  # Solo strings, no regex aún
            if term in text_lower:
                technical_count += text_lower.count(term)
    
    # Contar patrones estadísticos (regex)
    statistical_patterns = [
        r'\bp\s*[<>=]\s*0\.\d+',  # p-values
        r'\d+%\s*ci',  # Confidence intervals
        r'\d+\.\d+\s*\([^)]+\)',  # Estadísticas con intervalos
    ]
    stats_count = sum(len(re.findall(pattern, text_lower)) for pattern in statistical_patterns)
    
    # Contar acrónimos comunes (palabras en mayúsculas de 2-5 letras)
    acronyms = re.findall(r'\b[A-Z]{2,5}\b', text)
    acronyms_count = len(acronyms)
    
    # Densidad: términos técnicos por 100 palabras
    technical_density = (technical_count / n_words) * 100 if n_words > 0 else 0.0
    
    return {
        'technical_terms_count': technical_count,
        'technical_density': technical_density,
        'statistical_patterns_count': stats_count,
        'acronyms_count': acronyms_count,
    }

class ReadabilityFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer de sklearn para extraer features de legibilidad y términos técnicos.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Transforma textos en matriz de features de legibilidad.
        
        Args:
            X: Lista o Series de textos
            
        Returns:
            scipy.sparse.csr_matrix: Matriz sparse con features
        """
        features_list = []
        
        for text in X:
            # Features de legibilidad
            readability = extract_readability_features(str(text))
            
            # Features de términos técnicos
            technical = count_technical_terms(str(text))
            
            # Combinar todos los features en un array
            feature_vector = [
                readability['flesch_reading_ease'],
                readability['flesch_kincaid_grade'],
                readability['gunning_fog'],
                readability['avg_word_length'],
                readability['avg_sentence_length'],
                readability['complex_words_ratio'],
                readability['syllables_per_word'],
                technical['technical_terms_count'],
                technical['technical_density'],
                technical['statistical_patterns_count'],
                technical['acronyms_count'],
            ]
            
            features_list.append(feature_vector)
        
        # Convertir a numpy array y luego a sparse matrix
        features_array = np.array(features_list)
        return csr_matrix(features_array)

def load_data():
    """Carga los datos de entrenamiento y prueba."""
    print("="*80)
    print("CARGANDO DATOS")
    print("="*80)
    
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    print(f"\nTrain total: {len(train_df):,} registros")
    print(f"Test total:  {len(test_df):,} registros")
    
    # Filtrar solo registros con label (excluir unlabeled)
    train_labeled = train_df[train_df['label'].notna()].copy()
    test_labeled = test_df[test_df['label'].notna()].copy()
    
    print(f"\nTrain con label: {len(train_labeled):,} registros")
    print(f"Test con label:  {len(test_labeled):,} registros")
    
    # Distribución de clases
    print(f"\nDistribución Train:")
    print(train_labeled['label'].value_counts())
    
    print(f"\nDistribución Test:")
    print(test_labeled['label'].value_counts())
    
    return train_labeled, test_labeled

def prepare_text_data(df):
    """
    Prepara textos CORRECTAMENTE según label.
    
     CRÍTICO - CORRECCIÓN DEL ERROR ORIGINAL:
    - label='pls' → usar columna 'resumen' (el PLS real)
    - label='non_pls' → usar columna 'texto_original' (texto técnico)
    
    El clasificador debe aprender a distinguir ESTILOS diferentes.
    Mezclar columnas según disponibilidad (como en versión original) 
    confunde al modelo y degrada el performance.
    """
    texts = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        text = None
        
        if row['label'] == 'pls':
            # PLS: DEBE usar resumen (el texto simplificado)
            if pd.notna(row['resumen']):
                text = str(row['resumen']).strip()
                
        elif row['label'] == 'non_pls':
            # Técnico: DEBE usar texto_original (texto complejo)
            if pd.notna(row['texto_original']):
                text = str(row['texto_original']).strip()
        
        # Filtrar textos muy cortos (mínimo 50 caracteres para calidad)
        if text and len(text) >= 50:
            texts.append(text)
            valid_indices.append(idx)
    
    print(f"\n Textos válidos extraídos: {len(texts):,}/{len(df):,}")
    
    # Verificar distribución por label
    labels_extracted = df.loc[valid_indices, 'label']
    print(f"   Distribución extraída:")
    for label, count in labels_extracted.value_counts().items():
        print(f"     {label}: {count:,}")
    
    return pd.Series(texts, index=valid_indices)

def cross_validate_classifier(train_texts, train_labels, use_readability_features=True):
    """
    Cross-validation 5-fold estratificado para verificar estabilidad.
    
    MEJORADO: Ahora incluye features de legibilidad y términos técnicos.
    """
    print("\n" + "="*80)
    print("CROSS-VALIDATION (5-FOLD ESTRATIFICADO)")
    print("="*80)
    
    if use_readability_features:
        print("  Usando: TF-IDF + Features de Legibilidad + Términos Técnicos")
    else:
        print("  Usando: TF-IDF solamente (baseline)")
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_tfidf = vectorizer.fit_transform(train_texts)
    
    # Features de legibilidad y términos técnicos
    if use_readability_features:
        readability_extractor = ReadabilityFeatureExtractor()
        X_readability = readability_extractor.transform(train_texts)
        
        # Combinar TF-IDF con features de legibilidad
        X = hstack([X_tfidf, X_readability])
        print(f"   TF-IDF features: {X_tfidf.shape[1]:,}")
        print(f"   Readability features: {X_readability.shape[1]}")
        print(f"   Total features: {X.shape[1]:,}")
    else:
        X = X_tfidf
        print(f"   TF-IDF features: {X.shape[1]:,}")
    
    # Clasificador (misma configuración que entrenamiento final)
    clf = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    
    # CV estratificado (mantiene proporción de clases en cada fold)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Múltiples métricas
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro'
    }
    
    results = {}
    for metric_name, metric in scoring.items():
        scores = cross_val_score(
            clf, X, train_labels, 
            cv=cv, 
            scoring=metric, 
            n_jobs=-1
        )
        results[metric_name] = {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'folds': scores.tolist()
        }
        
        print(f"\n{metric_name.upper()}:")
        print(f"  Media: {scores.mean():.4f} (±{scores.std():.4f})")
        print(f"  Folds: {[f'{s:.3f}' for s in scores]}")
    
    return results

def evaluate_detailed(test_labels, y_pred, y_proba, test_texts):
    """
    Evaluación detallada con análisis de errores.
    
    NUEVA FUNCIONALIDAD - Expansión significativa de evaluación original.
    Incluye métricas por clase, análisis de errores, y ejemplos de fallos.
    """
    print("\n" + "="*80)
    print("EVALUACIÓN DETALLADA")
    print("="*80)
    
    # Métricas generales
    accuracy = accuracy_score(test_labels, y_pred)
    f1_macro = f1_score(test_labels, y_pred, average='macro')
    f1_weighted = f1_score(test_labels, y_pred, average='weighted')
    
    print(f"\nMÉTRICAS GENERALES:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  F1 Macro:    {f1_macro:.4f}")
    print(f"  F1 Weighted: {f1_weighted:.4f}")
    
    # Métricas por clase
    precision, recall, f1_class, support = precision_recall_fscore_support(
        test_labels, y_pred, 
        average=None, 
        labels=['non_pls', 'pls']
    )
    
    print(f"\n{'Clase':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print("-"*70)
    print(f"{'non_pls':<15} {precision[0]:<12.4f} {recall[0]:<12.4f} {f1_class[0]:<12.4f} {support[0]:<12}")
    print(f"{'pls':<15} {precision[1]:<12.4f} {recall[1]:<12.4f} {f1_class[1]:<12.4f} {support[1]:<12}")
    
    # Classification Report detallado
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(test_labels, y_pred, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, y_pred, labels=['non_pls', 'pls'])
    
    print("CONFUSION MATRIX:")
    print(f"                  Pred: non_pls  Pred: pls")
    print(f"True: non_pls          {cm[0,0]:>6}      {cm[0,1]:>6}")
    print(f"True: pls              {cm[1,0]:>6}      {cm[1,1]:>6}")
    
    # ROC-AUC
    y_test_numeric = (test_labels == 'pls').astype(int)
    y_proba_pls = y_proba[:, 1]
    auc = roc_auc_score(y_test_numeric, y_proba_pls)
    
    print(f"\nAUC-ROC: {auc:.4f}")
    
    # Análisis de errores
    print("\n" + "="*80)
    print("ANÁLISIS DE ERRORES")
    print("="*80)
    
    errors = test_labels != y_pred
    n_errors = errors.sum()
    
    print(f"\nTotal errores: {n_errors} ({n_errors/len(test_labels)*100:.1f}%)")
    
    # False Positives (Técnico predicho como PLS)
    fp_mask = (test_labels == 'non_pls') & (y_pred == 'pls')
    n_fp = fp_mask.sum()
    print(f"\nFalse Positives (Técnico→PLS): {n_fp}")
    print(f"  El modelo piensa que texto técnico es PLS")
    print(f"  Impacto: Podría NO generar PLS cuando debería")
    
    # False Negatives (PLS predicho como Técnico)
    fn_mask = (test_labels == 'pls') & (y_pred == 'non_pls')
    n_fn = fn_mask.sum()
    print(f"\nFalse Negatives (PLS→Técnico): {n_fn}")
    print(f"  El modelo piensa que PLS es texto técnico")
    print(f"  Impacto: Podría generar PLS innecesariamente")
    
    # Mostrar ejemplos de errores
    if n_fp > 0:
        print(f"\n EJEMPLO FALSE POSITIVE:")
        fp_indices = test_labels[fp_mask].index
        fp_idx = fp_indices[0]
        fp_text = test_texts.loc[fp_idx]
        fp_prob = y_proba[test_texts.index.get_loc(fp_idx), 1]
        
        print(f"  Texto (primeros 200 chars): '{fp_text[:200]}...'")
        print(f"  Prob PLS predicha: {fp_prob:.3f}")
        print(f"  Verdad: non_pls | Predicción: pls")
    
    if n_fn > 0:
        print(f"\n EJEMPLO FALSE NEGATIVE:")
        fn_indices = test_labels[fn_mask].index
        fn_idx = fn_indices[0]
        fn_text = test_texts.loc[fn_idx]
        fn_prob = y_proba[test_texts.index.get_loc(fn_idx), 1]
        
        print(f"  Texto (primeros 200 chars): '{fn_text[:200]}...'")
        print(f"  Prob PLS predicha: {fn_prob:.3f}")
        print(f"  Verdad: pls | Predicción: non_pls")
    
    return {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'auc': float(auc),
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1_class.tolist(),
        'support_per_class': support.tolist(),
        'confusion_matrix': cm.tolist(),
        'n_errors': int(n_errors),
        'false_positives': int(n_fp),
        'false_negatives': int(n_fn)
    }

def train_baseline_classifier(train_df, test_df, use_readability_features=True):
    """
    Entrena clasificador baseline con TF-IDF + Logistic Regression.
    
    MEJORADO: 
    - Incluye cross-validation y evaluación detallada
    - NUEVO: Features de legibilidad y términos técnicos médicos
    """
    print("\n" + "="*80)
    print("ENTRENANDO CLASIFICADOR BASELINE")
    print("="*80)
    
    if use_readability_features:
        print("  MEJORA IMPLEMENTADA: Features de legibilidad + términos técnicos")
    
    # Preparar datos con función CORREGIDA
    print("\n Preparando textos...")
    train_texts = prepare_text_data(train_df)
    test_texts = prepare_text_data(test_df)
    
    # Obtener labels correspondientes
    train_labels = train_df.loc[train_texts.index, 'label']
    test_labels = test_df.loc[test_texts.index, 'label']
    
    print(f"\nTrain: {len(train_texts):,} textos válidos")
    print(f"Test:  {len(test_texts):,} textos válidos")
    
    # Cross-validation ANTES de entrenar modelo final
    cv_results = cross_validate_classifier(train_texts, train_labels, use_readability_features)
    
    # TF-IDF Vectorizer
    print("\n Vectorizando con TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)
    
    print(f"   Dimensiones TF-IDF: {X_train_tfidf.shape}")
    print(f"   Features TF-IDF (n-gramas): {X_train_tfidf.shape[1]:,}")
    
    # Features de legibilidad y términos técnicos
    if use_readability_features:
        print("\n Extrayendo features de legibilidad y términos técnicos...")
        readability_extractor = ReadabilityFeatureExtractor()
        X_train_readability = readability_extractor.transform(train_texts)
        X_test_readability = readability_extractor.transform(test_texts)
        
        print(f"   Features de legibilidad: {X_train_readability.shape[1]}")
        print(f"     - Flesch Reading Ease")
        print(f"     - Flesch-Kincaid Grade")
        print(f"     - Gunning Fog Index")
        print(f"     - Promedio longitud palabras/oraciones")
        print(f"     - Ratio palabras complejas")
        print(f"     - Densidad términos técnicos médicos")
        print(f"     - Patrones estadísticos (p-values, CI, etc.)")
        
        # Combinar TF-IDF con features de legibilidad
        X_train = hstack([X_train_tfidf, X_train_readability])
        X_test = hstack([X_test_tfidf, X_test_readability])
        
        print(f"\n   Total features combinadas: {X_train.shape[1]:,}")
    else:
        X_train = X_train_tfidf
        X_test = X_test_tfidf
    
    # Logistic Regression con manejo de desbalance
    print("\n  Entrenando Logistic Regression...")
    print("   Regularización: L2 (C=1.0)")
    print("   Manejo de desbalance: class_weight='balanced'")
    print("   Nota: class_weight='balanced' es el enfoque estándar de sklearn")
    print("         para desbalance de clases. Es efectivo y académicamente válido.")
    
    clf = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        C=1.0
    )
    
    clf.fit(X_train, train_labels)
    
    # Predicciones
    print("\n Evaluando en test set...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    # Evaluación detallada
    detailed_metrics = evaluate_detailed(test_labels, y_pred, y_proba, test_texts)
    
    # Guardar modelo
    model_dir = Path('models/baseline_classifier')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(vectorizer, model_dir / 'vectorizer.pkl')
    joblib.dump(clf, model_dir / 'classifier.pkl')
    
    # Guardar extractor de legibilidad si se usó
    if use_readability_features:
        joblib.dump(readability_extractor, model_dir / 'readability_extractor.pkl')
        print(f"\n   Readability extractor guardado")
    
    # Guardar métricas comprehensivas
    metrics = {
        'model': 'baseline_tfidf_logreg_with_readability' if use_readability_features else 'baseline_tfidf_logreg',
        'use_readability_features': use_readability_features,
        'test_metrics': detailed_metrics,
        'cv_metrics': cv_results,
        'train_samples': len(train_texts),
        'test_samples': len(test_texts),
        'tfidf_features': X_train_tfidf.shape[1],
        'readability_features': X_train_readability.shape[1] if use_readability_features else 0,
        'total_features': X_train.shape[1]
    }
    
    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n Modelo guardado en: {model_dir}")
    
    # Crear y guardar función gate mejorada
    save_gate_function(clf, vectorizer, model_dir, readability_extractor if use_readability_features else None)
    
    return clf, vectorizer, metrics

def create_gate_function(clf, vectorizer, threshold=0.7, readability_extractor=None):
    """
    Crea función 'gate' MEJORADA con umbral ajustable.
    
    MEJORA sobre versión original:
    - Ahora usa probabilidades en vez de solo predicción binaria
    - Umbral ajustable (default 0.7 = conservador)
    - Puede retornar confidence score
    - NUEVO: Soporta features de legibilidad si están disponibles
    
    Esta función se usa ANTES de generar PLS:
    - Si el texto ya es PLS (prob > threshold) → skip generación
    - Si el texto es técnico (prob < threshold) → generar PLS
    
    Args:
        clf: Clasificador entrenado
        vectorizer: TF-IDF vectorizer entrenado
        threshold: Probabilidad mínima para considerar como PLS
                  0.5 = decisión balanceada
                  0.7 = conservador (recomendado, menos false positives)
                  0.9 = muy conservador
        readability_extractor: ReadabilityFeatureExtractor opcional
    
    Uso:
        gate = create_gate_function(clf, vectorizer, threshold=0.7, readability_extractor=extractor)
        is_pls, confidence = gate(text, return_confidence=True)
        if is_pls:
            print(f"Skip generación (confidence: {confidence:.2f})")
        else:
            generate_pls(text)
    """
    def gate(text, return_confidence=False):
        """
        Determina si un texto ya es PLS.
        
        Args:
            text: Texto a clasificar
            return_confidence: Si True, retorna (is_pls, prob_pls)
        
        Returns:
            bool o tuple: is_pls o (is_pls, prob_pls)
        """
        if not text or len(str(text).strip()) < 50:
            # Texto muy corto, asumir que necesita generación
            return (False, 0.0) if return_confidence else False
        
        # Vectorizar con TF-IDF
        X_tfidf = vectorizer.transform([str(text)])
        
        # Agregar features de legibilidad si están disponibles
        if readability_extractor is not None:
            X_readability = readability_extractor.transform([str(text)])
            X = hstack([X_tfidf, X_readability])
        else:
            X = X_tfidf
        
        # Predecir
        proba = clf.predict_proba(X)[0]
        
        # proba[0] = prob non_pls, proba[1] = prob pls
        prob_pls = float(proba[1])
        is_pls = prob_pls >= threshold
        
        if return_confidence:
            return is_pls, prob_pls
        return is_pls
    
    return gate

def save_gate_function(clf, vectorizer, model_dir, readability_extractor=None):
    """
    Guarda helper para cargar y usar el gate fácilmente.
    """
    gate_helper_code = '''"""
Helper para cargar y usar el gate del clasificador PLS/non-PLS.

Uso básico:
    from models.baseline_classifier.gate_helper import load_gate
    gate = load_gate()
    
    if gate(text):
        print("Texto ya es PLS, skip generación")
    else:
        print("Texto es técnico, generar PLS")

Uso con confidence:
    gate = load_gate(threshold=0.7)  # Ajustar umbral si necesario
    is_pls, confidence = gate(text, return_confidence=True)
    print(f"Es PLS: {is_pls}, Confidence: {confidence:.2f}")
"""
import joblib
from pathlib import Path
from scipy.sparse import hstack

def create_gate_function(clf, vectorizer, threshold=0.7, readability_extractor=None):
    """
    Crea función gate con umbral ajustable.
    
    Args:
        threshold: 0.5 (balanceado), 0.7 (conservador), 0.9 (muy conservador)
        readability_extractor: Extractor opcional de features de legibilidad
    """
    def gate(text, return_confidence=False):
        if not text or len(str(text).strip()) < 50:
            return (False, 0.0) if return_confidence else False
        
        # Vectorizar con TF-IDF
        X_tfidf = vectorizer.transform([str(text)])
        
        # Agregar features de legibilidad si están disponibles
        if readability_extractor is not None:
            X_readability = readability_extractor.transform([str(text)])
            X = hstack([X_tfidf, X_readability])
        else:
            X = X_tfidf
        
        proba = clf.predict_proba(X)[0]
        prob_pls = float(proba[1])
        is_pls = prob_pls >= threshold
        
        if return_confidence:
            return is_pls, prob_pls
        return is_pls
    
    return gate

def load_gate(threshold=0.7):
    """
    Carga el gate del clasificador entrenado.
    
    Args:
        threshold: Probabilidad mínima para considerar como PLS (default: 0.7)
    
    Returns:
        gate function
    """
    model_dir = Path('models/baseline_classifier')
    vectorizer = joblib.load(model_dir / 'vectorizer.pkl')
    clf = joblib.load(model_dir / 'classifier.pkl')
    
    # Intentar cargar readability extractor si existe
    readability_extractor = None
    readability_path = model_dir / 'readability_extractor.pkl'
    if readability_path.exists():
        readability_extractor = joblib.load(readability_path)
    
    return create_gate_function(clf, vectorizer, threshold=threshold, 
                                readability_extractor=readability_extractor)

if __name__ == "__main__":
    # Ejemplo de uso
    gate = load_gate(threshold=0.7)
    
    test_pls = "This is a plain language summary of a medical study. The results show that the treatment works."
    test_tech = "The randomized controlled trial (RCT) demonstrated statistically significant efficacy (p<0.001)."
    
    is_pls1, conf1 = gate(test_pls, return_confidence=True)
    is_pls2, conf2 = gate(test_tech, return_confidence=True)
    
    print(f"PLS text: is_pls={is_pls1}, confidence={conf1:.3f}")
    print(f"Tech text: is_pls={is_pls2}, confidence={conf2:.3f}")
'''
    
    helper_path = model_dir / 'gate_helper.py'
    with open(helper_path, 'w', encoding='utf-8') as f:
        f.write(gate_helper_code)
    
    print(f" Gate helper guardado en: {helper_path}")

def evaluate_by_source(test_df, clf, vectorizer, readability_extractor=None):
    """
    Evalúa el modelo por fuente de datos.
    
    Útil para detectar si el modelo funciona bien en todas las fuentes
    o si tiene sesgo hacia alguna fuente específica.
    
    MEJORADO: Soporta features de legibilidad si están disponibles.
    """
    print("\n" + "="*80)
    print("EVALUACIÓN POR FUENTE")
    print("="*80)
    
    if 'source_dataset' not in test_df.columns:
        print("  No hay columna 'source_dataset', skip evaluación por fuente")
        return {}
    
    source_metrics = {}
    
    for source in test_df['source_dataset'].unique():
        if pd.isna(source):
            continue
            
        source_data = test_df[test_df['source_dataset'] == source]
        source_texts = prepare_text_data(source_data)
        
        if len(source_texts) < 10:  # Skip sources with too few samples
            print(f"\n  {source}: Solo {len(source_texts)} muestras, skip")
            continue
            
        source_labels = source_data.loc[source_texts.index, 'label']
        
        # Vectorizar y predecir
        X_source_tfidf = vectorizer.transform(source_texts)
        
        # Agregar features de legibilidad si están disponibles
        if readability_extractor is not None:
            X_source_readability = readability_extractor.transform(source_texts)
            X_source = hstack([X_source_tfidf, X_source_readability])
        else:
            X_source = X_source_tfidf
        
        y_pred_source = clf.predict(X_source)
        
        # Métricas
        f1_macro = f1_score(source_labels, y_pred_source, average='macro')
        accuracy = accuracy_score(source_labels, y_pred_source)
        
        source_metrics[source] = {
            'samples': len(source_texts),
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'distribution': source_labels.value_counts().to_dict()
        }
        
        print(f"\n{source}:")
        print(f"  Muestras: {len(source_texts):,}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Macro: {f1_macro:.4f}")
        print(f"  Distribución: {source_labels.value_counts().to_dict()}")
    
    # Guardar métricas por fuente
    if source_metrics:
        with open('models/baseline_classifier/source_metrics.json', 'w') as f:
            json.dump(source_metrics, f, indent=2)
        print(f"\n Métricas por fuente guardadas")
    
    return source_metrics

def main():
    """
    Script principal con flujo completo de entrenamiento y evaluación.
    """
    # Cargar datos
    train_df, test_df = load_data()
    
    # Entrenar clasificador baseline CON features de legibilidad
    use_readability = True  # Cambiar a False para usar solo TF-IDF
    clf, vectorizer, metrics = train_baseline_classifier(train_df, test_df, use_readability_features=use_readability)
    
    # Cargar readability extractor si se usó
    readability_extractor = None
    if use_readability:
        model_dir = Path('models/baseline_classifier')
        readability_path = model_dir / 'readability_extractor.pkl'
        if readability_path.exists():
            readability_extractor = joblib.load(readability_path)
    
    # Evaluar por fuente
    source_metrics = evaluate_by_source(test_df, clf, vectorizer, readability_extractor)
    
    # Verificar si cumple target académico
    print("\n" + "="*80)
    print("VERIFICACIÓN DE TARGET")
    print("="*80)
    
    target_f1 = 0.85
    actual_f1 = metrics['test_metrics']['f1_macro']
    
    if actual_f1 >= target_f1:
        print(f" TARGET CUMPLIDO: F1 Macro {actual_f1:.4f} >= {target_f1}")
        print("   El clasificador alcanza el objetivo académico del proyecto")
    else:
        print(f"  TARGET NO CUMPLIDO: F1 Macro {actual_f1:.4f} < {target_f1}")
        print(f"   Gap: {target_f1 - actual_f1:.4f}")
        print("   Considerar: más features, otros modelos, o ajuste de hiperparámetros")
    
    # Demostración del gate function
    print("\n" + "="*80)
    print("DEMOSTRACIÓN FUNCIÓN GATE")
    print("="*80)
    
    # Cargar readability extractor para el gate si está disponible
    gate_readability_extractor = None
    if use_readability:
        model_dir = Path('models/baseline_classifier')
        readability_path = model_dir / 'readability_extractor.pkl'
        if readability_path.exists():
            gate_readability_extractor = joblib.load(readability_path)
    
    gate = create_gate_function(clf, vectorizer, threshold=0.7, readability_extractor=gate_readability_extractor)
    
    # Ejemplos de uso
    examples = [
        ("Ejemplo PLS", 
         "This is a plain language summary explaining the study results in simple terms. "
         "Researchers tested a new medicine. The medicine helped people feel better. "
         "About 6 out of 10 people improved. The studies were good quality."),
        
        ("Ejemplo Técnico", 
         "The randomized controlled trial evaluated the efficacy of the intervention "
         "using a double-blind placebo-controlled methodology with primary endpoints "
         "measured at 12 weeks (RR 0.72, 95% CI 0.58-0.89, p<0.001).")
    ]
    
    for label, text in examples:
        is_pls, prob = gate(text, return_confidence=True)
        action = 'SKIP generación' if is_pls else 'GENERAR PLS'
        
        print(f"\n{label}:")
        print(f"  Texto: '{text[:80]}...'")
        print(f"  Es PLS: {is_pls}")
        print(f"  Probabilidad PLS: {prob:.3f}")
        print(f"  → Acción: {action}")
    
    # Instrucciones de uso
    print("\n" + "="*80)
    print("CÓMO USAR EL GATE EN TU PIPELINE")
    print("="*80)
    print("""
Para usar el gate en la generación de PLS sintéticos:

1. Cargar el gate:
   from models.baseline_classifier.gate_helper import load_gate
   gate = load_gate(threshold=0.7)

2. Antes de generar PLS:
   for texto in textos_tecnicos:
       if gate(texto):
           print("Ya es PLS, skip")
           continue
       else:
           pls = generar_pls_con_gpt(texto)
           
3. Con confidence score:
   is_pls, confidence = gate(texto, return_confidence=True)
   if is_pls and confidence > 0.8:
       print(f"Muy seguro que es PLS (conf={confidence:.2f}), skip")
   elif confidence < 0.6:
       print(f"Incierto (conf={confidence:.2f}), revisar manualmente")
""")
    print("="*80)
    
    print("\n ENTRENAMIENTO Y EVALUACIÓN COMPLETADOS")
    print(f" Modelo guardado en: models/baseline_classifier/")
    print(f" Métricas guardadas en: models/baseline_classifier/metrics.json")

if __name__ == "__main__":
    main()