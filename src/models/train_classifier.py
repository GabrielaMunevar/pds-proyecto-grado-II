"""
Clasificador Baseline MEJORADO para PLS/non-PLS
TF-IDF + Logistic Regression

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
import joblib
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

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

def cross_validate_classifier(train_texts, train_labels):
    """
    Cross-validation 5-fold estratificado para verificar estabilidad.
    
    NUEVA FUNCIONALIDAD - No existía en versión original.
    Valida que el modelo generaliza bien y no depende del split específico.
    """
    print("\n" + "="*80)
    print("CROSS-VALIDATION (5-FOLD ESTRATIFICADO)")
    print("="*80)
    
    # Vectorizer (misma configuración que entrenamiento final)
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X = vectorizer.fit_transform(train_texts)
    
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

def train_baseline_classifier(train_df, test_df):
    """
    Entrena clasificador baseline con TF-IDF + Logistic Regression.
    
    MEJORADO: Incluye cross-validation y evaluación detallada.
    """
    print("\n" + "="*80)
    print("ENTRENANDO CLASIFICADOR BASELINE")
    print("="*80)
    
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
    cv_results = cross_validate_classifier(train_texts, train_labels)
    
    # TF-IDF Vectorizer
    print("\n Vectorizando con TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    print(f"   Dimensiones TF-IDF: {X_train.shape}")
    print(f"   Features (n-gramas): {X_train.shape[1]:,}")
    
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
    
    # Guardar métricas comprehensivas
    metrics = {
        'model': 'baseline_tfidf_logreg',
        'test_metrics': detailed_metrics,
        'cv_metrics': cv_results,
        'train_samples': len(train_texts),
        'test_samples': len(test_texts),
        'features': X_train.shape[1]
    }
    
    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n Modelo guardado en: {model_dir}")
    
    # Crear y guardar función gate mejorada
    save_gate_function(clf, vectorizer, model_dir)
    
    return clf, vectorizer, metrics

def create_gate_function(clf, vectorizer, threshold=0.7):
    """
    Crea función 'gate' MEJORADA con umbral ajustable.
    
    MEJORA sobre versión original:
    - Ahora usa probabilidades en vez de solo predicción binaria
    - Umbral ajustable (default 0.7 = conservador)
    - Puede retornar confidence score
    
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
    
    Uso:
        gate = create_gate_function(clf, vectorizer, threshold=0.7)
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
        
        # Vectorizar y predecir
        X = vectorizer.transform([str(text)])
        proba = clf.predict_proba(X)[0]
        
        # proba[0] = prob non_pls, proba[1] = prob pls
        prob_pls = float(proba[1])
        is_pls = prob_pls >= threshold
        
        if return_confidence:
            return is_pls, prob_pls
        return is_pls
    
    return gate

def save_gate_function(clf, vectorizer, model_dir):
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

def create_gate_function(clf, vectorizer, threshold=0.7):
    """
    Crea función gate con umbral ajustable.
    
    Args:
        threshold: 0.5 (balanceado), 0.7 (conservador), 0.9 (muy conservador)
    """
    def gate(text, return_confidence=False):
        if not text or len(str(text).strip()) < 50:
            return (False, 0.0) if return_confidence else False
        
        X = vectorizer.transform([str(text)])
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
    return create_gate_function(clf, vectorizer, threshold=threshold)

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

def evaluate_by_source(test_df, clf, vectorizer):
    """
    Evalúa el modelo por fuente de datos.
    
    Útil para detectar si el modelo funciona bien en todas las fuentes
    o si tiene sesgo hacia alguna fuente específica.
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
        X_source = vectorizer.transform(source_texts)
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
    
    # Entrenar clasificador baseline
    clf, vectorizer, metrics = train_baseline_classifier(train_df, test_df)
    
    # Evaluar por fuente
    source_metrics = evaluate_by_source(test_df, clf, vectorizer)
    
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
    
    gate = create_gate_function(clf, vectorizer, threshold=0.7)
    
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