#!/usr/bin/env python3
"""
Clasificador Baseline para PLS/non-PLS
Usa TF-IDF + Logistic Regression como línea base.

Uso:
    python src/models/train_classifier.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

def load_data():
    """Carga los datos de entrenamiento y prueba."""
    print("Cargando datos...")
    
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    print(f"Train: {len(train_df)} registros")
    print(f"Test: {len(test_df)} registros")
    
    # Filtrar solo registros con label (excluir unlabeled)
    train_labeled = train_df[train_df['label'].notna()].copy()
    test_labeled = test_df[test_df['label'].notna()].copy()
    
    print(f"Train con label: {len(train_labeled)} registros")
    print(f"Test con label: {len(test_labeled)} registros")
    
    return train_labeled, test_labeled

def prepare_text_data(df):
    """Prepara los datos de texto para entrenamiento."""
    # Usar texto_original como input principal
    texts = df['texto_original'].fillna('').astype(str)
    
    # Si no hay texto_original, usar resumen
    mask_no_text = texts.str.len() < 10
    texts.loc[mask_no_text] = df.loc[mask_no_text, 'resumen'].fillna('').astype(str)
    
    # Filtrar textos muy cortos
    texts = texts[texts.str.len() >= 20]
    
    return texts

def train_baseline_classifier(train_df, test_df):
    """Entrena clasificador baseline con TF-IDF + Logistic Regression."""
    print("\n=== ENTRENANDO CLASIFICADOR BASELINE ===")
    
    # Preparar datos
    train_texts = prepare_text_data(train_df)
    test_texts = prepare_text_data(test_df)
    
    # Obtener labels correspondientes
    train_labels = train_df.loc[train_texts.index, 'label']
    test_labels = test_df.loc[test_texts.index, 'label']
    
    print(f"Textos de entrenamiento: {len(train_texts)}")
    print(f"Textos de prueba: {len(test_texts)}")
    
    # Distribución de clases
    print("\nDistribución de clases en train:")
    print(train_labels.value_counts())
    
    # TF-IDF Vectorizer
    print("\nVectorizando textos con TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    print(f"Dimensiones TF-IDF: {X_train.shape}")
    
    # Logistic Regression
    print("\nEntrenando Logistic Regression...")
    clf = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    
    clf.fit(X_train, train_labels)
    
    # Predicciones
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    
    # Métricas
    f1_macro = f1_score(test_labels, y_pred, average='macro')
    f1_weighted = f1_score(test_labels, y_pred, average='weighted')
    
    print(f"\n=== RESULTADOS BASELINE ===")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, y_pred))
    
    # Guardar modelo
    model_dir = Path('models/baseline_classifier')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(vectorizer, model_dir / 'vectorizer.pkl')
    joblib.dump(clf, model_dir / 'classifier.pkl')
    
    # Guardar métricas
    metrics = {
        'model': 'baseline_tfidf_logreg',
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'train_samples': len(train_texts),
        'test_samples': len(test_texts),
        'features': X_train.shape[1]
    }
    
    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nModelo guardado en: {model_dir}")
    
    return clf, vectorizer, metrics

def evaluate_by_source(test_df, clf, vectorizer):
    """Evalúa el modelo por fuente de datos."""
    print("\n=== EVALUACIÓN POR FUENTE ===")
    
    source_metrics = {}
    
    for source in test_df['source_dataset'].unique():
        source_data = test_df[test_df['source_dataset'] == source]
        source_texts = prepare_text_data(source_data)
        
        if len(source_texts) < 10:  # Skip sources with too few samples
            continue
            
        source_labels = source_data.loc[source_texts.index, 'label']
        
        # Vectorizar y predecir
        X_source = vectorizer.transform(source_texts)
        y_pred_source = clf.predict(X_source)
        
        # Métricas
        f1_macro = f1_score(source_labels, y_pred_source, average='macro')
        
        source_metrics[source] = {
            'samples': len(source_texts),
            'f1_macro': float(f1_macro),
            'distribution': source_labels.value_counts().to_dict()
        }
        
        print(f"\n{source}:")
        print(f"  Muestras: {len(source_texts)}")
        print(f"  F1 Macro: {f1_macro:.4f}")
        print(f"  Distribución: {source_labels.value_counts().to_dict()}")
    
    # Guardar métricas por fuente
    with open('models/baseline_classifier/source_metrics.json', 'w') as f:
        json.dump(source_metrics, f, indent=2)
    
    return source_metrics

def main():
    # Cargar datos
    train_df, test_df = load_data()
    
    # Entrenar baseline
    clf, vectorizer, metrics = train_baseline_classifier(train_df, test_df)
    
    # Evaluar por fuente
    source_metrics = evaluate_by_source(test_df, clf, vectorizer)
    
    # Verificar si cumple target
    target_f1 = 0.85
    if metrics['f1_macro'] >= target_f1:
        print(f"\nTARGET CUMPLIDO: F1 Macro {metrics['f1_macro']:.4f} >= {target_f1}")
    else:
        print(f"\nTARGET NO CUMPLIDO: F1 Macro {metrics['f1_macro']:.4f} < {target_f1}")

if __name__ == "__main__":
    main()
