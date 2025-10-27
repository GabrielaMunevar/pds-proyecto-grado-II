#!/usr/bin/env python3
"""
Generador de Pares Sintéticos para Entrenar el Generador de PLS
Crea pares técnico-simple usando los datos disponibles.

Estrategia:
1. Usar textos PLS como "simple" 
2. Generar versiones "técnicas" usando técnicas de complejificación
3. Crear pares sintéticos para entrenar el generador

Uso:
    python src/data/create_synthetic_pairs.py
"""

import pandas as pd
import numpy as np
import re
import random
from pathlib import Path
import json
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

def load_data():
    """Carga los datos procesados."""
    print("Cargando datos...")
    
    df = pd.read_csv('data/processed/dataset_clean.csv')
    
    # Filtrar solo registros con PLS
    pls_data = df[df['label'] == 'pls'].copy()
    
    print(f"Total registros PLS: {len(pls_data)}")
    
    return pls_data

def complexify_text(text, complexity_level=0.7):
    """
    Convierte texto simple en versión más técnica.
    
    Args:
        text: Texto simple
        complexity_level: Nivel de complejidad (0.0-1.0)
    """
    if pd.isna(text) or len(str(text)) < 20:
        return text
    
    text = str(text)
    
    # Diccionario de simplificaciones a reversar
    simplifications = {
        # Términos médicos simples -> técnicos
        'heart attack': 'myocardial infarction',
        'heart disease': 'cardiovascular disease',
        'high blood pressure': 'hypertension',
        'diabetes': 'diabetes mellitus',
        'cancer': 'neoplasia',
        'stroke': 'cerebrovascular accident',
        'kidney disease': 'renal disease',
        'liver disease': 'hepatic disease',
        'lung disease': 'pulmonary disease',
        'brain disease': 'neurological disease',
        
        # Verbos simples -> técnicos
        'help': 'facilitate',
        'show': 'demonstrate',
        'find': 'identify',
        'use': 'utilize',
        'give': 'administer',
        'take': 'consume',
        'get': 'acquire',
        'make': 'produce',
        'work': 'function',
        'stop': 'discontinue',
        
        # Adjetivos simples -> técnicos
        'good': 'beneficial',
        'bad': 'adverse',
        'big': 'significant',
        'small': 'minimal',
        'fast': 'rapid',
        'slow': 'gradual',
        'safe': 'well-tolerated',
        'unsafe': 'contraindicated',
        'effective': 'efficacious',
        'ineffective': 'inefficacious',
    }
    
    # Aplicar transformaciones
    complex_text = text
    for simple, complex_term in simplifications.items():
        if random.random() < complexity_level:
            # Reemplazar solo palabras completas
            pattern = r'\b' + re.escape(simple) + r'\b'
            complex_text = re.sub(pattern, complex_term, complex_text, flags=re.IGNORECASE)
    
    # Agregar terminología médica técnica
    if random.random() < complexity_level:
        technical_terms = [
            'clinical trial', 'randomized controlled trial', 'placebo-controlled',
            'double-blind', 'adverse events', 'primary endpoint', 'secondary endpoint',
            'statistical significance', 'confidence interval', 'p-value',
            'inclusion criteria', 'exclusion criteria', 'protocol',
            'informed consent', 'ethics committee', 'regulatory approval'
        ]
        
        # Insertar términos técnicos aleatoriamente
        words = complex_text.split()
        if len(words) > 10:
            insert_pos = random.randint(5, len(words)-5)
            technical_term = random.choice(technical_terms)
            words.insert(insert_pos, f"({technical_term})")
            complex_text = ' '.join(words)
    
    # Hacer el texto más formal
    if random.random() < complexity_level:
        formal_replacements = {
            r'\bwe\b': 'the research team',
            r'\byou\b': 'patients',
            r'\bpeople\b': 'individuals',
            r'\bstudy\b': 'investigation',
            r'\btest\b': 'assessment',
            r'\btry\b': 'attempt',
            r'\bcheck\b': 'evaluate',
        }
        
        for pattern, replacement in formal_replacements.items():
            complex_text = re.sub(pattern, replacement, complex_text, flags=re.IGNORECASE)
    
    return complex_text

def create_synthetic_pairs(pls_data):
    """Crea pares sintéticos técnico-simple."""
    print("\n=== CREANDO PARES SINTÉTICOS ===")
    
    pairs = []
    
    for idx, row in pls_data.iterrows():
        # Usar resumen como texto "simple"
        simple_text = row['resumen']
        
        if pd.isna(simple_text) or len(str(simple_text)) < 20:
            continue
        
        # Crear versión técnica
        technical_text = complexify_text(simple_text, complexity_level=0.8)
        
        # Crear par sintético
        pair = {
            'texto_tecnico': technical_text,
            'texto_simple': simple_text,
            'source_dataset': row['source_dataset'],
            'source_bucket': row['source_bucket'],
            'doc_id': f"synthetic_{row['doc_id']}",
            'split': row['split'],
            'original_label': row['label'],
            'word_count_tech': len(technical_text.split()),
            'word_count_simple': len(simple_text.split()),
            'compression_ratio': len(simple_text.split()) / len(technical_text.split()) if len(technical_text.split()) > 0 else 0,
            'is_synthetic': True
        }
        
        pairs.append(pair)
    
    print(f"Pares sintéticos creados: {len(pairs)}")
    
    return pairs

def analyze_pairs(pairs):
    """Analiza los pares creados."""
    print("\n=== ANÁLISIS DE PARES SINTÉTICOS ===")
    
    if not pairs:
        print("No hay pares para analizar")
        return
    
    df_pairs = pd.DataFrame(pairs)
    
    print(f"Total pares: {len(df_pairs)}")
    print(f"Promedio palabras técnico: {df_pairs['word_count_tech'].mean():.1f}")
    print(f"Promedio palabras simple: {df_pairs['word_count_simple'].mean():.1f}")
    print(f"Ratio compresión promedio: {df_pairs['compression_ratio'].mean():.2f}")
    
    print("\nDistribución por fuente:")
    print(df_pairs['source_dataset'].value_counts())
    
    print("\nDistribución por split:")
    print(df_pairs['split'].value_counts())
    
    # Ejemplos
    print("\n=== EJEMPLOS DE PARES ===")
    for i, pair in enumerate(pairs[:3]):
        print(f"\nEjemplo {i+1}:")
        print(f"Técnico ({pair['word_count_tech']} palabras):")
        print(f"  {pair['texto_tecnico'][:200]}...")
        print(f"Simple ({pair['word_count_simple']} palabras):")
        print(f"  {pair['texto_simple'][:200]}...")
        print(f"Ratio compresión: {pair['compression_ratio']:.2f}")

def save_pairs(pairs):
    """Guarda los pares sintéticos."""
    print("\n=== GUARDANDO PARES ===")
    
    # Crear directorio
    output_dir = Path('data/processed/synthetic_pairs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar como DataFrame
    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(output_dir / 'synthetic_pairs.csv', index=False)
    
    # Guardar como JSONL para entrenamiento
    with open(output_dir / 'synthetic_pairs.jsonl', 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    # Guardar estadísticas
    stats = {
        'total_pairs': len(pairs),
        'avg_tech_words': float(df_pairs['word_count_tech'].mean()),
        'avg_simple_words': float(df_pairs['word_count_simple'].mean()),
        'avg_compression_ratio': float(df_pairs['compression_ratio'].mean()),
        'source_distribution': df_pairs['source_dataset'].value_counts().to_dict(),
        'split_distribution': df_pairs['split'].value_counts().to_dict()
    }
    
    with open(output_dir / 'synthetic_pairs_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Pares guardados en: {output_dir}")
    print(f"- synthetic_pairs.csv ({len(pairs)} pares)")
    print(f"- synthetic_pairs.jsonl ({len(pairs)} pares)")
    print(f"- synthetic_pairs_stats.json (estadísticas)")

def main():
    """Función principal."""
    print("=== GENERADOR DE PARES SINTÉTICOS ===")
    
    # Cargar datos
    pls_data = load_data()
    
    if len(pls_data) == 0:
        print("No hay datos PLS disponibles")
        return
    
    # Crear pares sintéticos
    pairs = create_synthetic_pairs(pls_data)
    
    if len(pairs) == 0:
        print("No se pudieron crear pares sintéticos")
        return
    
    # Analizar pares
    analyze_pairs(pairs)
    
    # Guardar pares
    save_pairs(pairs)
    
    print("\nPares sintéticos creados exitosamente!")
    print("\nPróximo paso: Entrenar generador BART con estos pares")

if __name__ == "__main__":
    main()

