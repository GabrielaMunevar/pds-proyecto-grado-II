#!/usr/bin/env python3
"""
Analizador de Calidad de Pares Sintéticos
===========================================

Analiza la calidad de los pares sintéticos generados para identificar
problemas antes del entrenamiento.

Uso:
    python src/analysis/analyze_synthetic_pairs.py
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from collections import Counter
import re

def load_pairs(file_path):
    """Cargar pares sintéticos"""
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs

def calculate_overlap(text1, text2):
    """Calcular overlap léxico entre dos textos"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if len(words1) == 0 or len(words2) == 0:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

def analyze_compression(pairs, sample_size=100):
    """Analizar ratio de compresión"""
    print("\n=== ANÁLISIS DE COMPRESIÓN ===")
    
    ratios = []
    for pair in pairs[:sample_size]:
        tech_words = len(pair['texto_tecnico'].split())
        simple_words = len(pair['texto_simple'].split())
        
        if tech_words > 0:
            ratio = simple_words / tech_words
            ratios.append(ratio)
    
    ratios = np.array(ratios)
    
    print(f"Ratio promedio: {ratios.mean():.3f}")
    print(f"Ratio mediano: {np.median(ratios):.3f}")
    print(f"Std: {ratios.std():.3f}")
    print(f"Min: {ratios.min():.3f}")
    print(f"Max: {ratios.max():.3f}")
    
    # Categorizar
    good = np.sum((ratios >= 0.3) & (ratios <= 0.8))
    ok = np.sum((ratios < 0.3) | (ratios > 0.8))
    
    print(f"\nDistribución:")
    print(f"  Bueno (0.3-0.8): {good}/{len(ratios)} ({good/len(ratios)*100:.1f}%)")
    print(f"  Inadecuado: {ok}/{len(ratios)} ({ok/len(ratios)*100:.1f}%)")
    
    return ratios

def analyze_overlap(pairs, sample_size=100):
    """Analizar overlap léxico"""
    print("\n=== ANÁLISIS DE OVERLAP LÉXICO ===")
    
    overlaps = []
    for pair in pairs[:sample_size]:
        overlap = calculate_overlap(pair['texto_tecnico'], pair['texto_simple'])
        overlaps.append(overlap)
    
    overlaps = np.array(overlaps)
    
    print(f"Overlap promedio: {overlaps.mean():.3f}")
    print(f"Overlap mediano: {np.median(overlaps):.3f}")
    print(f"Std: {overlaps.std():.3f}")
    
    # Categorizar
    low = np.sum(overlaps < 0.5)
    medium = np.sum((overlaps >= 0.5) & (overlaps < 0.8))
    high = np.sum(overlaps >= 0.8)
    
    print(f"\nDistribución:")
    print(f"  Bajo (<0.5): {low}/{len(overlaps)} ({low/len(overlaps)*100:.1f}%)")
    print(f"  Medio (0.5-0.8): {medium}/{len(overlaps)} ({medium/len(overlaps)*100:.1f}%)")
    print(f"  Alto (≥0.8): {high}/{len(overlaps)} ({high/len(overlaps)*100:.1f}%)")
    
    return overlaps

def analyze_word_diversity(pairs, sample_size=100):
    """Analizar diversidad de palabras"""
    print("\n=== ANÁLISIS DE DIVERSIDAD ===")
    
    # Contar palabras únicas
    tech_unique_words = set()
    simple_unique_words = set()
    
    for pair in pairs[:sample_size]:
        tech_unique_words.update(pair['texto_tecnico'].lower().split())
        simple_unique_words.update(pair['texto_simple'].lower().split())
    
    # Palabras nuevas en simple
    new_words = simple_unique_words - tech_unique_words
    kept_words = simple_unique_words & tech_unique_words
    
    print(f"Palabras únicas en técnico: {len(tech_unique_words)}")
    print(f"Palabras únicas en simple: {len(simple_unique_words)}")
    print(f"Palabras nuevas en simple: {len(new_words)} ({len(new_words)/len(simple_unique_words)*100:.1f}%)")
    print(f"Palabras mantenidas: {len(kept_words)} ({len(kept_words)/len(simple_unique_words)*100:.1f}%)")
    
    return {
        'tech_unique': len(tech_unique_words),
        'simple_unique': len(simple_unique_words),
        'new_words': len(new_words),
        'kept_words': len(kept_words)
    }

def analyze_examples(pairs, n=5):
    """Mostrar ejemplos aleatorios"""
    print("\n=== EJEMPLOS ALEATORIOS ===")
    
    import random
    random.seed(42)
    sample_pairs = random.sample(pairs, min(n, len(pairs)))
    
    for i, pair in enumerate(sample_pairs, 1):
        print(f"\n--- Ejemplo {i} ---")
        print(f"\nTÉCNICO ({len(pair['texto_tecnico'].split())} palabras):")
        print(pair['texto_tecnico'][:300] + "..." if len(pair['texto_tecnico']) > 300 else pair['texto_tecnico'])
        
        print(f"\nSIMPLE ({len(pair['texto_simple'].split())} palabras):")
        print(pair['texto_simple'][:300] + "..." if len(pair['texto_simple']) > 300 else pair['texto_simple'])
        
        overlap = calculate_overlap(pair['texto_tecnico'], pair['texto_simple'])
        print(f"\nOverlap léxico: {overlap:.3f}")
        
        comp_ratio = len(pair['texto_simple'].split()) / len(pair['texto_tecnico'].split()) if len(pair['texto_tecnico'].split()) > 0 else 0
        print(f"Ratio compresión: {comp_ratio:.3f}")

def analyze_word_changes(pairs, sample_size=200):
    """Analizar cambios de palabras específicos"""
    print("\n=== ANÁLISIS DE CAMBIOS ESPECÍFICOS ===")
    
    # Palabras que cambian frecuentemente
    changes = {}
    
    for pair in pairs[:sample_size]:
        tech_words = pair['texto_tecnico'].lower().split()
        simple_words = pair['texto_simple'].lower().split()
        
        tech_set = set(tech_words)
        simple_set = set(simple_words)
        
        # Palabras nuevas
        new = simple_set - tech_set
        for word in new:
            changes[word] = changes.get(word, 0) + 1
    
    # Top cambios
    top_changes = sorted(changes.items(), key=lambda x: x[1], reverse=True)[:20]
    
    print("Top 20 palabras nuevas en simple:")
    for word, count in top_changes:
        print(f"  {word}: {count} veces")

def main():
    """Función principal"""
    print("=" * 80)
    print("ANÁLISIS DE PARES SINTÉTICOS")
    print("=" * 80)
    
    # Cargar pares
    pairs_file = Path('data/processed/synthetic_pairs/synthetic_pairs.jsonl')
    
    if not pairs_file.exists():
        print(f"Archivo no encontrado: {pairs_file}")
        return
    
    print(f"\nCargando pares desde: {pairs_file}")
    pairs = load_pairs(pairs_file)
    
    print(f"Total pares cargados: {len(pairs)}")
    
    # Análisis 1: Compresión
    ratios = analyze_compression(pairs)
    
    # Análisis 2: Overlap
    overlaps = analyze_overlap(pairs)
    
    # Análisis 3: Diversidad
    diversity = analyze_word_diversity(pairs)
    
    # Análisis 4: Cambios específicos
    analyze_word_changes(pairs)
    
    # Análisis 5: Ejemplos
    analyze_examples(pairs, n=5)
    
    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN Y PROBLEMAS IDENTIFICADOS")
    print("=" * 80)
    
    problems = []
    
    if ratios.mean() > 0.9:
        problems.append("Ratio de compresión muy alto (>0.9). Los textos son casi idénticos.")
    
    if overlaps.mean() > 0.8:
        problems.append("Overlap léxico muy alto (>0.8). Cambios insuficientes.")
    
    if diversity['new_words'] / diversity['simple_unique'] < 0.2:
        problems.append("Pocas palabras nuevas (<20%). Simplificación limitada.")
    
    if problems:
        print("\nPROBLEMAS DETECTADOS:")
        for problem in problems:
            print(f"  {problem}")
    else:
        print("\nCalidad general es aceptable.")
    
    print(f"\nMétricas clave:")
    print(f"  - Ratio compresión: {ratios.mean():.3f} (target: 0.3-0.8)")
    print(f"  - Overlap léxico: {overlaps.mean():.3f} (target: <0.7)")
    print(f"  - Palabras nuevas: {diversity['new_words']/diversity['simple_unique']*100:.1f}% (target: >20%)")
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80)

if __name__ == '__main__':
    main()

