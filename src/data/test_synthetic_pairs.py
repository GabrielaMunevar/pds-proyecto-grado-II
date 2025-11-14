#!/usr/bin/env python3
"""
Script de Testing para Validar Mejoras en Pares Sintéticos
Compara la versión antigua vs nueva y valida métricas de calidad.

Uso:
    python src/data/test_synthetic_pairs.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from typing import Dict, List

# Importar funciones del módulo mejorado
from create_synthetic_pairs import (
    complexify_text,
    calculate_lexical_overlap,
    calculate_ngram_overlap,
    load_data,
    create_synthetic_pairs,
    analyze_pairs
)

def test_complexify_function():
    """Test básico de la función complexify_text."""
    print("\n" + "="*60)
    print("TEST 1: Función complexify_text()")
    print("="*60)
    
    # Texto de ejemplo simple
    simple_text = """
    This medicine helps people with high blood pressure. 
    It works by lowering blood pressure. 
    The study showed that it is safe and effective. 
    People who took the medicine had fewer heart problems.
    """
    
    print("\nTexto original (simple):")
    print(simple_text.strip())
    
    # Generar versión técnica
    technical_text = complexify_text(simple_text, complexity_level=0.85)
    
    print("\nTexto complejificado (técnico):")
    print(technical_text.strip())
    
    # Calcular métricas
    overlap = calculate_lexical_overlap(technical_text, simple_text)
    bigram_overlap = calculate_ngram_overlap(technical_text, simple_text, n=2)
    
    print(f"\n📊 Métricas:")
    print(f"   Overlap léxico: {overlap:.1%}")
    print(f"   Overlap bigramas: {bigram_overlap:.1%}")
    print(f"   Palabras original: {len(simple_text.split())}")
    print(f"   Palabras técnico: {len(technical_text.split())}")
    print(f"   Expansión: {len(technical_text.split()) / len(simple_text.split()):.2f}x")
    
    # Validar mejoras
    print(f"\n✅ Validaciones:")
    if 0.50 <= overlap <= 0.65:
        print(f"   ✅ Overlap en rango ideal (50-65%) - balance entre diversidad y naturalidad")
    elif overlap < 0.50:
        print(f"   ⚠️  Overlap muy bajo - puede indicar transformaciones demasiado agresivas")
    elif overlap > 0.65:
        print(f"   ⚠️  Overlap alto - considerar ajustes para más diversidad")
    
    if len(technical_text.split()) > len(simple_text.split()) * 1.1:
        print(f"   ✅ Texto técnico es más largo (expansión adecuada)")
    else:
        print(f"   ⚠️  Texto técnico no se expandió suficientemente")
    
    return {
        'overlap': overlap,
        'bigram_overlap': bigram_overlap,
        'expansion_ratio': len(technical_text.split()) / len(simple_text.split())
    }

def test_multiple_examples(n_examples: int = 10):
    """Test con múltiples ejemplos del dataset."""
    print("\n" + "="*60)
    print(f"TEST 2: Múltiples Ejemplos (n={n_examples})")
    print("="*60)
    
    # Cargar datos
    try:
        pls_data = load_data()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Saltando este test...")
        return None
    
    if len(pls_data) == 0:
        print("❌ No hay datos disponibles")
        return None
    
    # Tomar muestra aleatoria
    sample = pls_data.sample(min(n_examples, len(pls_data)), random_state=42)
    
    overlaps = []
    bigram_overlaps = []
    expansion_ratios = []
    
    print(f"\nProcesando {len(sample)} ejemplos...")
    
    for idx, row in sample.iterrows():
        simple_text = row['resumen']
        if pd.isna(simple_text) or len(str(simple_text)) < 20:
            continue
        
        technical_text = complexify_text(simple_text, complexity_level=0.85)
        
        overlap = calculate_lexical_overlap(technical_text, simple_text)
        bigram_overlap = calculate_ngram_overlap(technical_text, simple_text, n=2)
        expansion = len(technical_text.split()) / len(simple_text.split()) if len(simple_text.split()) > 0 else 0
        
        overlaps.append(overlap)
        bigram_overlaps.append(bigram_overlap)
        expansion_ratios.append(expansion)
    
    # Estadísticas
    print(f"\n📊 Estadísticas de {len(overlaps)} ejemplos:")
    print(f"   Overlap léxico promedio: {np.mean(overlaps):.1%}")
    print(f"   Overlap léxico mediano: {np.median(overlaps):.1%}")
    print(f"   Overlap léxico mínimo: {np.min(overlaps):.1%}")
    print(f"   Overlap léxico máximo: {np.max(overlaps):.1%}")
    print(f"   Overlap bigramas promedio: {np.mean(bigram_overlaps):.1%}")
    print(f"   Expansión promedio: {np.mean(expansion_ratios):.2f}x")
    
    # Validaciones
    avg_overlap = np.mean(overlaps)
    below_70 = sum(1 for o in overlaps if o < 0.70)
    below_60 = sum(1 for o in overlaps if o < 0.60)
    
    print(f"\n✅ Validaciones:")
    print(f"   Pares con overlap < 70%: {below_70}/{len(overlaps)} ({below_70/len(overlaps)*100:.1f}%)")
    print(f"   Pares con overlap < 60%: {below_60}/{len(overlaps)} ({below_60/len(overlaps)*100:.1f}%)")
    
    if 0.50 <= avg_overlap <= 0.65:
        print(f"   ✅ Overlap promedio en rango ideal (50-65%) - balance entre diversidad y naturalidad")
    elif avg_overlap < 0.50:
        print(f"   ⚠️  Overlap promedio muy bajo - puede indicar transformaciones demasiado agresivas")
    elif avg_overlap > 0.65:
        print(f"   ⚠️  Overlap promedio alto - considerar ajustes para más diversidad")
    
    return {
        'avg_overlap': avg_overlap,
        'median_overlap': np.median(overlaps),
        'min_overlap': np.min(overlaps),
        'max_overlap': np.max(overlaps),
        'pairs_below_70': below_70,
        'pairs_below_60': below_60,
        'total_pairs': len(overlaps)
    }

def test_full_dataset_sample(sample_size: int = 100):
    """Test con una muestra más grande del dataset completo."""
    print("\n" + "="*60)
    print(f"TEST 3: Muestra Grande del Dataset (n={sample_size})")
    print("="*60)
    
    # Cargar datos
    try:
        pls_data = load_data()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Saltando este test...")
        return None
    
    if len(pls_data) == 0:
        print("❌ No hay datos disponibles")
        return None
    
    # Tomar muestra
    sample = pls_data.sample(min(sample_size, len(pls_data)), random_state=42)
    
    # Crear pares sintéticos
    pairs = create_synthetic_pairs(sample, complexity_level=0.85)
    
    if len(pairs) == 0:
        print("❌ No se pudieron crear pares")
        return None
    
    # Analizar
    stats = analyze_pairs(pairs)
    
    return stats

def compare_with_old_version():
    """Compara con versión antigua si existe."""
    print("\n" + "="*60)
    print("TEST 4: Comparación con Versión Anterior")
    print("="*60)
    
    old_file = Path('data/processed/synthetic_pairs/synthetic_pairs_stats.json')
    new_file = Path('data/processed/synthetic_pairs_improved/synthetic_pairs_stats.json')
    
    if not old_file.exists():
        print("⚠️  No se encontró versión anterior para comparar")
        print("   (Ejecutar primero la versión antigua si existe)")
        return None
    
    if not new_file.exists():
        print("⚠️  No se encontró versión nueva para comparar")
        print("   (Ejecutar primero: python src/data/create_synthetic_pairs.py)")
        return None
    
    with open(old_file, 'r') as f:
        old_stats = json.load(f)
    
    with open(new_file, 'r') as f:
        new_stats = json.load(f)
    
    print("\n📊 Comparación:")
    print(f"\n   Total pares:")
    print(f"      Antes: {old_stats.get('total_pairs', 'N/A')}")
    print(f"      Ahora: {new_stats.get('total_pairs', 'N/A')}")
    
    if 'avg_lexical_overlap' in new_stats:
        print(f"\n   Overlap léxico promedio:")
        print(f"      Antes: ~94.5% (estimado)")
        print(f"      Ahora: {new_stats['avg_lexical_overlap']:.1%}")
        improvement = 0.945 - new_stats['avg_lexical_overlap']
        print(f"      Mejora: -{improvement:.1%} puntos")
    
    print(f"\n   Ratio compresión:")
    print(f"      Antes: {old_stats.get('avg_compression_ratio', 'N/A'):.2f}")
    print(f"      Ahora: {new_stats.get('avg_compression_ratio', 'N/A'):.2f}")
    
    return {
        'old_stats': old_stats,
        'new_stats': new_stats
    }

def run_all_tests():
    """Ejecuta todos los tests."""
    print("="*60)
    print("SUITE DE TESTS: Validación de Pares Sintéticos Mejorados")
    print("="*60)
    
    results = {}
    
    # Test 1: Función básica
    try:
        results['test1'] = test_complexify_function()
    except Exception as e:
        print(f"❌ Error en Test 1: {e}")
        results['test1'] = None
    
    # Test 2: Múltiples ejemplos
    try:
        results['test2'] = test_multiple_examples(n_examples=10)
    except Exception as e:
        print(f"❌ Error en Test 2: {e}")
        results['test2'] = None
    
    # Test 3: Muestra grande
    try:
        results['test3'] = test_full_dataset_sample(sample_size=100)
    except Exception as e:
        print(f"❌ Error en Test 3: {e}")
        results['test3'] = None
    
    # Test 4: Comparación
    try:
        results['test4'] = compare_with_old_version()
    except Exception as e:
        print(f"❌ Error en Test 4: {e}")
        results['test4'] = None
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    
    if results.get('test2'):
        t2 = results['test2']
        avg_overlap = t2.get('avg_overlap', 0)
        print(f"\n✅ Overlap léxico promedio: {avg_overlap:.1%}")
        print(f"✅ Objetivo: Balance entre overlap bajo y naturalidad (ideal: 50-65%)")
        if 0.50 <= avg_overlap <= 0.65:
            print(f"   🎉 RANGO IDEAL! (balance entre diversidad y naturalidad)")
        elif avg_overlap < 0.50:
            print(f"   ⚠️  Overlap muy bajo - puede indicar transformaciones demasiado agresivas")
        elif avg_overlap > 0.65:
            print(f"   ⚠️  Overlap alto - considerar ajustes para más diversidad")
        print(f"✅ Pares con overlap < 70%: {t2.get('pairs_below_70', 0)}/{t2.get('total_pairs', 0)}")
    
    print("\n" + "="*60)
    
    return results

if __name__ == "__main__":
    results = run_all_tests()
    
    # Exit code basado en resultados
    if results.get('test2'):
        avg_overlap = results['test2'].get('avg_overlap', 1.0)
        if 0.50 <= avg_overlap <= 0.65:
            sys.exit(0)  # Éxito - rango ideal
        elif avg_overlap < 0.50 or avg_overlap > 0.70:
            sys.exit(1)  # Fuera del rango aceptable
        else:
            sys.exit(0)  # Aceptable aunque no ideal
    else:
        sys.exit(1)  # Error en test

