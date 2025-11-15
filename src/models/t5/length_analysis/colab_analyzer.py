"""
Análisis de longitudes específico del Colab.

Este módulo contiene la función analyze_document_lengths exacta del Colab.
"""

import numpy as np
from typing import List, Dict


def analyze_document_lengths(pairs: List[Dict], tokenizer, max_length: int = 512) -> Dict:
    """
    Analiza la distribución de longitudes de documentos.
    CRÍTICO: Identificar cuántos documentos exceden la ventana de contexto.
    
    Esta es la implementación exacta del Colab.
    
    Args:
        pairs: Lista de pares (texto_tecnico, texto_simple)
        tokenizer: Tokenizer para contar tokens
        max_length: Longitud máxima de tokens permitida
    
    Returns:
        Diccionario con estadísticas de longitudes
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE LONGITUDES DE DOCUMENTOS")
    print("="*80)
    
    technical_lengths = []
    simple_lengths = []
    truncated_count = 0
    
    for pair in pairs:
        tech_tokens = len(tokenizer.encode(pair['texto_tecnico']))
        simple_tokens = len(tokenizer.encode(pair['texto_simple']))
        
        technical_lengths.append(tech_tokens)
        simple_lengths.append(simple_tokens)
        
        if tech_tokens > max_length:
            truncated_count += 1
    
    print(f"\n ESTADÍSTICAS DE LONGITUD:")
    print(f"  Textos técnicos:")
    print(f"    - Promedio: {np.mean(technical_lengths):.0f} tokens")
    print(f"    - Mediana: {np.median(technical_lengths):.0f} tokens")
    print(f"    - Mínimo: {np.min(technical_lengths)} tokens")
    print(f"    - Máximo: {np.max(technical_lengths)} tokens")
    print(f"    - Documentos que exceden {max_length} tokens: {truncated_count} ({100*truncated_count/len(pairs):.1f}%)")
    
    if truncated_count > 0:
        tokens_lost = np.mean([max(0, l - max_length) for l in technical_lengths])
        print(f"    - Tokens perdidos promedio por truncación: {tokens_lost:.0f}")
    
    print(f"\n  Textos simples (PLS):")
    print(f"    - Promedio: {np.mean(simple_lengths):.0f} tokens")
    print(f"    - Mediana: {np.median(simple_lengths):.0f} tokens")
    print(f"    - Máximo: {np.max(simple_lengths)} tokens")
    
    return {
        'technical': {
            'mean': float(np.mean(technical_lengths)),
            'median': float(np.median(technical_lengths)),
            'max': int(np.max(technical_lengths)),
            'truncated': truncated_count,
            'truncated_pct': float(100*truncated_count/len(pairs))
        },
        'simple': {
            'mean': float(np.mean(simple_lengths)),
            'median': float(np.median(simple_lengths)),
            'max': int(np.max(simple_lengths))
        }
    }

