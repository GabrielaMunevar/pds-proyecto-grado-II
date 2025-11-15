"""
Análisis de longitudes de documentos para T5.

Este módulo proporciona funciones específicas para analizar longitudes
de documentos en el contexto del modelo T5.
"""

from typing import List, Dict
from pathlib import Path
import sys

# Importar utilidades
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils.length_analysis import analyze_lengths_for_training, analyze_document_lengths, print_length_analysis_report


def analyze_lengths_for_t5(
    pairs: List[Dict],
    tokenizer,
    max_length_source: int = 400,
    save_report: bool = True,
    output_dir: Path = None
) -> Dict:
    """
    Analiza longitudes de documentos para entrenamiento con T5.
    
    Args:
        pairs: Lista de pares para entrenamiento
        tokenizer: Tokenizer del modelo T5
        max_length_source: Longitud máxima considerando el prompt
        save_report: Si True, guarda el reporte en JSON
        output_dir: Directorio donde guardar el reporte
    
    Returns:
        Diccionario con resultados del análisis
    """
    return analyze_lengths_for_training(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length_source=max_length_source,
        save_report=save_report,
        output_dir=output_dir
    )


def analyze_lengths_before_after_chunking(
    pairs_before: List[Dict],
    pairs_after: List[Dict],
    tokenizer,
    max_length_source: int = 400,
    output_dir: Path = None
) -> Dict:
    """
    Compara análisis de longitudes antes y después del chunking.
    
    Args:
        pairs_before: Pares originales (antes de chunking)
        pairs_after: Pares después de chunking
        tokenizer: Tokenizer del modelo
        max_length_source: Longitud máxima considerando el prompt
        output_dir: Directorio donde guardar reportes
    
    Returns:
        Diccionario con comparación de análisis
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE LONGITUDES ANTES DEL CHUNKING")
    print("="*80)
    
    analysis_before = analyze_lengths_for_training(
        pairs=pairs_before,
        tokenizer=tokenizer,
        max_length_source=max_length_source,
        save_report=True,
        output_dir=output_dir / 'before_chunking' if output_dir else None
    )
    
    print("\n" + "="*80)
    print("ANÁLISIS DE LONGITUDES DESPUÉS DEL CHUNKING")
    print("="*80)
    
    analysis_after = analyze_lengths_for_training(
        pairs=pairs_after,
        tokenizer=tokenizer,
        max_length_source=max_length_source,
        save_report=True,
        output_dir=output_dir / 'after_chunking' if output_dir else None
    )
    
    print(f"\n COMPARACIÓN:")
    print(f"  Documentos originales: {len(pairs_before)}")
    print(f"  Documentos después de chunking: {len(pairs_after)}")
    print(f"  Incremento: {len(pairs_after) - len(pairs_before)} documentos ({((len(pairs_after) - len(pairs_before)) / len(pairs_before) * 100):.1f}%)")
    print(f"  Documentos truncados antes: {analysis_before['technical']['truncation']['num_truncated']}")
    print(f"  Documentos truncados después: {analysis_after['technical']['truncation']['num_truncated']}")
    
    return {
        'before': analysis_before,
        'after': analysis_after,
        'comparison': {
            'original_count': len(pairs_before),
            'expanded_count': len(pairs_after),
            'increase': len(pairs_after) - len(pairs_before),
            'increase_percentage': ((len(pairs_after) - len(pairs_before)) / len(pairs_before) * 100) if pairs_before else 0,
            'truncated_before': analysis_before['technical']['truncation']['num_truncated'],
            'truncated_after': analysis_after['technical']['truncation']['num_truncated']
        }
    }

