"""
Análisis de longitudes para T5.

Este módulo contiene funciones para analizar la distribución de longitudes
de documentos y calcular cuánta información se pierde por truncation.
"""

from .analyzer import analyze_lengths_for_t5, analyze_lengths_before_after_chunking

__all__ = ['analyze_lengths_for_t5', 'analyze_lengths_before_after_chunking']

