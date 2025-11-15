"""
Evaluación del modelo T5.

Este módulo contiene funciones para evaluar el modelo T5 generando PLS
y calculando métricas de calidad.
"""

from .evaluator import evaluate_t5_generator, generate_pls, load_t5_model

__all__ = ['evaluate_t5_generator', 'generate_pls', 'load_t5_model']

