"""
Entrenamiento del modelo T5.

Este módulo contiene funciones para entrenar el modelo T5.
Incluye tanto la versión básica como la versión completa del Colab.
"""

from .trainer import train_t5_generator, train_t5_large, load_synthetic_pairs, tokenize_function

__all__ = ['train_t5_generator', 'train_t5_large', 'load_synthetic_pairs', 'tokenize_function']

