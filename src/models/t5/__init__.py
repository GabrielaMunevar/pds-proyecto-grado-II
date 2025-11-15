"""
Módulo T5 para generación de Plain Language Summaries.

Este módulo contiene todas las funcionalidades relacionadas con el modelo T5:
- Entrenamiento
- Evaluación
- Chunking de documentos largos
- Análisis de longitudes

Estructura:
- training/: Funciones de entrenamiento del modelo
- evaluation/: Funciones de evaluación y generación de PLS
- chunking/: Funciones para dividir documentos largos en chunks
- length_analysis/: Análisis de longitudes de documentos
"""

# Importaciones principales para facilitar el uso
from .training.trainer import (
    train_t5_generator, 
    train_t5_large, 
    load_synthetic_pairs, 
    tokenize_function
)
from .evaluation.evaluator import evaluate_t5_generator, generate_pls, load_t5_model
from .evaluation.metrics import compute_metrics
from .chunking.chunker import (
    expand_pairs_with_chunking, 
    chunk_technical_text,
    split_into_chunks_by_tokens,
    expand_pairs_with_chunking_by_tokens
)
from .length_analysis.analyzer import analyze_lengths_for_t5, analyze_lengths_before_after_chunking
from .length_analysis.colab_analyzer import analyze_document_lengths

__all__ = [
    # Training
    'train_t5_generator',
    'train_t5_large',  # Versión completa del Colab
    'load_synthetic_pairs',
    'tokenize_function',
    # Evaluation
    'evaluate_t5_generator',
    'generate_pls',
    'load_t5_model',
    'compute_metrics',  # Métricas durante entrenamiento
    # Chunking
    'expand_pairs_with_chunking',
    'chunk_technical_text',
    'split_into_chunks_by_tokens',  # Versión del Colab
    'expand_pairs_with_chunking_by_tokens',  # Versión del Colab
    # Length Analysis
    'analyze_lengths_for_t5',
    'analyze_lengths_before_after_chunking',
    'analyze_document_lengths'  # Versión del Colab
]

