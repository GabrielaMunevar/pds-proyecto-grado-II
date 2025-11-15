"""
Funcionalidades de chunking para T5.

Este módulo contiene funciones para dividir documentos largos en chunks
que quepan en la ventana de contexto de T5.
"""

from .chunker import (
    expand_pairs_with_chunking,
    chunk_technical_text,
    split_into_chunks_by_tokens,
    expand_pairs_with_chunking_by_tokens
)

__all__ = [
    'expand_pairs_with_chunking',
    'chunk_technical_text',
    'split_into_chunks_by_tokens',
    'expand_pairs_with_chunking_by_tokens'
]

