"""
Funciones de chunking específicas para T5.

Este módulo contiene funciones para dividir documentos largos en chunks
que quepan en la ventana de contexto de T5 (512 tokens).

Incluye dos estrategias:
1. Chunking basado en párrafos (de utils.text_chunking)
2. Chunking directo por tokens (más preciso, usado en Colab)
"""

from typing import List, Dict
from pathlib import Path
import sys
import numpy as np

# Importar utilidades
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils.text_chunking import split_into_chunks


def expand_pairs_with_chunking(pairs: List[Dict], tokenizer, max_tokens: int = 400) -> List[Dict]:
    """
    Expande pares largos en múltiples pares de chunks para entrenamiento.
    
    Si un documento técnico es muy largo, lo divide en chunks y crea
    múltiples ejemplos de entrenamiento. Cada chunk se empareja con
    el texto simple completo (si es corto) o con chunks del texto simple.
    
    Args:
        pairs: Lista de pares (texto_tecnico, texto_simple)
        tokenizer: Tokenizer para contar tokens
        max_tokens: Máximo de tokens por chunk
    
    Returns:
        Lista expandida de pares
    """
    expanded_pairs = []
    total_chunks = 0
    
    for pair in pairs:
        texto_tecnico = pair['texto_tecnico']
        texto_simple = pair['texto_simple']
        
        # Contar tokens del texto técnico
        tech_tokens = len(tokenizer.encode(texto_tecnico, add_special_tokens=False))
        simple_tokens = len(tokenizer.encode(texto_simple, add_special_tokens=False))
        
        # Si el texto técnico es corto, usar par original
        if tech_tokens <= max_tokens:
            expanded_pairs.append(pair)
        else:
            # Dividir texto técnico en chunks
            tech_chunks = split_into_chunks(
                texto_tecnico,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                overlap=50
            )
            
            # Si el texto simple también es largo, dividirlo
            if simple_tokens > max_tokens * 2:
                simple_chunks = split_into_chunks(
                    texto_simple,
                    tokenizer=tokenizer,
                    max_tokens=max_tokens * 2,  # Texto simple puede ser más largo
                    overlap=50
                )
                # Emparejar chunks técnicos con chunks simples (mismo índice o circular)
                for i, tech_chunk in enumerate(tech_chunks):
                    simple_chunk = simple_chunks[min(i, len(simple_chunks) - 1)]
                    expanded_pairs.append({
                        'texto_tecnico': tech_chunk,
                        'texto_simple': simple_chunk,
                        'split': pair.get('split', 'train'),
                        'source': pair.get('source', 'unknown'),
                        'is_chunk': True,
                        'chunk_idx': i,
                        'total_chunks': len(tech_chunks)
                    })
            else:
                # Usar texto simple completo para cada chunk técnico
                for i, tech_chunk in enumerate(tech_chunks):
                    expanded_pairs.append({
                        'texto_tecnico': tech_chunk,
                        'texto_simple': texto_simple,
                        'split': pair.get('split', 'train'),
                        'source': pair.get('source', 'unknown'),
                        'is_chunk': True,
                        'chunk_idx': i,
                        'total_chunks': len(tech_chunks)
                    })
            
            total_chunks += len(tech_chunks)
    
    print(f"Pares expandidos: {len(expanded_pairs)} (originales: {len(pairs)}, chunks creados: {total_chunks})")
    return expanded_pairs


def chunk_technical_text(text: str, tokenizer, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    """
    Divide un texto técnico en chunks.
    
    Args:
        text: Texto técnico a dividir
        tokenizer: Tokenizer para contar tokens
        max_tokens: Máximo de tokens por chunk
        overlap: Tokens de overlap entre chunks
    
    Returns:
        Lista de chunks
    """
    return split_into_chunks(
        text,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap=overlap
    )


def split_into_chunks_by_tokens(text: str, tokenizer, max_tokens: int = 450, overlap_tokens: int = 50) -> List[str]:
    """
    Divide texto en chunks con overlap, trabajando directamente con tokens.
    CRÍTICO: Cada chunk debe tener ≤ max_tokens tokens.
    
    Esta es la implementación exacta del Colab que trabaja directamente con tokens
    en lugar de texto, lo que es más preciso.
    
    Args:
        text: Texto a dividir
        tokenizer: Tokenizer para contar tokens
        max_tokens: Máximo de tokens por chunk (debe ser < 512 para dejar espacio al prompt)
        overlap_tokens: Tokens de overlap entre chunks
    
    Returns:
        Lista de chunks, cada uno con ≤ max_tokens tokens
    """
    # Tokenizar el texto completo
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Si el texto cabe en un chunk, retornarlo tal cual
    if len(tokens) <= max_tokens:
        return [text]
    
    # Dividir en chunks con overlap
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        # Calcular fin del chunk
        end_idx = min(start_idx + max_tokens, len(tokens))
        
        # Decodificar este chunk
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        # Siguiente chunk comienza con overlap (retroceder overlap_tokens)
        if end_idx < len(tokens):
            start_idx = end_idx - overlap_tokens
        else:
            break
    
    return chunks


def expand_pairs_with_chunking_by_tokens(
    pairs: List[Dict], 
    tokenizer, 
    max_tokens: int = 450, 
    overlap_tokens: int = 50
) -> List[Dict]:
    """
    Expande pares aplicando chunking a documentos largos (versión por tokens).
    Cada chunk se convierte en un ejemplo de entrenamiento.
    CRÍTICO: max_tokens debe ser < 512 para dejar espacio al prompt.
    
    Esta es la implementación exacta del Colab que trabaja directamente con tokens.
    
    Args:
        pairs: Lista de pares originales
        tokenizer: Tokenizer para contar tokens
        max_tokens: Máximo de tokens por chunk (450 deja ~60 para el prompt)
        overlap_tokens: Tokens de overlap entre chunks
    
    Returns:
        Lista expandida de pares, cada chunk es un par separado
    """
    print("\n" + "="*80)
    print("APLICANDO CHUNKING A DOCUMENTOS LARGOS (POR TOKENS)")
    print("="*80)
    print(f"  Max tokens por chunk: {max_tokens}")
    print(f"  Overlap entre chunks: {overlap_tokens} tokens")
    
    expanded_pairs = []
    chunked_count = 0
    total_chunks_created = 0
    
    for pair in pairs:
        tech_text = pair['texto_tecnico']
        
        # Verificar si necesita chunking (usando tokenizer para contar tokens reales)
        tech_tokens = len(tokenizer.encode(tech_text, add_special_tokens=False))
        
        if tech_tokens > max_tokens:
            # Dividir en chunks usando método por tokens
            chunks = split_into_chunks_by_tokens(
                tech_text, 
                tokenizer, 
                max_tokens=max_tokens, 
                overlap_tokens=overlap_tokens
            )
            chunked_count += 1
            total_chunks_created += len(chunks)
            
            # Crear un par por cada chunk (usar el mismo PLS para todos)
            # Estrategia: cada chunk aprende a generar el mismo PLS completo
            for i, chunk in enumerate(chunks):
                new_pair = pair.copy()
                new_pair['texto_tecnico'] = chunk
                new_pair['chunk_id'] = i
                new_pair['total_chunks'] = len(chunks)
                new_pair['original_tokens'] = tech_tokens
                new_pair['chunk_tokens'] = len(tokenizer.encode(chunk, add_special_tokens=False))
                expanded_pairs.append(new_pair)
        else:
            # No necesita chunking
            new_pair = pair.copy()
            new_pair['chunk_id'] = 0
            new_pair['total_chunks'] = 1
            new_pair['original_tokens'] = tech_tokens
            new_pair['chunk_tokens'] = tech_tokens
            expanded_pairs.append(new_pair)
    
    print(f"  Documentos originales: {len(pairs)}")
    print(f"  Documentos que requirieron chunking: {chunked_count} ({100*chunked_count/len(pairs):.1f}%)")
    print(f"  Total chunks creados: {total_chunks_created}")
    print(f"  Documentos después de chunking: {len(expanded_pairs)}")
    print(f"  Incremento: {len(expanded_pairs) - len(pairs)} documentos ({((len(expanded_pairs) - len(pairs)) / len(pairs) * 100):.1f}%)")
    
    # Verificar que los chunks realmente tienen ≤ max_tokens
    chunk_lengths = []
    for pair in expanded_pairs:
        chunk_tokens = len(tokenizer.encode(pair['texto_tecnico'], add_special_tokens=False))
        chunk_lengths.append(chunk_tokens)
    
    chunks_over_limit = sum(1 for l in chunk_lengths if l > max_tokens)
    print(f"\n  Verificación de chunks:")
    print(f"    Chunks > {max_tokens} tokens: {chunks_over_limit} ({100*chunks_over_limit/len(expanded_pairs):.1f}%)")
    print(f"    Promedio tokens por chunk: {np.mean(chunk_lengths):.0f}")
    print(f"    Máximo tokens en chunk: {np.max(chunk_lengths)}")
    
    if chunks_over_limit > 0:
        print(f"    ADVERTENCIA: {chunks_over_limit} chunks exceden el límite!")
    
    return expanded_pairs

