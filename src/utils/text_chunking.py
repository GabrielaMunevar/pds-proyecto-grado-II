"""
Utilidades para chunking de documentos largos.

Este módulo implementa funciones para dividir textos largos en chunks
que quepan en la ventana de contexto de los modelos, preservando
el contexto mediante overlap entre chunks.
"""

from typing import List, Optional, Callable
import re


def split_into_chunks(
    text: str,
    tokenizer: Optional[Callable] = None,
    max_tokens: int = 400,
    overlap: int = 50,
    min_chunk_size: int = 50
) -> List[str]:
    """
    Divide texto en chunks con overlap para preservar contexto.
    
    Esta función divide el texto técnico en chunks que quepan en la ventana
    de contexto del modelo, preservando el contexto mediante overlap entre
    chunks consecutivos.
    
    Args:
        text: Texto a dividir en chunks
        tokenizer: Tokenizer opcional para contar tokens exactos. Si es None,
                   usa estimación basada en palabras (1 token ≈ 0.75 palabras)
        max_tokens: Número máximo de tokens por chunk (default: 400)
        overlap: Número de tokens de overlap entre chunks (default: 50)
        min_chunk_size: Tamaño mínimo de chunk en caracteres (default: 50)
    
    Returns:
        Lista de chunks de texto
    
    Example:
        >>> chunks = split_into_chunks(long_text, tokenizer, max_tokens=400)
        >>> for chunk in chunks:
        ...     process_chunk(chunk)
    """
    if not text or len(text.strip()) < min_chunk_size:
        return [text] if text else []
    
    # Dividir por párrafos primero (preserva estructura semántica)
    paragraphs = _split_into_paragraphs(text)
    
    if not paragraphs:
        return [text]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        # Contar tokens del párrafo
        if tokenizer:
            try:
                para_tokens = len(tokenizer.encode(para, add_special_tokens=False))
            except:
                # Fallback a estimación si tokenizer falla
                para_tokens = _estimate_tokens(para)
        else:
            para_tokens = _estimate_tokens(para)
        
        # Si el párrafo solo excede el límite, dividirlo por oraciones
        if para_tokens > max_tokens:
            # Guardar chunk actual si existe
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Dividir párrafo largo en oraciones
            sentences = _split_into_sentences(para)
            for sent in sentences:
                sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False)) if tokenizer else _estimate_tokens(sent)
                
                if current_tokens + sent_tokens > max_tokens:
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        # Overlap: mantener última oración
                        if len(current_chunk) > 1:
                            current_chunk = [current_chunk[-1]]
                            current_tokens = len(tokenizer.encode(current_chunk[0], add_special_tokens=False)) if tokenizer else _estimate_tokens(current_chunk[0])
                        else:
                            current_chunk = []
                            current_tokens = 0
                
                current_chunk.append(sent)
                current_tokens += sent_tokens
        
        # Si agregar este párrafo excede el límite
        elif current_tokens + para_tokens > max_tokens:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                # Overlap: mantener último párrafo para preservar contexto
                if len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1]]
                    current_tokens = len(tokenizer.encode(current_chunk[0], add_special_tokens=False)) if tokenizer else _estimate_tokens(current_chunk[0])
                else:
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(para)
            current_tokens = para_tokens
        else:
            # Agregar párrafo al chunk actual
            current_chunk.append(para)
            current_tokens += para_tokens
    
    # Agregar último chunk si existe
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    # Si no se generaron chunks (texto muy corto), devolver texto original
    if not chunks:
        return [text]
    
    return chunks


def process_long_document(
    text: str,
    tokenizer: Optional[Callable] = None,
    max_tokens: int = 400,
    process_fn: Optional[Callable] = None
) -> List[str]:
    """
    Procesa un documento largo dividiéndolo en chunks y aplicando una función.
    
    Útil para generar PLS por chunks y luego combinar los resultados.
    
    Args:
        text: Texto largo a procesar
        tokenizer: Tokenizer para contar tokens
        max_tokens: Máximo de tokens por chunk
        process_fn: Función opcional para procesar cada chunk. Si es None,
                    solo devuelve los chunks
    
    Returns:
        Lista de resultados procesados (o chunks si process_fn es None)
    """
    chunks = split_into_chunks(text, tokenizer, max_tokens)
    
    if process_fn:
        results = []
        for chunk in chunks:
            result = process_fn(chunk)
            results.append(result)
        return results
    
    return chunks


def _split_into_paragraphs(text: str) -> List[str]:
    """Divide texto en párrafos por doble salto de línea."""
    # Dividir por doble salto de línea
    paragraphs = re.split(r'\n\s*\n+', text)
    
    # Limpiar y filtrar párrafos vacíos o muy cortos
    valid_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if para and len(para.split()) >= 5:  # Mínimo 5 palabras
            valid_paragraphs.append(para)
    
    return valid_paragraphs if valid_paragraphs else [text]


def _split_into_sentences(text: str) -> List[str]:
    """Divide texto en oraciones usando regex simple."""
    # Patrón para dividir por puntuación de fin de oración
    sentences = re.split(r'([.!?]+\s+)', text)
    
    # Reconstruir oraciones con su puntuación
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i + 1]).strip()
        else:
            sentence = sentences[i].strip()
        
        if sentence and len(sentence.split()) >= 3:  # Mínimo 3 palabras
            result.append(sentence)
    
    # Si no se encontraron oraciones, devolver texto completo
    return result if result else [text]


def _estimate_tokens(text: str) -> int:
    """
    Estima número de tokens basado en palabras.
    
    Aproximación: 1 token ≈ 0.75 palabras (promedio para inglés)
    """
    word_count = len(text.split())
    return int(word_count / 0.75)

