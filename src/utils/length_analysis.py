"""
Análisis de longitudes de documentos.

Este módulo proporciona funciones para analizar la distribución de longitudes
de documentos y calcular cuánta información se pierde por truncation.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


def analyze_document_lengths(
    pairs: List[Dict],
    tokenizer,
    max_length_source: int = 512,
    include_prompt: bool = True,
    prompt: Optional[str] = None
) -> Dict:
    """
    Analiza la distribución de longitudes de documentos y calcula pérdida por truncation.
    
    Args:
        pairs: Lista de pares (texto_tecnico, texto_simple)
        tokenizer: Tokenizer para contar tokens
        max_length_source: Longitud máxima de tokens permitida (ventana de contexto)
        include_prompt: Si True, incluye el prompt en el conteo de tokens
        prompt: Prompt a usar si include_prompt es True. Si es None, usa el estándar.
    
    Returns:
        Diccionario con estadísticas de longitudes
    """
    from config import get_prompt, apply_prompt
    
    if prompt is None:
        prompt = get_prompt()
    
    lengths_technical = []
    lengths_simple = []
    truncated_technical = []
    truncated_simple = []
    
    print("Analizando longitudes de documentos...")
    
    for pair in tqdm(pairs, desc="Analizando"):
        texto_tecnico = pair.get('texto_tecnico', '')
        texto_simple = pair.get('texto_simple', '')
        
        # Contar tokens del texto técnico (con o sin prompt)
        if include_prompt:
            tech_text_with_prompt = apply_prompt(texto_tecnico, prompt)
            tech_tokens = len(tokenizer.encode(tech_text_with_prompt, add_special_tokens=False))
        else:
            tech_tokens = len(tokenizer.encode(texto_tecnico, add_special_tokens=False))
        
        # Contar tokens del texto simple
        simple_tokens = len(tokenizer.encode(texto_simple, add_special_tokens=False))
        
        lengths_technical.append(tech_tokens)
        lengths_simple.append(simple_tokens)
        
        # Verificar truncation
        if tech_tokens > max_length_source:
            truncated_technical.append({
                'tokens': tech_tokens,
                'lost_tokens': tech_tokens - max_length_source,
                'lost_percentage': ((tech_tokens - max_length_source) / tech_tokens) * 100
            })
        
        # Para texto simple, usar max_length_target (típicamente 256)
        max_length_target = 256
        if simple_tokens > max_length_target:
            truncated_simple.append({
                'tokens': simple_tokens,
                'lost_tokens': simple_tokens - max_length_target,
                'lost_percentage': ((simple_tokens - max_length_target) / simple_tokens) * 100
            })
    
    # Calcular estadísticas
    lengths_technical = np.array(lengths_technical)
    lengths_simple = np.array(lengths_simple)
    
    # Estadísticas texto técnico
    stats_technical = {
        'mean': float(np.mean(lengths_technical)),
        'median': float(np.median(lengths_technical)),
        'std': float(np.std(lengths_technical)),
        'min': int(np.min(lengths_technical)),
        'max': int(np.max(lengths_technical)),
        'p25': float(np.percentile(lengths_technical, 25)),
        'p75': float(np.percentile(lengths_technical, 75)),
        'p90': float(np.percentile(lengths_technical, 90)),
        'p95': float(np.percentile(lengths_technical, 95)),
        'p99': float(np.percentile(lengths_technical, 99))
    }
    
    # Estadísticas texto simple
    stats_simple = {
        'mean': float(np.mean(lengths_simple)),
        'median': float(np.median(lengths_simple)),
        'std': float(np.std(lengths_simple)),
        'min': int(np.min(lengths_simple)),
        'max': int(np.max(lengths_simple)),
        'p25': float(np.percentile(lengths_simple, 25)),
        'p75': float(np.percentile(lengths_simple, 75)),
        'p90': float(np.percentile(lengths_simple, 90)),
        'p95': float(np.percentile(lengths_simple, 95)),
        'p99': float(np.percentile(lengths_simple, 99))
    }
    
    # Análisis de truncation
    num_truncated_technical = len(truncated_technical)
    num_truncated_simple = len(truncated_simple)
    
    if truncated_technical:
        avg_lost_tokens_technical = np.mean([t['lost_tokens'] for t in truncated_technical])
        avg_lost_percentage_technical = np.mean([t['lost_percentage'] for t in truncated_technical])
        total_lost_tokens_technical = sum([t['lost_tokens'] for t in truncated_technical])
    else:
        avg_lost_tokens_technical = 0.0
        avg_lost_percentage_technical = 0.0
        total_lost_tokens_technical = 0
    
    if truncated_simple:
        avg_lost_tokens_simple = np.mean([t['lost_tokens'] for t in truncated_simple])
        avg_lost_percentage_simple = np.mean([t['lost_percentage'] for t in truncated_simple])
        total_lost_tokens_simple = sum([t['lost_tokens'] for t in truncated_simple])
    else:
        avg_lost_tokens_simple = 0.0
        avg_lost_percentage_simple = 0.0
        total_lost_tokens_simple = 0
    
    # Calcular tokens perdidos promedio (incluyendo documentos no truncados)
    all_lost_tokens_technical = [max(0, l - max_length_source) for l in lengths_technical]
    avg_lost_tokens_all_technical = float(np.mean(all_lost_tokens_technical))
    
    all_lost_tokens_simple = [max(0, l - 256) for l in lengths_simple]
    avg_lost_tokens_all_simple = float(np.mean(all_lost_tokens_simple))
    
    results = {
        'total_documents': len(pairs),
        'max_length_source': max_length_source,
        'max_length_target': 256,
        'include_prompt': include_prompt,
        'prompt_used': prompt if include_prompt else None,
        
        # Estadísticas texto técnico
        'technical': {
            'stats': stats_technical,
            'truncation': {
                'num_truncated': num_truncated_technical,
                'percentage_truncated': (num_truncated_technical / len(pairs)) * 100 if pairs else 0,
                'avg_lost_tokens': avg_lost_tokens_technical,
                'avg_lost_percentage': avg_lost_percentage_technical,
                'total_lost_tokens': int(total_lost_tokens_technical),
                'avg_lost_tokens_all_docs': avg_lost_tokens_all_technical
            }
        },
        
        # Estadísticas texto simple
        'simple': {
            'stats': stats_simple,
            'truncation': {
                'num_truncated': num_truncated_simple,
                'percentage_truncated': (num_truncated_simple / len(pairs)) * 100 if pairs else 0,
                'avg_lost_tokens': avg_lost_tokens_simple,
                'avg_lost_percentage': avg_lost_percentage_simple,
                'total_lost_tokens': int(total_lost_tokens_simple),
                'avg_lost_tokens_all_docs': avg_lost_tokens_all_simple
            }
        },
        
        # Distribuciones completas (para visualización)
        'distributions': {
            'technical_lengths': lengths_technical.tolist(),
            'simple_lengths': lengths_simple.tolist()
        }
    }
    
    return results


def print_length_analysis_report(results: Dict):
    """
    Imprime un reporte detallado del análisis de longitudes.
    
    Args:
        results: Resultados de analyze_document_lengths()
    """
    print("\n" + "="*80)
    print("ANÁLISIS DE LONGITUDES DE DOCUMENTOS")
    print("="*80)
    
    print(f"\n📊 RESUMEN GENERAL")
    print(f"  Total de documentos: {results['total_documents']}")
    print(f"  Ventana de contexto (source): {results['max_length_source']} tokens")
    print(f"  Ventana de contexto (target): {results['max_length_target']} tokens")
    if results['include_prompt']:
        print(f"  Prompt incluido: {results['prompt_used']}")
    
    # Texto técnico
    print(f"\n📝 TEXTO TÉCNICO (Input)")
    print(f"  Longitud promedio: {results['technical']['stats']['mean']:.0f} tokens")
    print(f"  Longitud mediana: {results['technical']['stats']['median']:.0f} tokens")
    print(f"  Desviación estándar: {results['technical']['stats']['std']:.0f} tokens")
    print(f"  Rango: {results['technical']['stats']['min']} - {results['technical']['stats']['max']} tokens")
    print(f"  Percentiles:")
    print(f"    P25: {results['technical']['stats']['p25']:.0f} tokens")
    print(f"    P75: {results['technical']['stats']['p75']:.0f} tokens")
    print(f"    P90: {results['technical']['stats']['p90']:.0f} tokens")
    print(f"    P95: {results['technical']['stats']['p95']:.0f} tokens")
    print(f"    P99: {results['technical']['stats']['p99']:.0f} tokens")
    
    print(f"\n  ⚠️  TRUNCATION:")
    trunc_tech = results['technical']['truncation']
    print(f"    Documentos que exceden {results['max_length_source']} tokens: "
          f"{trunc_tech['num_truncated']} ({trunc_tech['percentage_truncated']:.1f}%)")
    print(f"    Tokens perdidos promedio (solo truncados): {trunc_tech['avg_lost_tokens']:.0f} tokens")
    print(f"    Porcentaje perdido promedio (solo truncados): {trunc_tech['avg_lost_percentage']:.1f}%")
    print(f"    Tokens perdidos promedio (todos los documentos): {trunc_tech['avg_lost_tokens_all_docs']:.0f} tokens")
    print(f"    Total de tokens perdidos: {trunc_tech['total_lost_tokens']:,} tokens")
    
    # Texto simple
    print(f"\n📄 TEXTO SIMPLE (Target)")
    print(f"  Longitud promedio: {results['simple']['stats']['mean']:.0f} tokens")
    print(f"  Longitud mediana: {results['simple']['stats']['median']:.0f} tokens")
    print(f"  Desviación estándar: {results['simple']['stats']['std']:.0f} tokens")
    print(f"  Rango: {results['simple']['stats']['min']} - {results['simple']['stats']['max']} tokens")
    
    print(f"\n  ⚠️  TRUNCATION:")
    trunc_simple = results['simple']['truncation']
    print(f"    Documentos que exceden {results['max_length_target']} tokens: "
          f"{trunc_simple['num_truncated']} ({trunc_simple['percentage_truncated']:.1f}%)")
    print(f"    Tokens perdidos promedio (solo truncados): {trunc_simple['avg_lost_tokens']:.0f} tokens")
    print(f"    Porcentaje perdido promedio (solo truncados): {trunc_simple['avg_lost_percentage']:.1f}%")
    print(f"    Tokens perdidos promedio (todos los documentos): {trunc_simple['avg_lost_tokens_all_docs']:.0f} tokens")
    print(f"    Total de tokens perdidos: {trunc_simple['total_lost_tokens']:,} tokens")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    if trunc_tech['percentage_truncated'] > 50:
        print(f"    ⚠️  Más del 50% de documentos se truncan. Considerar:")
        print(f"       - Usar chunking (ya implementado)")
        print(f"       - Aumentar max_length_source si es posible")
        print(f"       - Usar modelo con ventana de contexto más grande")
    elif trunc_tech['percentage_truncated'] > 20:
        print(f"    ⚠️  Más del 20% de documentos se truncan. El chunking ayudará.")
    else:
        print(f"    ✅ Menos del 20% de documentos se truncan. Situación manejable.")
    
    if trunc_tech['avg_lost_tokens_all_docs'] > 100:
        print(f"    ⚠️  Se pierden más de 100 tokens en promedio. Chunking crítico.")
    elif trunc_tech['avg_lost_tokens_all_docs'] > 50:
        print(f"    ⚠️  Se pierden más de 50 tokens en promedio. Chunking recomendado.")
    else:
        print(f"    ✅ Pérdida de tokens manejable.")
    
    print("="*80)


def save_length_analysis(results: Dict, output_path: Path):
    """
    Guarda el análisis de longitudes en un archivo JSON.
    
    Args:
        results: Resultados de analyze_document_lengths()
        output_path: Ruta donde guardar el archivo
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Crear versión sin distribuciones completas (muy grandes)
    results_to_save = results.copy()
    results_to_save['distributions'] = {
        'technical_lengths': 'removed_for_size',
        'simple_lengths': 'removed_for_size',
        'note': 'Full distributions removed to reduce file size. Use analyze_document_lengths() to regenerate.'
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Análisis guardado en: {output_path}")


def analyze_lengths_for_training(
    pairs: List[Dict],
    tokenizer,
    max_length_source: int = 400,  # Considerando el prompt
    save_report: bool = True,
    output_dir: Path = None
) -> Dict:
    """
    Función conveniente para analizar longitudes antes del entrenamiento.
    
    Args:
        pairs: Lista de pares para entrenamiento
        tokenizer: Tokenizer del modelo
        max_length_source: Longitud máxima considerando el prompt
        save_report: Si True, guarda el reporte en JSON
        output_dir: Directorio donde guardar el reporte
    
    Returns:
        Diccionario con resultados del análisis
    """
    # Analizar
    results = analyze_document_lengths(
        pairs=pairs,
        tokenizer=tokenizer,
        max_length_source=max_length_source,
        include_prompt=True
    )
    
    # Imprimir reporte
    print_length_analysis_report(results)
    
    # Guardar si se solicita
    if save_report and output_dir:
        output_path = output_dir / 'length_analysis.json'
        save_length_analysis(results, output_path)
    
    return results

