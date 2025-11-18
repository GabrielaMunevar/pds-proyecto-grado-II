"""
Script para analizar las longitudes del corpus de textos médicos.

Analiza el dataset completo y genera estadísticas detalladas sobre:
- Longitud en caracteres
- Longitud en palabras
- Distribución por fuente
- Distribución por split (train/test)
- Visualizaciones

Uso:
    python notebooks/analyze_corpus_lengths.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Rutas
DATA_PATH = Path("data/processed/dataset_clean.csv")

def load_data(filter_non_pls: bool = False, filter_empty: bool = True) -> pd.DataFrame:
    """Carga el dataset completo.
    
    Args:
        filter_non_pls: Si True, filtra solo textos con label='non_pls' (textos médicos originales)
        filter_empty: Si True, filtra textos vacíos (longitud 0)
    """
    print(f"Cargando dataset desde: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Dataset cargado: {len(df):,} registros")
    
    if filter_non_pls:
        df = df[df['label'] == 'non_pls'].copy()
        print(f"Filtrado a textos non_pls: {len(df):,} registros")
    
    if filter_empty:
        # Filtrar textos vacíos
        df = df[df['texto_original'].notna() & (df['texto_original'].str.len() > 0)].copy()
        print(f"Filtrado textos vacíos: {len(df):,} registros")
    
    return df

def calculate_length_metrics(text: str) -> Dict[str, int]:
    """Calcula métricas de longitud para un texto."""
    if pd.isna(text) or not isinstance(text, str):
        return {
            'chars': 0,
            'words': 0,
            'sentences': 0,
            'paragraphs': 0
        }
    
    chars = len(text)
    words = len(text.split())
    sentences = len([s for s in text.split('.') if s.strip()])
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    
    return {
        'chars': chars,
        'words': words,
        'sentences': sentences,
        'paragraphs': paragraphs
    }

def analyze_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Analiza las longitudes de todos los textos."""
    print("\nCalculando métricas de longitud...")
    
    # Aplicar a cada texto
    metrics_list = []
    for idx, row in df.iterrows():
        if idx % 5000 == 0:
            print(f"  Procesando registro {idx:,}/{len(df):,}")
        
        texto = row.get('texto_original', '')
        metrics = calculate_length_metrics(texto)
        metrics_list.append(metrics)
    
    # Crear DataFrame con métricas
    metrics_df = pd.DataFrame(metrics_list)
    
    # Combinar con información original
    result_df = df[['doc_id', 'source_dataset', 'source_bucket', 'split', 'label']].copy()
    result_df = pd.concat([result_df, metrics_df], axis=1)
    
    return result_df

def generate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Genera estadísticas descriptivas."""
    print("\nGenerando estadísticas descriptivas...")
    
    stats = {
        'total_texts': len(df),
        'overall': {
            'chars': {
                'mean': float(df['chars'].mean()),
                'median': float(df['chars'].median()),
                'std': float(df['chars'].std()),
                'min': int(df['chars'].min()),
                'max': int(df['chars'].max()),
                'q25': float(df['chars'].quantile(0.25)),
                'q75': float(df['chars'].quantile(0.75)),
                'q90': float(df['chars'].quantile(0.90)),
                'q95': float(df['chars'].quantile(0.95)),
                'q99': float(df['chars'].quantile(0.99)),
            },
            'words': {
                'mean': float(df['words'].mean()),
                'median': float(df['words'].median()),
                'std': float(df['words'].std()),
                'min': int(df['words'].min()),
                'max': int(df['words'].max()),
                'q25': float(df['words'].quantile(0.25)),
                'q75': float(df['words'].quantile(0.75)),
                'q90': float(df['words'].quantile(0.90)),
                'q95': float(df['words'].quantile(0.95)),
                'q99': float(df['words'].quantile(0.99)),
            },
            'sentences': {
                'mean': float(df['sentences'].mean()),
                'median': float(df['sentences'].median()),
                'std': float(df['sentences'].std()),
                'min': int(df['sentences'].min()),
                'max': int(df['sentences'].max()),
            },
            'paragraphs': {
                'mean': float(df['paragraphs'].mean()),
                'median': float(df['paragraphs'].median()),
                'std': float(df['paragraphs'].std()),
                'min': int(df['paragraphs'].min()),
                'max': int(df['paragraphs'].max()),
            }
        },
        'by_source': {},
        'by_split': {},
        'by_label': {}
    }
    
    # Por fuente
    for source in df['source_dataset'].unique():
        if pd.notna(source):
            source_df = df[df['source_dataset'] == source]
            stats['by_source'][str(source)] = {
                'count': int(len(source_df)),
                'chars_mean': float(source_df['chars'].mean()),
                'chars_median': float(source_df['chars'].median()),
                'words_mean': float(source_df['words'].mean()),
                'words_median': float(source_df['words'].median()),
            }
    
    # Por split
    for split in df['split'].unique():
        if pd.notna(split):
            split_df = df[df['split'] == split]
            stats['by_split'][str(split)] = {
                'count': int(len(split_df)),
                'chars_mean': float(split_df['chars'].mean()),
                'chars_median': float(split_df['chars'].median()),
                'words_mean': float(split_df['words'].mean()),
                'words_median': float(split_df['words'].median()),
            }
    
    # Por etiqueta
    for label in df['label'].unique():
        if pd.notna(label):
            label_df = df[df['label'] == label]
            stats['by_label'][str(label)] = {
                'count': int(len(label_df)),
                'chars_mean': float(label_df['chars'].mean()),
                'chars_median': float(label_df['chars'].median()),
                'words_mean': float(label_df['words'].mean()),
                'words_median': float(label_df['words'].median()),
            }
    
    return stats

def print_statistics(stats: Dict[str, Any]):
    """Imprime estadísticas en consola."""
    print("\n" + "="*80)
    print("ESTADÍSTICAS DE LONGITUD DEL CORPUS")
    print("="*80)
    
    print(f"\nTotal de textos: {stats['total_texts']:,}")
    
    print("\n--- LONGITUD EN CARACTERES ---")
    chars = stats['overall']['chars']
    print(f"  Media:      {chars['mean']:,.0f}")
    print(f"  Mediana:    {chars['median']:,.0f}")
    print(f"  Desv. Est.: {chars['std']:,.0f}")
    print(f"  Mínimo:     {chars['min']:,}")
    print(f"  Máximo:     {chars['max']:,}")
    print(f"  Q25:        {chars['q25']:,.0f}")
    print(f"  Q75:        {chars['q75']:,.0f}")
    print(f"  Q90:        {chars['q90']:,.0f}")
    print(f"  Q95:        {chars['q95']:,.0f}")
    print(f"  Q99:        {chars['q99']:,.0f}")
    
    print("\n--- LONGITUD EN PALABRAS ---")
    words = stats['overall']['words']
    print(f"  Media:      {words['mean']:,.0f}")
    print(f"  Mediana:    {words['median']:,.0f}")
    print(f"  Desv. Est.: {words['std']:,.0f}")
    print(f"  Mínimo:     {words['min']:,}")
    print(f"  Máximo:     {words['max']:,}")
    print(f"  Q25:        {words['q25']:,.0f}")
    print(f"  Q75:        {words['q75']:,.0f}")
    print(f"  Q90:        {words['q90']:,.0f}")
    print(f"  Q95:        {words['q95']:,.0f}")
    print(f"  Q99:        {words['q99']:,.0f}")
    
    print("\n--- POR FUENTE ---")
    for source, data in stats['by_source'].items():
        print(f"  {source}:")
        print(f"    Cantidad: {data['count']:,}")
        print(f"    Caracteres (media/mediana): {data['chars_mean']:,.0f} / {data['chars_median']:,.0f}")
        print(f"    Palabras (media/mediana): {data['words_mean']:,.0f} / {data['words_median']:,.0f}")
    
    print("\n--- POR SPLIT ---")
    for split, data in stats['by_split'].items():
        print(f"  {split}:")
        print(f"    Cantidad: {data['count']:,}")
        print(f"    Caracteres (media/mediana): {data['chars_mean']:,.0f} / {data['chars_median']:,.0f}")
        print(f"    Palabras (media/mediana): {data['words_mean']:,.0f} / {data['words_median']:,.0f}")
    
    print("\n--- POR ETIQUETA ---")
    for label, data in stats['by_label'].items():
        print(f"  {label}:")
        print(f"    Cantidad: {data['count']:,}")
        print(f"    Caracteres (media/mediana): {data['chars_mean']:,.0f} / {data['chars_median']:,.0f}")
        print(f"    Palabras (media/mediana): {data['words_mean']:,.0f} / {data['words_median']:,.0f}")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analiza longitudes del corpus de textos médicos')
    parser.add_argument('--non-pls-only', action='store_true', 
                       help='Analizar solo textos médicos originales (non_pls)')
    parser.add_argument('--include-empty', action='store_true',
                       help='Incluir textos vacíos en el análisis')
    args = parser.parse_args()
    
    print("="*80)
    print("ANÁLISIS DE LONGITUDES DEL CORPUS DE TEXTOS MÉDICOS")
    print("="*80)
    
    # Cargar datos
    df = load_data(filter_non_pls=args.non_pls_only, filter_empty=not args.include_empty)
    
    # Analizar longitudes
    df_with_metrics = analyze_lengths(df)
    
    # Generar estadísticas
    stats = generate_statistics(df_with_metrics)
    
    # Imprimir estadísticas
    print_statistics(stats)
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)

if __name__ == "__main__":
    main()

