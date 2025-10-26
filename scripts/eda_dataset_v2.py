"""
EDA Completo del Dataset Clean V2
Genera análisis exhaustivo y visualizaciones del dataset procesado.

Uso:
    python scripts/eda_dataset_v2.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Configuración
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
pd.options.display.float_format = '{:.2f}'.format

# Paths
DATA_PATH = Path("data/processed/dataset_clean.csv")
STATS_PATH = Path("data/processed/dataset_clean_stats.json")
OUTPUT_DIR = Path("reports/eda_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_header(title):
    """Imprime encabezado formateado."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def load_data():
    """Carga el dataset y estadísticas."""
    print_header("📊 EDA COMPLETO - DATASET CLEAN")
    
    # Cargar dataset
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # Cargar estadísticas
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
    
    print(f"✅ Dataset cargado: {len(df):,} registros")
    print(f"✅ Estadísticas cargadas")
    print(f"   Dimensiones: {df.shape}")
    
    return df, stats


def analyze_structure(df):
    """Analiza la estructura básica del dataset."""
    print_header("1️⃣ ESTRUCTURA DEL DATASET")
    
    print("📋 COLUMNAS Y TIPOS:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        null_pct = nulls / len(df) * 100
        print(f"  {i:2d}. {col:20s} ({dtype}) - Nulos: {nulls:6,} ({null_pct:5.1f}%)")
    
    # Valores nulos
    print("\n🔍 VALORES NULOS:")
    nulls = df.isnull().sum()
    nulls_pct = (nulls / len(df) * 100).round(1)
    null_df = pd.DataFrame({'Nulos': nulls, 'Porcentaje': nulls_pct})
    null_df = null_df[null_df['Nulos'] > 0].sort_values('Nulos', ascending=False)
    
    if len(null_df) > 0:
        print(null_df.to_string())
    else:
        print("  ✅ No hay valores nulos")
    
    return null_df


def analyze_quality(df, stats):
    """Analiza la calidad de los datos."""
    print_header("2️⃣ CALIDAD DE DATOS")
    
    print("📊 ESTADÍSTICAS DEL PROCESAMIENTO:")
    print(f"  Total procesados: {stats['total_processed']:,}")
    print(f"  Mantenidos: {stats['kept']:,}")
    print(f"  Duplicados eliminados: {stats['duplicates']:,}")
    print(f"  Tasa de retención: {stats['kept']/stats['total_processed']*100:.1f}%")
    
    # Contenido vacío
    print("\n🔍 CONTENIDO VACÍO:")
    no_texto = df['texto_original'].fillna('').str.strip().eq('').sum()
    no_resumen = df['resumen'].fillna('').str.strip().eq('').sum()
    no_content = ((df['texto_original'].fillna('').str.strip() == '') & 
                  (df['resumen'].fillna('').str.strip() == '')).sum()
    with_both = ((df['texto_original'].fillna('').str.strip() != '') & 
                 (df['resumen'].fillna('').str.strip() != '')).sum()
    
    print(f"  Sin texto_original: {no_texto:,} ({no_texto/len(df)*100:.1f}%)")
    print(f"  Sin resumen: {no_resumen:,} ({no_resumen/len(df)*100:.1f}%)")
    print(f"  Sin ninguno: {no_content:,} ({no_content/len(df)*100:.1f}%)")
    print(f"  ✅ Con ambos: {with_both:,} ({with_both/len(df)*100:.1f}%)")
    
    return {'with_both': with_both}


def analyze_sources(df):
    """Analiza distribución por fuentes."""
    print_header("3️⃣ DISTRIBUCIÓN POR FUENTES")
    
    # Por source_dataset
    print("📊 POR FUENTE (source_dataset):")
    source_counts = df['source_dataset'].value_counts()
    source_pct = (df['source_dataset'].value_counts(normalize=True) * 100).round(1)
    
    for source in source_counts.index:
        print(f"  {source:15s}: {source_counts[source]:6,} ({source_pct[source]:5.1f}%)")
    
    # Por label
    print("\n📊 POR LABEL:")
    label_counts = df['label'].fillna('unlabeled').value_counts()
    label_pct = (df['label'].fillna('unlabeled').value_counts(normalize=True) * 100).round(1)
    
    for label in label_counts.index:
        print(f"  {str(label):15s}: {label_counts[label]:6,} ({label_pct[label]:5.1f}%)")
    
    # Tabla cruzada
    print("\n📊 TABLA CRUZADA: FUENTE x LABEL")
    cross_tab = pd.crosstab(df['source_dataset'], df['label'].fillna('unlabeled'))
    print(cross_tab.to_string())
    
    # Visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Barras por fuente
    source_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_title('Registros por Fuente', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fuente', fontsize=12)
    ax1.set_ylabel('Cantidad', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(source_counts.values):
        ax1.text(i, v + 500, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    # Pie por label
    colors = ['coral', 'lightblue', 'lightgreen']
    ax2.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title('Proporción por Label', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribution_by_source.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Gráfico guardado: {OUTPUT_DIR / 'distribution_by_source.png'}")
    plt.close()
    
    return source_counts, label_counts


def analyze_splits(df):
    """Analiza distribución de splits."""
    print_header("4️⃣ SPLITS (Train/Test)")
    
    split_counts = df['split'].fillna('unsplit').value_counts()
    split_pct = (df['split'].fillna('unsplit').value_counts(normalize=True) * 100).round(1)
    
    print("📊 DISTRIBUCIÓN:")
    for split in split_counts.index:
        print(f"  {split:10s}: {split_counts[split]:6,} ({split_pct[split]:5.1f}%)")
    
    # Tabla cruzada: Fuente x Split
    print("\n📊 TABLA CRUZADA: FUENTE x SPLIT")
    cross_tab_split = pd.crosstab(df['source_dataset'], df['split'].fillna('unsplit'))
    print(cross_tab_split.to_string())
    
    return split_counts


def analyze_length(df):
    """Analiza longitud de textos."""
    print_header("5️⃣ ANÁLISIS DE LONGITUD (palabras)")
    
    # Calcular total
    df['word_count_total'] = df['word_count_src'].fillna(0) + df['word_count_pls'].fillna(0)
    
    print("📏 ESTADÍSTICAS:")
    print("\nTexto Original (word_count_src):")
    print(df['word_count_src'].describe().to_string())
    
    print("\nResumen (word_count_pls):")
    print(df['word_count_pls'].describe().to_string())
    
    print("\nTotal:")
    print(df['word_count_total'].describe().to_string())
    
    # Por fuente
    print("\n📊 PROMEDIO POR FUENTE:")
    length_by_source = df.groupby('source_dataset').agg({
        'word_count_src': ['mean', 'median'],
        'word_count_pls': ['mean', 'median']
    }).round(0)
    print(length_by_source.to_string())
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histograma word_count_src
    df[df['word_count_src'] > 0]['word_count_src'].hist(bins=50, ax=axes[0,0],
                                                          color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribución: word_count_src', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Palabras')
    axes[0,0].set_ylabel('Frecuencia')
    axes[0,0].axvline(df['word_count_src'].median(), color='red', linestyle='--',
                      label=f'Mediana: {df["word_count_src"].median():.0f}')
    axes[0,0].legend()
    
    # Histograma word_count_pls
    df[df['word_count_pls'] > 0]['word_count_pls'].hist(bins=50, ax=axes[0,1],
                                                          color='lightcoral', edgecolor='black')
    axes[0,1].set_title('Distribución: word_count_pls', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Palabras')
    axes[0,1].set_ylabel('Frecuencia')
    axes[0,1].axvline(df['word_count_pls'].median(), color='red', linestyle='--',
                      label=f'Mediana: {df["word_count_pls"].median():.0f}')
    axes[0,1].legend()
    
    # Boxplot por fuente usando seaborn
    sns.boxplot(data=df[df['word_count_src'] > 0], x='source_dataset', y='word_count_src', ax=axes[1,0])
    axes[1,0].set_title('word_count_src por Fuente', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Fuente')
    axes[1,0].set_ylabel('Palabras')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Boxplot por label usando seaborn
    df_temp = df[df['word_count_src'] > 0].copy()
    df_temp['label_filled'] = df_temp['label'].fillna('unlabeled')
    sns.boxplot(data=df_temp, x='label_filled', y='word_count_src', ax=axes[1,1])
    axes[1,1].set_title('word_count_src por Label', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Label')
    axes[1,1].set_ylabel('Palabras')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'length_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Gráfico guardado: {OUTPUT_DIR / 'length_analysis.png'}")
    plt.close()
    
    return df


def analyze_readability(df):
    """Analiza legibilidad (Flesch scores)."""
    print_header("6️⃣ ANÁLISIS DE LEGIBILIDAD (Flesch)")
    
    print("📖 ESTADÍSTICAS DE FLESCH READING EASE:")
    print(df['flesch_score'].describe().to_string())
    
    print("\n📚 INTERPRETACIÓN:")
    print("  90-100: Muy fácil (5to grado)")
    print("  60-70:  Fácil (8vo grado)")
    print("  30-50:  Difícil (universidad)")
    print("  0-30:   Muy difícil (profesional)")
    
    # Categorizar
    def categorize_flesch(score):
        if pd.isna(score):
            return 'Unknown'
        elif score >= 90:
            return 'Muy fácil (90-100)'
        elif score >= 60:
            return 'Fácil (60-89)'
        elif score >= 30:
            return 'Difícil (30-59)'
        else:
            return 'Muy difícil (0-29)'
    
    df['flesch_category'] = df['flesch_score'].apply(categorize_flesch)
    
    print("\n📊 DISTRIBUCIÓN POR CATEGORÍA:")
    print(df['flesch_category'].value_counts().to_string())
    
    # Por fuente
    print("\n📊 FLESCH PROMEDIO POR FUENTE:")
    flesch_by_source = df.groupby('source_dataset')['flesch_score'].agg(['mean', 'median', 'std']).round(1)
    flesch_by_source = flesch_by_source.sort_values('mean', ascending=False)
    print(flesch_by_source.to_string())
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histograma
    df['flesch_score'].hist(bins=50, ax=axes[0,0], color='lightgreen', edgecolor='black')
    axes[0,0].set_title('Distribución de Flesch Reading Ease', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Flesch Score')
    axes[0,0].set_ylabel('Frecuencia')
    axes[0,0].axvline(df['flesch_score'].median(), color='red', linestyle='--',
                      label=f'Mediana: {df["flesch_score"].median():.1f}')
    axes[0,0].legend()
    
    # Barras por categoría
    df['flesch_category'].value_counts().plot(kind='bar', ax=axes[0,1],
                                               color='orange', edgecolor='black')
    axes[0,1].set_title('Registros por Categoría', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Categoría')
    axes[0,1].set_ylabel('Cantidad')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Boxplot por fuente usando seaborn
    sns.boxplot(data=df, x='source_dataset', y='flesch_score', ax=axes[1,0])
    axes[1,0].set_title('Flesch Score por Fuente', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Fuente')
    axes[1,0].set_ylabel('Flesch Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Barras por fuente
    flesch_by_source['mean'].plot(kind='bar', ax=axes[1,1], color='skyblue',
                                    edgecolor='black', yerr=flesch_by_source['std'])
    axes[1,1].set_title('Flesch Promedio por Fuente', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Fuente')
    axes[1,1].set_ylabel('Flesch Score')
    axes[1,1].axhline(y=60, color='green', linestyle='--', label='Fácil (60)')
    axes[1,1].axhline(y=30, color='orange', linestyle='--', label='Difícil (30)')
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'flesch_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Gráfico guardado: {OUTPUT_DIR / 'flesch_analysis.png'}")
    plt.close()
    
    return df, flesch_by_source


def analyze_quality_issues(df):
    """Analiza problemas de calidad detectados."""
    print_header("7️⃣ PROBLEMAS DE CALIDAD")
    
    # Registros con problemas
    with_issues = df['quality_issues'].fillna('').str.strip().ne('').sum()
    without_issues = len(df) - with_issues
    
    print(f"Registros CON problemas: {with_issues:,} ({with_issues/len(df)*100:.1f}%)")
    print(f"Registros SIN problemas: {without_issues:,} ({without_issues/len(df)*100:.1f}%)")
    
    if with_issues > 0:
        # Extraer tipos de problemas
        issues_list = []
        for issues_str in df['quality_issues'].fillna(''):
            if issues_str:
                for issue in issues_str.split('|'):
                    if issue.strip():
                        issue_type = issue.split(':')[0].strip()
                        issues_list.append(issue_type)
        
        issue_counts = Counter(issues_list)
        
        print("\n📊 TIPOS DE PROBLEMAS MÁS COMUNES:")
        for issue_type, count in issue_counts.most_common(10):
            print(f"  - {issue_type}: {count:,}")
    else:
        print("\n✅ No se detectaron problemas de calidad")
    
    return with_issues


def generate_summary(df, stats, quality_info):
    """Genera resumen ejecutivo."""
    print_header("📊 RESUMEN EJECUTIVO")
    
    print("1️⃣ DATOS GENERALES")
    print(f"  Total de registros: {len(df):,}")
    print(f"  Duplicados eliminados: {stats['duplicates']:,}")
    print(f"  Tasa de retención: {stats['kept']/stats['total_processed']*100:.1f}%")
    
    print("\n2️⃣ DISTRIBUCIÓN POR FUENTE")
    for source, count in df['source_dataset'].value_counts().items():
        pct = count/len(df)*100
        print(f"  {source:15s}: {count:6,} ({pct:5.1f}%)")
    
    print("\n3️⃣ DISTRIBUCIÓN POR LABEL")
    for label, count in df['label'].fillna('unlabeled').value_counts().items():
        pct = count/len(df)*100
        print(f"  {str(label):15s}: {count:6,} ({pct:5.1f}%)")
    
    print("\n4️⃣ SPLITS")
    for split, count in df['split'].fillna('unsplit').value_counts().items():
        pct = count/len(df)*100
        print(f"  {split:10s}: {count:6,} ({pct:5.1f}%)")
    
    print("\n5️⃣ MÉTRICAS DE CALIDAD")
    print(f"  Longitud promedio: {df['word_count_total'].mean():.0f} palabras")
    print(f"  Flesch score promedio: {df['flesch_score'].mean():.1f} (Muy difícil)")
    print(f"  Con texto Y resumen: {quality_info['with_both']:,} ({quality_info['with_both']/len(df)*100:.1f}%)")
    
    print("\n6️⃣ RECOMENDACIONES")
    print("  ✅ Dataset de alta calidad con 67K+ registros")
    print("  ⚠️ Flesch bajo (35): textos muy técnicos")
    print("  🎯 Acción: Emparejar Cochrane por DOI")
    print("  🎯 Acción: Balancear dataset (Cochrane domina)")
    print("  🎯 Acción: Crear validation set (15% de train)")
    
    # Guardar resumen (convertir todos a tipos nativos de Python)
    def convert_to_python_types(obj):
        """Convierte tipos numpy a tipos nativos de Python."""
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(i) for i in obj]
        else:
            return obj
    
    summary = {
        'total_records': int(len(df)),
        'duplicates_removed': int(stats['duplicates']),
        'by_source': convert_to_python_types(df['source_dataset'].value_counts().to_dict()),
        'by_label': convert_to_python_types(df['label'].fillna('unlabeled').value_counts().to_dict()),
        'by_split': convert_to_python_types(df['split'].fillna('unsplit').value_counts().to_dict()),
        'avg_flesch': float(df['flesch_score'].mean()),
        'avg_word_count': float(df['word_count_total'].mean()),
        'with_pairs': int(stats['with_pairs']),
        'with_both': int(quality_info['with_both']),
    }
    
    with open(OUTPUT_DIR / 'summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Resumen guardado: {OUTPUT_DIR / 'summary_stats.json'}")


def main():
    """Ejecuta el EDA completo."""
    # Cargar datos
    df, stats = load_data()
    
    # Análisis
    analyze_structure(df)
    quality_info = analyze_quality(df, stats)
    analyze_sources(df)
    analyze_splits(df)
    df = analyze_length(df)
    df, flesch_by_source = analyze_readability(df)
    with_issues = analyze_quality_issues(df)
    
    # Añadir info de quality
    quality_info['with_issues'] = with_issues
    
    # Resumen
    generate_summary(df, stats, quality_info)
    
    print_header("🎉 EDA COMPLETO FINALIZADO")
    print(f"✅ Reportes guardados en: {OUTPUT_DIR}")
    print(f"✅ Gráficos generados")
    print(f"\n💡 Revisa los archivos en {OUTPUT_DIR}:")
    for file in OUTPUT_DIR.glob('*'):
        print(f"   - {file.name}")


if __name__ == "__main__":
    main()

