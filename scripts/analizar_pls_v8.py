"""
Script para análisis detallado de PLS V8 generados.
Evalúa calidad, detecta problemas y genera recomendaciones.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configuración
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def analizar_pls_detallado(csv_path):
    """Análisis exhaustivo de PLS generados"""
    
    print("\n" + "="*80)
    print("ANÁLISIS DETALLADO DE PLS V8.1")
    print("="*80)
    
    # Cargar datos
    df = pd.read_csv(csv_path)
    print(f"\n📊 Dataset: {len(df)} PLS generados")
    
    # =============================================================================
    # 1. ANÁLISIS DE VALIDEZ
    # =============================================================================
    print("\n" + "="*80)
    print("1. ANÁLISIS DE VALIDEZ (140-210 palabras)")
    print("="*80)
    
    validos = df[df['es_valido'] == True]
    invalidos = df[df['es_valido'] == False]
    
    print(f"\n✅ PLS VÁLIDOS: {len(validos)} ({len(validos)/len(df)*100:.1f}%)")
    print(f"❌ PLS INVÁLIDOS: {len(invalidos)} ({len(invalidos)/len(df)*100:.1f}%)")
    
    if len(invalidos) > 0:
        print(f"\n🔍 ANÁLISIS DE PLS INVÁLIDOS:")
        print(f"   Longitud promedio: {invalidos['longitud_pls'].mean():.1f} palabras")
        print(f"   Rango: [{invalidos['longitud_pls'].min()}, {invalidos['longitud_pls'].max()}]")
        
        # Contar cuántos están por debajo vs por encima
        muy_cortos = invalidos[invalidos['longitud_pls'] < 140]
        muy_largos = invalidos[invalidos['longitud_pls'] > 210]
        
        print(f"\n   📉 Muy cortos (<140 palabras): {len(muy_cortos)} ({len(muy_cortos)/len(invalidos)*100:.1f}%)")
        if len(muy_cortos) > 0:
            print(f"      Rango: {muy_cortos['longitud_pls'].min()}-{muy_cortos['longitud_pls'].max()} palabras")
            print(f"      Promedio: {muy_cortos['longitud_pls'].mean():.1f} palabras")
        
        print(f"   📈 Muy largos (>210 palabras): {len(muy_largos)} ({len(muy_largos)/len(invalidos)*100:.1f}%)")
        if len(muy_largos) > 0:
            print(f"      Rango: {muy_largos['longitud_pls'].min()}-{muy_largos['longitud_pls'].max()} palabras")
            print(f"      Promedio: {muy_largos['longitud_pls'].mean():.1f} palabras")
    
    # =============================================================================
    # 2. ANÁLISIS DE CALIDAD
    # =============================================================================
    print("\n" + "="*80)
    print("2. DISTRIBUCIÓN DE CALIDAD")
    print("="*80)
    
    calidad_counts = df['calidad'].value_counts()
    print("\n📊 Distribución:")
    for calidad, count in calidad_counts.items():
        porcentaje = count/len(df)*100
        barra = "█" * int(porcentaje/2)
        print(f"   {calidad:12s}: {count:3d} ({porcentaje:5.1f}%) {barra}")
    
    # =============================================================================
    # 3. MÉTRICAS DE LEGIBILIDAD
    # =============================================================================
    print("\n" + "="*80)
    print("3. MÉTRICAS DE LEGIBILIDAD")
    print("="*80)
    
    print(f"\n📖 FLESCH READING EASE (target: 60-70)")
    print(f"   Media:   {df['flesch_reading_ease'].mean():.1f}")
    print(f"   Mediana: {df['flesch_reading_ease'].median():.1f}")
    print(f"   Std:     {df['flesch_reading_ease'].std():.1f}")
    print(f"   Rango:   [{df['flesch_reading_ease'].min():.1f}, {df['flesch_reading_ease'].max():.1f}]")
    
    # Categorizar FRE
    fre_muy_bajo = len(df[df['flesch_reading_ease'] < 50])
    fre_bajo = len(df[(df['flesch_reading_ease'] >= 50) & (df['flesch_reading_ease'] < 60)])
    fre_bueno = len(df[(df['flesch_reading_ease'] >= 60) & (df['flesch_reading_ease'] <= 70)])
    fre_alto = len(df[df['flesch_reading_ease'] > 70])
    
    print(f"\n   Distribución:")
    print(f"      <50 (muy difícil):  {fre_muy_bajo} ({fre_muy_bajo/len(df)*100:.1f}%)")
    print(f"      50-60 (difícil):    {fre_bajo} ({fre_bajo/len(df)*100:.1f}%)")
    print(f"      60-70 (OBJETIVO):   {fre_bueno} ({fre_bueno/len(df)*100:.1f}%) ✅")
    print(f"      >70 (muy fácil):    {fre_alto} ({fre_alto/len(df)*100:.1f}%)")
    
    print(f"\n📚 FLESCH-KINCAID GRADE (target: 7-9)")
    print(f"   Media:   {df['flesch_kincaid_grade'].mean():.1f}")
    print(f"   Mediana: {df['flesch_kincaid_grade'].median():.1f}")
    print(f"   Std:     {df['flesch_kincaid_grade'].std():.1f}")
    print(f"   Rango:   [{df['flesch_kincaid_grade'].min():.1f}, {df['flesch_kincaid_grade'].max():.1f}]")
    
    fkg_bajo = len(df[df['flesch_kincaid_grade'] < 7])
    fkg_bueno = len(df[(df['flesch_kincaid_grade'] >= 7) & (df['flesch_kincaid_grade'] <= 9)])
    fkg_alto = len(df[df['flesch_kincaid_grade'] > 9])
    
    print(f"\n   Distribución:")
    print(f"      <7 (muy simple):    {fkg_bajo} ({fkg_bajo/len(df)*100:.1f}%)")
    print(f"      7-9 (OBJETIVO):     {fkg_bueno} ({fkg_bueno/len(df)*100:.1f}%) ✅")
    print(f"      >9 (muy complejo):  {fkg_alto} ({fkg_alto/len(df)*100:.1f}%)")
    
    # =============================================================================
    # 4. ANÁLISIS DE REPETICIONES (NUEVAS MÉTRICAS V8)
    # =============================================================================
    print("\n" + "="*80)
    print("4. ANÁLISIS DE REPETICIONES (Anti-Repetición V8)")
    print("="*80)
    
    print(f"\n🔤 TYPE-TOKEN RATIO (Diversidad léxica)")
    print(f"   Media:   {df['type_token_ratio'].mean():.3f}")
    print(f"   Mediana: {df['type_token_ratio'].median():.3f}")
    print(f"   Std:     {df['type_token_ratio'].std():.3f}")
    print(f"   Rango:   [{df['type_token_ratio'].min():.3f}, {df['type_token_ratio'].max():.3f}]")
    print(f"\n   Interpretación: >0.5 es bueno, >0.6 es excelente")
    
    ttr_excelente = len(df[df['type_token_ratio'] > 0.6])
    ttr_bueno = len(df[(df['type_token_ratio'] >= 0.5) & (df['type_token_ratio'] <= 0.6)])
    ttr_bajo = len(df[df['type_token_ratio'] < 0.5])
    
    print(f"   Distribución:")
    print(f"      >0.6 (excelente): {ttr_excelente} ({ttr_excelente/len(df)*100:.1f}%) ✅")
    print(f"      0.5-0.6 (bueno):  {ttr_bueno} ({ttr_bueno/len(df)*100:.1f}%)")
    print(f"      <0.5 (bajo):      {ttr_bajo} ({ttr_bajo/len(df)*100:.1f}%)")
    
    print(f"\n🔁 REPETICIÓN DE BIGRAMAS")
    print(f"   Media:   {df['repetition_rate_bigrams'].mean():.3f}")
    print(f"   Mediana: {df['repetition_rate_bigrams'].median():.3f}")
    print(f"   Std:     {df['repetition_rate_bigrams'].std():.3f}")
    print(f"   Rango:   [{df['repetition_rate_bigrams'].min():.3f}, {df['repetition_rate_bigrams'].max():.3f}]")
    print(f"\n   Interpretación: <0.2 es bueno, <0.1 es excelente")
    
    bigram_excelente = len(df[df['repetition_rate_bigrams'] < 0.1])
    bigram_bueno = len(df[(df['repetition_rate_bigrams'] >= 0.1) & (df['repetition_rate_bigrams'] < 0.2)])
    bigram_alto = len(df[df['repetition_rate_bigrams'] >= 0.2])
    
    print(f"   Distribución:")
    print(f"      <0.1 (excelente): {bigram_excelente} ({bigram_excelente/len(df)*100:.1f}%) ✅")
    print(f"      0.1-0.2 (bueno):  {bigram_bueno} ({bigram_bueno/len(df)*100:.1f}%)")
    print(f"      >=0.2 (alto):     {bigram_alto} ({bigram_alto/len(df)*100:.1f}%)")
    
    print(f"\n🔁 REPETICIÓN DE TRIGRAMAS")
    print(f"   Media:   {df['repetition_rate_trigrams'].mean():.3f}")
    print(f"   Mediana: {df['repetition_rate_trigrams'].median():.3f}")
    print(f"   Std:     {df['repetition_rate_trigrams'].std():.3f}")
    print(f"   Rango:   [{df['repetition_rate_trigrams'].min():.3f}, {df['repetition_rate_trigrams'].max():.3f}]")
    print(f"\n   Interpretación: <0.1 es bueno, <0.05 es excelente")
    
    trigram_excelente = len(df[df['repetition_rate_trigrams'] < 0.05])
    trigram_bueno = len(df[(df['repetition_rate_trigrams'] >= 0.05) & (df['repetition_rate_trigrams'] < 0.1)])
    trigram_alto = len(df[df['repetition_rate_trigrams'] >= 0.1])
    
    print(f"   Distribución:")
    print(f"      <0.05 (excelente): {trigram_excelente} ({trigram_excelente/len(df)*100:.1f}%) ✅")
    print(f"      0.05-0.1 (bueno):  {trigram_bueno} ({trigram_bueno/len(df)*100:.1f}%)")
    print(f"      >=0.1 (alto):      {trigram_alto} ({trigram_alto/len(df)*100:.1f}%)")
    
    # =============================================================================
    # 5. EJEMPLOS ESPECÍFICOS
    # =============================================================================
    print("\n" + "="*80)
    print("5. EJEMPLOS REPRESENTATIVOS")
    print("="*80)
    
    # Mejor PLS (excelente)
    if len(validos[validos['calidad'] == 'excelente']) > 0:
        mejor = validos[validos['calidad'] == 'excelente'].iloc[0]
        print(f"\n✅ EJEMPLO DE PLS EXCELENTE (ID: {mejor['id']})")
        print(f"   Longitud: {mejor['longitud_pls']} palabras")
        print(f"   FRE: {mejor['flesch_reading_ease']:.1f} | FKG: {mejor['flesch_kincaid_grade']:.1f}")
        print(f"   TTR: {mejor['type_token_ratio']:.3f}")
        print(f"\n   TEXTO:")
        print("-" * 80)
        print(mejor['pls_generado'][:500] + "..." if len(mejor['pls_generado']) > 500 else mejor['pls_generado'])
        print("-" * 80)
    
    # Peor PLS (pobre)
    if len(invalidos) > 0:
        peor = invalidos.iloc[0]
        print(f"\n❌ EJEMPLO DE PLS POBRE (ID: {peor['id']})")
        print(f"   Longitud: {peor['longitud_pls']} palabras")
        print(f"   FRE: {peor['flesch_reading_ease']:.1f} | FKG: {peor['flesch_kincaid_grade']:.1f}")
        print(f"   TTR: {peor['type_token_ratio']:.3f}")
        if peor['problemas_detectados']:
            print(f"   ⚠️ PROBLEMA: {peor['problemas_detectados']}")
        print(f"\n   TEXTO:")
        print("-" * 80)
        print(peor['pls_generado'][:500] + "..." if len(peor['pls_generado']) > 500 else peor['pls_generado'])
        print("-" * 80)
    
    # =============================================================================
    # 6. ANÁLISIS DE PROBLEMAS COMUNES
    # =============================================================================
    print("\n" + "="*80)
    print("6. PROBLEMAS DETECTADOS")
    print("="*80)
    
    problemas = df[df['problemas_detectados'] != '']['problemas_detectados'].value_counts()
    if len(problemas) > 0:
        print(f"\n⚠️ TIPOS DE PROBLEMAS ENCONTRADOS:")
        for problema, count in problemas.items():
            print(f"   • {problema}: {count} casos")
    else:
        print("\n✅ No se detectaron problemas críticos")
    
    warnings = df[df['warnings'] != '']['warnings'].value_counts()
    if len(warnings) > 0:
        print(f"\n⚠️ ADVERTENCIAS ENCONTRADAS:")
        for warning, count in warnings.head(5).items():
            print(f"   • {warning}: {count} casos")
    
    # =============================================================================
    # 7. COSTOS
    # =============================================================================
    print("\n" + "="*80)
    print("7. ANÁLISIS DE COSTOS")
    print("="*80)
    
    costo_total = df['costo_estimado'].sum()
    costo_promedio = df['costo_estimado'].mean()
    
    print(f"\n💰 COSTOS:")
    print(f"   Costo total: ${costo_total:.4f}")
    print(f"   Costo promedio por PLS: ${costo_promedio:.6f}")
    print(f"\n   Proyección a 10,000 PLS: ${costo_promedio * 10000:.2f}")
    print(f"   Proyección a 20,000 PLS: ${costo_promedio * 20000:.2f}")
    
    # =============================================================================
    # 8. RECOMENDACIONES
    # =============================================================================
    print("\n" + "="*80)
    print("8. RECOMENDACIONES")
    print("="*80)
    
    recomendaciones = []
    
    # Problema de longitud
    if len(invalidos) / len(df) > 0.2:  # Más del 20% inválidos
        if len(muy_cortos) > len(muy_largos):
            recomendaciones.append({
                'prioridad': '🔴 ALTA',
                'problema': f'{len(muy_cortos)} PLS muy cortos (<140 palabras)',
                'solucion': 'Aumentar MIN_TOKENS en el prompt o ajustar MAX_TOKENS a 350',
                'accion': 'Modificar línea de MAX_TOKENS en el script'
            })
        else:
            recomendaciones.append({
                'prioridad': '🟡 MEDIA',
                'problema': f'{len(muy_largos)} PLS muy largos (>210 palabras)',
                'solucion': 'Reducir MAX_TOKENS o ser más estricto en el prompt',
                'accion': 'Ajustar MAX_TOKENS o agregar límite explícito al prompt'
            })
    
    # FRE bajo
    if df['flesch_reading_ease'].mean() < 60:
        recomendaciones.append({
            'prioridad': '🟡 MEDIA',
            'problema': f'FRE promedio ({df["flesch_reading_ease"].mean():.1f}) por debajo del target (60-70)',
            'solucion': 'Enfatizar más la simplicidad en el prompt',
            'accion': 'Agregar ejemplos con oraciones más cortas'
        })
    
    # FKG alto
    if df['flesch_kincaid_grade'].mean() > 9:
        recomendaciones.append({
            'prioridad': '🟡 MEDIA',
            'problema': f'FKG promedio ({df["flesch_kincaid_grade"].mean():.1f}) por encima del target (7-9)',
            'solucion': 'Simplificar más el vocabulario',
            'accion': 'Reducir temperatura a 0.4 o reforzar "use short words"'
        })
    
    # TTR o repeticiones
    if df['type_token_ratio'].mean() < 0.5:
        recomendaciones.append({
            'prioridad': '🟡 MEDIA',
            'problema': f'TTR bajo ({df["type_token_ratio"].mean():.3f}), mucha repetición de palabras',
            'solucion': 'Aumentar frequency_penalty',
            'accion': 'Cambiar FREQUENCY_PENALTY de 0.5 a 0.7'
        })
    
    if len(recomendaciones) == 0:
        print("\n✅ No se detectaron problemas significativos")
        print("   Los PLS generados cumplen con los estándares de calidad")
        print("   Puedes proceder con la generación de 10,000 PLS")
    else:
        print(f"\n📋 SE ENCONTRARON {len(recomendaciones)} ÁREAS DE MEJORA:\n")
        for i, rec in enumerate(recomendaciones, 1):
            print(f"{i}. {rec['prioridad']}")
            print(f"   Problema: {rec['problema']}")
            print(f"   Solución: {rec['solucion']}")
            print(f"   Acción:   {rec['accion']}")
            print()
    
    # =============================================================================
    # 9. RESUMEN EJECUTIVO
    # =============================================================================
    print("\n" + "="*80)
    print("9. RESUMEN EJECUTIVO")
    print("="*80)
    
    # Calcular score general
    score_validez = len(validos) / len(df) * 100
    score_fre = min(100, max(0, (df['flesch_reading_ease'].mean() - 40) / 30 * 100))
    score_ttr = min(100, df['type_token_ratio'].mean() / 0.7 * 100)
    score_repeticion = min(100, (1 - df['repetition_rate_trigrams'].mean() / 0.2) * 100)
    
    score_general = (score_validez + score_fre + score_ttr + score_repeticion) / 4
    
    print(f"\n📊 SCORE GENERAL: {score_general:.1f}/100")
    print(f"\n   Desglose:")
    print(f"      Validez (140-210 palabras):  {score_validez:.1f}/100")
    print(f"      Legibilidad (FRE):            {score_fre:.1f}/100")
    print(f"      Diversidad léxica (TTR):      {score_ttr:.1f}/100")
    print(f"      Anti-repetición (trigramas):  {score_repeticion:.1f}/100")
    
    if score_general >= 80:
        print(f"\n   ✅ CALIDAD: EXCELENTE")
        print(f"   Recomendación: Proceder con generación de 10k PLS")
    elif score_general >= 60:
        print(f"\n   ⚠️  CALIDAD: BUENA")
        print(f"   Recomendación: Considerar ajustes menores antes de 10k")
    else:
        print(f"\n   ❌ CALIDAD: MEJORABLE")
        print(f"   Recomendación: Ajustar parámetros antes de 10k")
    
    print("\n" + "="*80)
    return df, recomendaciones

if __name__ == "__main__":
    # Buscar el archivo más reciente (v8.1 primero, luego v8 como fallback)
    csv_path_v81 = Path("data/synthetic_pls/pls_prueba_50_v8.1.csv")
    csv_path_v8 = Path("data/synthetic_pls/pls_prueba_50_v8.csv")
    
    if csv_path_v81.exists():
        csv_path = csv_path_v81
        print(f"📁 Analizando: {csv_path_v81.name}")
    elif csv_path_v8.exists():
        csv_path = csv_path_v8
        print(f"📁 Analizando: {csv_path_v8.name} (versión anterior)")
    else:
        print(f"❌ No se encuentra ningún archivo de prueba")
        print("   Buscados:")
        print(f"   - {csv_path_v81}")
        print(f"   - {csv_path_v8}")
        print("\n   Por favor, ejecuta primero: python scripts/generar_pls_sinteticos_async.py 1")
        exit(1)
    
    df, recomendaciones = analizar_pls_detallado(csv_path)
    
    # Guardar recomendaciones
    if recomendaciones:
        with open("logs/recomendaciones_v8.json", "w") as f:
            json.dump(recomendaciones, f, indent=2)
        print(f"\n💾 Recomendaciones guardadas en: logs/recomendaciones_v8.json")

