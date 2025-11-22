"""
Script para comparar todas las versiones de prompts (V1, V2, V3).

Compara métricas de legibilidad entre las 3 versiones.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def cargar_version(version):
    """Carga un archivo de versión específica"""
    archivo = Path(f'../data/synthetic_pls/pls_prueba_50_{version}.csv')
    if not archivo.exists():
        print(f"ADVERTENCIA: No se encontró: {archivo}")
        return None
    return pd.read_csv(archivo)

def main():
    print("="*70)
    print("COMPARACIÓN DE TODAS LAS VERSIONES DE PROMPTS")
    print("="*70)
    
    # Cargar todas las versiones
    df_v1 = cargar_version('v1')
    df_v2 = cargar_version('v2')
    df_v3 = cargar_version('v3')
    
    versiones = []
    if df_v1 is not None:
        versiones.append(('V1 (Español)', df_v1))
    if df_v2 is not None:
        versiones.append(('V2 (Español Simple)', df_v2))
    if df_v3 is not None:
        versiones.append(('V3 (Inglés)', df_v3))
    
    if len(versiones) == 0:
        print("\nERROR: No se encontraron archivos de versiones")
        print("\nAsegúrate de tener:")
        print("  - data/synthetic_pls/pls_prueba_50_v1.csv")
        print("  - data/synthetic_pls/pls_prueba_50_v2.csv")
        print("  - data/synthetic_pls/pls_prueba_50_v3.csv")
        return
    
    # Tabla comparativa
    print("\n" + "="*70)
    print("TABLA COMPARATIVA")
    print("="*70)
    print(f"\n{'Métrica':<30} ", end='')
    for nombre, _ in versiones:
        print(f"{nombre:<20}", end='')
    print("\n" + "-"*70)
    
    # Flesch Reading Ease
    print(f"{'Flesch Reading Ease':<30} ", end='')
    for _, df in versiones:
        valor = df['flesch_reading_ease'].mean()
        simbolo = "[OK]" if 55 <= valor <= 75 else "[X]"
        print(f"{valor:>6.1f} {simbolo:<12}", end='')
    print(f"\n{'  Target: 60-70':<30}")
    
    # Flesch-Kincaid Grade
    print(f"\n{'Flesch-Kincaid Grade':<30} ", end='')
    for _, df in versiones:
        valor = df['flesch_kincaid_grade'].mean()
        simbolo = "[OK]" if valor <= 10 else "[X]"
        print(f"{valor:>6.1f} {simbolo:<12}", end='')
    print(f"\n{'  Target: 7-9':<30}")
    
    # Longitud PLS
    print(f"\n{'Longitud PLS (palabras)':<30} ", end='')
    for _, df in versiones:
        valor = df['longitud_pls'].mean()
        print(f"{valor:>6.0f}        ", end='')
    print(f"\n{'  Target: 150-250':<30}")
    
    # % en rango
    print(f"\n{'% en rango 150-250':<30} ", end='')
    for _, df in versiones:
        en_rango = len(df[(df['longitud_pls'] >= 150) & (df['longitud_pls'] <= 250)])
        porcentaje = (en_rango / len(df)) * 100
        simbolo = "[OK]" if porcentaje >= 80 else "[X]"
        print(f"{porcentaje:>6.0f}% {simbolo:<11}", end='')
    print(f"\n{'  Target: >80%':<30}")
    
    # Ratio compresión
    print(f"\n{'Ratio compresión':<30} ", end='')
    for _, df in versiones:
        valor = df['ratio_compresion'].mean()
        print(f"{valor:>6.2f}        ", end='')
    print()
    
    # Costo
    print(f"\n{'Costo promedio (USD)':<30} ", end='')
    for _, df in versiones:
        valor = df['costo_estimado'].mean()
        print(f"${valor:>7.6f}     ", end='')
    print()
    
    print("\n" + "="*70)
    print("RESUMEN DE MEJORA")
    print("="*70)
    
    if len(versiones) >= 2:
        # Comparar primera vs última versión
        nombre_inicial, df_inicial = versiones[0]
        nombre_final, df_final = versiones[-1]
        
        flesch_inicial = df_inicial['flesch_reading_ease'].mean()
        flesch_final = df_final['flesch_reading_ease'].mean()
        mejora_flesch = flesch_final - flesch_inicial
        
        grade_inicial = df_inicial['flesch_kincaid_grade'].mean()
        grade_final = df_final['flesch_kincaid_grade'].mean()
        mejora_grade = grade_inicial - grade_final  # Invertido: menor es mejor
        
        print(f"\nDesde {nombre_inicial} hasta {nombre_final}:")
        print(f"\nFlesch Reading Ease:")
        if mejora_flesch > 0:
            print(f"  [OK] Mejoró +{mejora_flesch:.1f} puntos ({flesch_inicial:.1f} → {flesch_final:.1f})")
        else:
            print(f"  [X] Empeoró {mejora_flesch:.1f} puntos ({flesch_inicial:.1f} → {flesch_final:.1f})")
        
        print(f"\nFlesch-Kincaid Grade:")
        if mejora_grade > 0:
            print(f"  [OK] Mejoró -{mejora_grade:.1f} puntos ({grade_inicial:.1f} → {grade_final:.1f})")
        else:
            print(f"  [X] Empeoró +{abs(mejora_grade):.1f} puntos ({grade_inicial:.1f} → {grade_final:.1f})")
    
    # Determinar mejor versión
    print("\n" + "="*70)
    print("MEJOR VERSIÓN SEGÚN TARGETS")
    print("="*70)
    
    mejor_score = -1
    mejor_version = None
    
    for nombre, df in versiones:
        score = 0
        flesch = df['flesch_reading_ease'].mean()
        grade = df['flesch_kincaid_grade'].mean()
        en_rango = len(df[(df['longitud_pls'] >= 150) & (df['longitud_pls'] <= 250)])
        porcentaje = (en_rango / len(df)) * 100
        
        # Scoring simple
        if 60 <= flesch <= 70:
            score += 3
        elif 55 <= flesch <= 75:
            score += 1
        
        if 7 <= grade <= 9:
            score += 3
        elif grade <= 10:
            score += 1
        
        if porcentaje >= 90:
            score += 2
        elif porcentaje >= 80:
            score += 1
        
        print(f"\n{nombre}: Score = {score}/8")
        if score > mejor_score:
            mejor_score = score
            mejor_version = nombre
    
    print(f"\nGANADOR: {mejor_version} (Score: {mejor_score}/8)")
    
    # Gráficos
    if len(versiones) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        nombres = [n for n, _ in versiones]
        
        # Flesch Reading Ease
        valores_flesch = [df['flesch_reading_ease'].mean() for _, df in versiones]
        axes[0, 0].bar(nombres, valores_flesch, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Min target')
        axes[0, 0].axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Max target')
        axes[0, 0].set_ylabel('Puntuación')
        axes[0, 0].set_title('Flesch Reading Ease (60-70 = ideal)', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Flesch-Kincaid Grade
        valores_grade = [df['flesch_kincaid_grade'].mean() for _, df in versiones]
        axes[0, 1].bar(nombres, valores_grade, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 1].axhline(y=8, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Target')
        axes[0, 1].axhline(y=7, color='green', linestyle='--', alpha=0.3)
        axes[0, 1].axhline(y=9, color='green', linestyle='--', alpha=0.3)
        axes[0, 1].set_ylabel('Nivel de grado')
        axes[0, 1].set_title('Flesch-Kincaid Grade (7-9 = ideal)', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Longitud PLS
        valores_longitud = [df['longitud_pls'].mean() for _, df in versiones]
        axes[1, 0].bar(nombres, valores_longitud, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1, 0].axhline(y=150, color='orange', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=250, color='orange', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(y=200, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Ideal')
        axes[1, 0].set_ylabel('Palabras')
        axes[1, 0].set_title('Longitud Promedio PLS (150-250 ideal)', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Distribución Flesch RE
        for nombre, df in versiones:
            axes[1, 1].hist(df['flesch_reading_ease'], bins=15, alpha=0.5, label=nombre)
        axes[1, 1].axvline(x=60, color='green', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(x=70, color='green', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Flesch Reading Ease')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Distribución Flesch RE por Versión', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparacion_versiones.png', dpi=300, bbox_inches='tight')
        print(f"\nGráfico guardado: comparacion_versiones.png")
        plt.show()
    
    # Recomendación final
    print("\n" + "="*70)
    print("RECOMENDACIÓN FINAL")
    print("="*70)
    
    if mejor_version and df_v3 is not None and 'V3' in mejor_version:
        flesch_v3 = df_v3['flesch_reading_ease'].mean()
        grade_v3 = df_v3['flesch_kincaid_grade'].mean()
        
        if flesch_v3 >= 55 and grade_v3 <= 10:
            print("\nV3 (Inglés) cumple los targets mínimos")
            print("\nPROCEDER A PRODUCCIÓN:")
            print("   1. cd scripts")
            print("   2. python generar_pls_sinteticos.py")
            print("   3. Opción 2: Generar 10,000 PLS")
        else:
            print("\nADVERTENCIA: V3 mejora pero no cumple todos los targets")
            print("\n🔧 AJUSTES RECOMENDADOS:")
            if flesch_v3 < 55:
                print(f"   - Flesch RE muy bajo ({flesch_v3:.1f}). Necesita subir.")
                print("   - Considera ajustar temperatura a 0.5")
            if grade_v3 > 10:
                print(f"   - Grade muy alto ({grade_v3:.1f}). Demasiado complejo.")
                print("   - Considera modificar el prompt para oraciones aún más cortas")
    else:
        print("\nADVERTENCIA: Ninguna versión cumple completamente los targets")
        print("\n🔄 OPCIONES:")
        print("   1. Ajustar temperatura del modelo (0.5 en vez de 0.7)")
        print("   2. Usar GPT-4 en vez de GPT-4o-mini")
        print("   3. Crear prompt V4 con ajustes adicionales")

if __name__ == "__main__":
    main()

