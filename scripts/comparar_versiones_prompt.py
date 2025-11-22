"""
Script auxiliar para comparar resultados entre diferentes versiones del prompt.

Compara métricas de legibilidad entre:
- V1: Prompt original (más complejo)
- V2: Prompt simplificado (más simple)

Uso:
    python comparar_versiones_prompt.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def cargar_datos(archivo):
    """Carga un CSV de resultados PLS"""
    if not Path(archivo).exists():
        return None
    return pd.read_csv(archivo)

def mostrar_comparacion(df_v1, df_v2):
    """Muestra comparación de métricas entre dos versiones"""
    
    print("="*70)
    print("COMPARACIÓN DE VERSIONES DEL PROMPT")
    print("="*70)
    
    if df_v1 is None:
        print("\nADVERTENCIA: No se encontró archivo V1")
        return
    
    if df_v2 is None:
        print("\nADVERTENCIA: No se encontró archivo V2")
        return
    
    print(f"\nLONGITUD DE PLS:")
    print(f"   V1 (original):     {df_v1['longitud_pls'].mean():.1f} palabras")
    print(f"   V2 (simplificado): {df_v2['longitud_pls'].mean():.1f} palabras")
    diff_longitud = df_v2['longitud_pls'].mean() - df_v1['longitud_pls'].mean()
    print(f"   Diferencia:        {diff_longitud:+.1f} palabras")
    
    print(f"\nFLESCH READING EASE (60-70 = TARGET):")
    print(f"   V1 (original):     {df_v1['flesch_reading_ease'].mean():.1f}")
    print(f"   V2 (simplificado): {df_v2['flesch_reading_ease'].mean():.1f}")
    diff_flesch = df_v2['flesch_reading_ease'].mean() - df_v1['flesch_reading_ease'].mean()
    mejora_flesch = "[OK] MEJORA" if diff_flesch > 0 else "[X] EMPEORA"
    print(f"   Diferencia:        {diff_flesch:+.1f} {mejora_flesch}")
    
    print(f"\nFLESCH-KINCAID GRADE (~8.0 = TARGET):")
    print(f"   V1 (original):     {df_v1['flesch_kincaid_grade'].mean():.1f}")
    print(f"   V2 (simplificado): {df_v2['flesch_kincaid_grade'].mean():.1f}")
    diff_grade = df_v2['flesch_kincaid_grade'].mean() - df_v1['flesch_kincaid_grade'].mean()
    mejora_grade = "[OK] MEJORA" if diff_grade < 0 else "[X] EMPEORA"
    print(f"   Diferencia:        {diff_grade:+.1f} {mejora_grade}")
    
    print(f"\nRATIO DE COMPRESIÓN:")
    print(f"   V1 (original):     {df_v1['ratio_compresion'].mean():.2f}")
    print(f"   V2 (simplificado): {df_v2['ratio_compresion'].mean():.2f}")
    
    print(f"\nCOSTO PROMEDIO:")
    print(f"   V1 (original):     ${df_v1['costo_estimado'].mean():.6f}")
    print(f"   V2 (simplificado): ${df_v2['costo_estimado'].mean():.6f}")
    
    print("\n" + "="*70)
    print("EVALUACIÓN:")
    print("="*70)
    
    # Evaluar si V2 está más cerca de los targets
    v1_flesch_dist = abs(df_v1['flesch_reading_ease'].mean() - 65)
    v2_flesch_dist = abs(df_v2['flesch_reading_ease'].mean() - 65)
    
    v1_grade_dist = abs(df_v1['flesch_kincaid_grade'].mean() - 8.0)
    v2_grade_dist = abs(df_v2['flesch_kincaid_grade'].mean() - 8.0)
    
    if v2_flesch_dist < v1_flesch_dist:
        print("[OK] V2 está más cerca del target Flesch Reading Ease (60-70)")
    else:
        print("[X] V2 está más lejos del target Flesch Reading Ease (60-70)")
    
    if v2_grade_dist < v1_grade_dist:
        print("[OK] V2 está más cerca del target Flesch-Kincaid Grade (~8.0)")
    else:
        print("[X] V2 está más lejos del target Flesch-Kincaid Grade (~8.0)")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Flesch Reading Ease
    axes[0].boxplot([df_v1['flesch_reading_ease'], df_v2['flesch_reading_ease']], 
                     labels=['V1 (Original)', 'V2 (Simplificado)'])
    axes[0].axhline(y=65, color='g', linestyle='--', label='Target (60-70)')
    axes[0].set_ylabel('Flesch Reading Ease')
    axes[0].set_title('Comparación de Legibilidad')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Flesch-Kincaid Grade
    axes[1].boxplot([df_v1['flesch_kincaid_grade'], df_v2['flesch_kincaid_grade']], 
                     labels=['V1 (Original)', 'V2 (Simplificado)'])
    axes[1].axhline(y=8.0, color='g', linestyle='--', label='Target (~8.0)')
    axes[1].set_ylabel('Flesch-Kincaid Grade')
    axes[1].set_title('Comparación de Nivel de Grado')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparacion_prompts.png', dpi=300, bbox_inches='tight')
    print("\nGráfico guardado en: comparacion_prompts.png")
    plt.show()


def main():
    """Función principal"""
    
    # Rutas de archivos
    archivo_v1 = Path('../data/synthetic_pls/pls_prueba_50.csv')
    archivo_v2 = Path('../data/synthetic_pls/pls_prueba_50_v2.csv')
    
    print("\nBuscando archivos...")
    print(f"   V1: {archivo_v1}")
    print(f"   V2: {archivo_v2}")
    
    df_v1 = cargar_datos(archivo_v1)
    df_v2 = cargar_datos(archivo_v2)
    
    if df_v1 is None:
        print(f"\nADVERTENCIA: No se encontró {archivo_v1}")
        print("   Debe existir el archivo de la primera prueba (50 PLS con prompt original)")
    
    if df_v2 is None:
        print(f"\nADVERTENCIA: No se encontró {archivo_v2}")
        print("   Primero genera 50 PLS con el nuevo prompt:")
        print("   1. cd scripts")
        print("   2. python generar_pls_sinteticos.py")
        print("   3. Selecciona opción 1 (modo prueba)")
        print("   4. Renombra el archivo generado a 'pls_prueba_50_v2.csv'")
        return
    
    mostrar_comparacion(df_v1, df_v2)


if __name__ == "__main__":
    main()

