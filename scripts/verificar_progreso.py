"""Verificar progreso de generación de PLS"""
import pandas as pd
from pathlib import Path

archivo = Path('../data/synthetic_pls/pls_produccion_10k.csv')

if archivo.exists():
    df = pd.read_csv(archivo)
    
    print("="*70)
    print("PROGRESO DE GENERACIÓN")
    print("="*70)
    print(f"\nPLS generados: {len(df)}")
    print(f"\nMétricas promedio:")
    print(f"   Flesch Reading Ease: {df['flesch_reading_ease'].mean():.1f} (target: 60-70)")
    print(f"   Flesch-Kincaid Grade: {df['flesch_kincaid_grade'].mean():.1f} (target: ~8)")
    print(f"   Longitud PLS: {df['longitud_pls'].mean():.0f} palabras")
    print(f"   Ratio compresión: {df['ratio_compresion'].mean():.2f}")
    print(f"\nCosto acumulado: ${df['costo_estimado'].sum():.4f}")
    
    # Proyección
    if len(df) < 10000:
        factor = 10000 / len(df)
        costo_proyectado = df['costo_estimado'].sum() * factor
        print(f"\nProyección a 10,000 PLS:")
        print(f"   Costo estimado: ${costo_proyectado:.2f}")
else:
    print("ERROR: No se encontró archivo de producción")

