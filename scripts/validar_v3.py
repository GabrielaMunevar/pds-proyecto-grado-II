"""
Script de validación rápida para resultados V3 (Prompt en Inglés).

Verifica que las métricas de legibilidad cumplan los targets:
- Flesch Reading Ease: 60-70 (mínimo 55)
- Flesch-Kincaid Grade: 7-9 (máximo 10)
- Longitud: 150-250 palabras
"""

import pandas as pd
from pathlib import Path
import sys

def validar_v3():
    """Valida los resultados de V3"""
    
    archivo = Path('../data/synthetic_pls/pls_prueba_50.csv')
    
    if not archivo.exists():
        print("\nERROR: No se encontró pls_prueba_50.csv")
        print(f"   Ruta buscada: {archivo}")
        print("\n   Primero genera 50 PLS:")
        print("   1. cd scripts")
        print("   2. python generar_pls_sinteticos.py")
        print("   3. Opción 1 (Modo PRUEBA)")
        return False
    
    # Cargar datos
    df = pd.read_csv(archivo)
    
    # Calcular métricas
    flesch = df['flesch_reading_ease'].mean()
    flesch_std = df['flesch_reading_ease'].std()
    grade = df['flesch_kincaid_grade'].mean()
    grade_std = df['flesch_kincaid_grade'].std()
    longitud = df['longitud_pls'].mean()
    longitud_std = df['longitud_pls'].std()
    en_rango = len(df[(df['longitud_pls'] >= 150) & (df['longitud_pls'] <= 250)])
    
    print("\n" + "="*70)
    print("VALIDACIÓN V3 - PROMPT INGLÉS ULTRA-SIMPLIFICADO")
    print("="*70)
    
    # Flesch Reading Ease
    print(f"\n1. FLESCH READING EASE:")
    print(f"   Media: {flesch:.1f} (±{flesch_std:.1f})")
    print(f"   Rango: {df['flesch_reading_ease'].min():.1f} - {df['flesch_reading_ease'].max():.1f}")
    
    if 60 <= flesch <= 70:
        print(f"   [OK] PERFECTO - Dentro del target ideal (60-70)")
        flesch_ok = True
    elif 55 <= flesch < 60:
        print(f"   [ADV] ACEPTABLE - Cerca del target (55-60)")
        print(f"      Necesita subir {60 - flesch:.1f} puntos para ser ideal")
        flesch_ok = True
    elif 70 < flesch <= 75:
        print(f"   [ADV] ACEPTABLE - Ligeramente sobre target (70-75)")
        print(f"      Puede ser demasiado simple")
        flesch_ok = True
    else:
        print(f"   [X] FUERA DE RANGO")
        if flesch < 55:
            print(f"      Demasiado complejo. Necesita subir {55 - flesch:.1f} puntos")
        else:
            print(f"      Demasiado simple. Considera reducir")
        flesch_ok = False
    
    # Flesch-Kincaid Grade
    print(f"\n2. FLESCH-KINCAID GRADE LEVEL:")
    print(f"   Media: {grade:.1f} (±{grade_std:.1f})")
    print(f"   Rango: {df['flesch_kincaid_grade'].min():.1f} - {df['flesch_kincaid_grade'].max():.1f}")
    
    if 7 <= grade <= 9:
        print(f"   [OK] PERFECTO - Nivel 8vo grado ideal (7-9)")
        grade_ok = True
    elif grade <= 10:
        print(f"   [ADV] ACEPTABLE - Cerca del target (≤10)")
        print(f"      Necesita bajar {grade - 9:.1f} puntos para ser ideal")
        grade_ok = True
    else:
        print(f"   [X] MUY ALTO - Nivel universitario")
        print(f"      Necesita bajar {grade - 10:.1f} puntos mínimo")
        grade_ok = False
    
    # Longitud
    print(f"\n3. LONGITUD:")
    print(f"   Media: {longitud:.0f} palabras (±{longitud_std:.0f})")
    print(f"   Rango: {df['longitud_pls'].min():.0f} - {df['longitud_pls'].max():.0f}")
    print(f"   En rango 150-250: {en_rango}/50 ({en_rango/50*100:.0f}%)")
    
    if en_rango >= 45:
        print(f"   [OK] EXCELENTE - ≥90% en rango")
        longitud_ok = True
    elif en_rango >= 40:
        print(f"   [ADV] BUENO - ≥80% en rango")
        longitud_ok = True
    else:
        print(f"   [X] INSUFICIENTE - <80% en rango")
        print(f"      Objetivo: al menos 40/50 en rango")
        longitud_ok = False
    
    # Costo
    print(f"\n4. COSTOS:")
    costo_total = df['costo_estimado'].sum()
    costo_promedio = df['costo_estimado'].mean()
    print(f"   Total 50 PLS: ${costo_total:.4f}")
    print(f"   Promedio/PLS: ${costo_promedio:.6f}")
    
    # Proyección a 10k
    costo_10k = costo_promedio * 10000
    print(f"\n   Proyección 10,000 PLS: ${costo_10k:.2f}")
    if costo_10k <= 5:
        print(f"   [OK] Costo razonable")
    else:
        print(f"   [ADV] Costo alto - considerar optimizaciones")
    
    # Ejemplos
    print(f"\n5. DISTRIBUCIÓN DE CALIDAD:")
    
    # Clasificar por calidad
    excelente = df[(df['flesch_reading_ease'] >= 60) & 
                   (df['flesch_reading_ease'] <= 70) &
                   (df['flesch_kincaid_grade'] >= 7) &
                   (df['flesch_kincaid_grade'] <= 9)]
    
    bueno = df[((df['flesch_reading_ease'] >= 55) & 
                (df['flesch_reading_ease'] < 60) |
                (df['flesch_reading_ease'] > 70) & 
                (df['flesch_reading_ease'] <= 75)) &
               (df['flesch_kincaid_grade'] <= 10)]
    
    print(f"   Excelente (FRE 60-70, Grade 7-9): {len(excelente)}/50 ({len(excelente)/50*100:.0f}%)")
    print(f"   Bueno (cerca de targets):          {len(bueno)}/50 ({len(bueno)/50*100:.0f}%)")
    print(f"   Necesita mejora:                   {50 - len(excelente) - len(bueno)}/50")
    
    # Decisión final
    print("\n" + "="*70)
    print("DECISIÓN FINAL:")
    print("="*70)
    
    todos_ok = flesch_ok and grade_ok and longitud_ok
    
    if todos_ok and flesch >= 60 and grade <= 9:
        print("\nEXCELENTE - LISTO PARA PRODUCCIÓN")
        print("\nPróximos pasos:")
        print("  1. Revisar manualmente 10-15 ejemplos")
        print("  2. Ejecutar: python generar_pls_sinteticos.py")
        print("  3. Seleccionar opción 2 (Modo PRODUCCIÓN: 10,000 PLS)")
        print(f"  4. Costo estimado: ${costo_10k:.2f}")
        print(f"  5. Tiempo estimado: ~3.5-4 horas")
        return True
        
    elif todos_ok:
        print("\nADVERTENCIA: ACEPTABLE - PUEDE PROCEDER CON PRECAUCIÓN")
        print("\nLas métricas están cerca de los targets pero no perfectas.")
        print("Recomendaciones:")
        print("  1. Revisar manualmente 15-20 ejemplos")
        print("  2. Considerar ajustes menores al prompt o temperatura")
        print("  3. O proceder con producción si los ejemplos son buenos")
        return True
        
    else:
        print("\nERROR: NECESITA MEJORAS - NO PROCEDER A PRODUCCIÓN AÚN")
        print("\nProblemas identificados:")
        if not flesch_ok:
            print(f"  - Flesch Reading Ease fuera de rango ({flesch:.1f})")
        if not grade_ok:
            print(f"  - Flesch-Kincaid Grade muy alto ({grade:.1f})")
        if not longitud_ok:
            print(f"  - Muchos PLS fuera de rango 150-250 palabras")
        
        print("\nAcciones recomendadas:")
        print("  1. Ajustar temperatura del modelo:")
        print("     En generar_pls_sinteticos.py línea ~350")
        print("     Cambiar temperature=0.7 a temperature=0.5")
        print("  2. O usar GPT-4 en vez de GPT-4o-mini")
        print("  3. O modificar el prompt para enfatizar más simplicidad")
        print("  4. Generar otros 50 PLS de prueba")
        return False
    
    print("="*70 + "\n")


def mostrar_ejemplos(n=5):
    """Muestra n ejemplos aleatorios"""
    
    archivo = Path('../data/synthetic_pls/pls_prueba_50.csv')
    
    if not archivo.exists():
        print("\nERROR: No se encontró el archivo de resultados")
        return
    
    df = pd.read_csv(archivo)
    
    print("\n" + "="*70)
    print(f"MOSTRANDO {n} EJEMPLOS ALEATORIOS")
    print("="*70)
    
    for i, (idx, row) in enumerate(df.sample(n).iterrows(), 1):
        print(f"\n{'='*70}")
        print(f"EJEMPLO {i} (ID: {idx})")
        print(f"{'='*70}")
        print(f"Flesch RE: {row['flesch_reading_ease']:.1f} | FK Grade: {row['flesch_kincaid_grade']:.1f} | Palabras: {row['longitud_pls']}")
        print(f"\n{row['pls_generado']}")
        
        # Clasificación
        if 60 <= row['flesch_reading_ease'] <= 70 and 7 <= row['flesch_kincaid_grade'] <= 9:
            calidad = "[OK] EXCELENTE"
        elif 55 <= row['flesch_reading_ease'] <= 75 and row['flesch_kincaid_grade'] <= 10:
            calidad = "[ADV] BUENO"
        else:
            calidad = "[X] NECESITA MEJORA"
        
        print(f"\nCalidad: {calidad}")


def main():
    """Función principal"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Validar resultados V3')
    parser.add_argument('--ejemplos', type=int, default=0, 
                       help='Mostrar N ejemplos aleatorios')
    
    args = parser.parse_args()
    
    # Validar
    resultado = validar_v3()
    
    # Mostrar ejemplos si se solicita
    if args.ejemplos > 0:
        mostrar_ejemplos(args.ejemplos)
    
    # Retornar código de salida
    sys.exit(0 if resultado else 1)


if __name__ == "__main__":
    main()

