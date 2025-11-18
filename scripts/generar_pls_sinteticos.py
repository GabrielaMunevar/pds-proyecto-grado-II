"""
Script para generar Plain Language Summaries (PLS) sintéticos usando OpenAI API.

Este script:
- Carga textos médicos de revisiones Cochrane
- Genera PLS usando gpt-4o-mini
- Implementa rate limiting y manejo de errores robusto
- Guarda checkpoints periódicos
- Calcula métricas de evaluación automáticas
- Estima costos en tiempo real

Autor: Proyecto de Grado
Fecha: 2025-11-16
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import textstat
import json

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pls_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración de la API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = 'gpt-4o-mini'
MAX_RETRIES = 3
RETRY_DELAY = 2  # segundos
RATE_LIMIT_REQUESTS_PER_MINUTE = 50
CHECKPOINT_INTERVAL = 100

# Costos de la API (por 1M tokens)
COST_INPUT_PER_1M = 0.15  # USD
COST_OUTPUT_PER_1M = 0.60  # USD

# Rutas
DATA_PATH = Path('data/processed/dataset_clean.csv')
OUTPUT_DIR = Path('data/synthetic_pls')
CHECKPOINT_DIR = Path('checkpoints')
ERROR_LOG = Path('errores.log')

# Crear directorios si no existen
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert in creating Plain Language Summaries (PLS) for Cochrane systematic reviews.

OBJECTIVE: Write for 8th grade reading level (13-14 year olds).

MANDATORY NUMERICAL TARGETS:
- Flesch Reading Ease: 60-70 (minimum 55)
- Flesch-Kincaid Grade: 7-9 (maximum 10)
- Length: 150-250 words
- Sentences: ≤12 words each

CRITICAL RULES:

1. ULTRA-SHORT SENTENCES (MOST IMPORTANT):
   • Maximum 12 words per sentence
   • If you have more than 12 words: split into TWO sentences
   • Use periods (.) frequently
   • Avoid commas, use periods instead

   BAD: "Researchers reviewed 15 studies that included 3,456 people to evaluate whether the treatment works."
   GOOD: "Researchers reviewed 15 studies. The studies included 3,456 people. They wanted to know if the treatment works."

2. SIMPLE WORDS (1-2 syllables 90% of the time):
   Required transformations:
   • "administer" → "give"
   • "evaluate" → "check" or "test"
   • "demonstrate" → "show"
   • "evidence" → "proof"
   • "significant" → "clear" or "important"
   • "participants" → "people"
   • "implement" → "use" or "do"
   • "treatment" → "medicine" or simple name
   • "effective" → "works"
   • "adverse" → "bad" or "harmful"

3. EXPLAIN EVERYTHING IMMEDIATELY:
   BAD: "They received corticosteroids"
   GOOD: "They received corticosteroids. These are drugs that reduce swelling."
   
   BAD: "Randomized study"
   GOOD: "Doctors randomly split people into two groups"

4. FIXED STRUCTURE (Follow this pattern):
   
   PARAGRAPH 1 (2-3 sentences):
   • Sentence 1: What is the health problem?
   • Sentence 2: Who does it affect?
   
   PARAGRAPH 2 (2-3 sentences):
   • Sentence 3: What did researchers want to know?
   • Sentence 4: How many studies did they review?
   
   PARAGRAPH 3 (3-4 sentences):
   • Sentences 5-6: What did they find? (Main result)
   • Sentence 7: Result in simple numbers
   
   PARAGRAPH 4 (2 sentences):
   • Sentence 8: How good are the studies?
   • Sentence 9: What does this mean?

5. SIMPLE NUMBERS:
   BAD: "RR 0.72 (95% CI 0.58-0.89, p<0.001)"
   GOOD: "About 3 out of 10 people improved with the medicine"
   
   BAD: "28% reduction"
   GOOD: "It helped about 3 in every 10 people"

6. EVIDENCE QUALITY (Choose ONE phrase):
   • Low: "The studies are small. We are not very sure."
   • Moderate: "The studies are good. We are fairly sure."
   • High: "The studies are very good. We are confident."

7. ACTIVE VOICE:
   BAD: "An evaluation was performed"
   GOOD: "Doctors evaluated"
   
   BAD: "Improvement was observed"
   GOOD: "People improved"

CORRECT TRANSFORMATION EXAMPLES:

EXAMPLE 1 (Good - FRE=64, FKG=8.2):
ORIGINAL TECHNICAL TEXT:
"Systematic review of 12 RCTs (n=2,890) demonstrated that antibiotic 
prophylaxis significantly reduced postoperative infection rates 
(RR 0.65, 95% CI 0.52-0.81, moderate quality evidence)."

V3 PLS:
"Infections after surgery are common. They can be dangerous. Doctors 
reviewed 12 studies. The studies had 2,890 people. They tested giving 
antibiotics before surgery. Antibiotics are medicines that kill bacteria. 
People who took antibiotics had fewer infections. About 13 fewer people 
out of 100 got an infection. The studies are good. The results are reliable."

Analysis:
- 10 sentences
- Average: 9.4 words/sentence
- Simple words: 95%
- No jargon

EXAMPLE 2 (Good - FRE=61, FKG=8.7):
ORIGINAL TECHNICAL TEXT:
"Meta-analysis of 8 trials (n=1,456) found no significant difference 
in pain scores between acupuncture and sham (MD -0.5, 95% CI -1.2 to 0.2, 
I²=62%, low quality evidence)."

V3 PLS:
"Many people have chronic pain. The pain lasts months or years. Doctors 
reviewed 8 studies. The studies had 1,456 people. They compared real 
acupuncture with fake acupuncture. Acupuncture uses needles in the skin. 
They found no clear difference between the two. Both groups had similar pain. 
The studies are small. We are not very sure about the results."

Analysis:
- 10 sentences
- Average: 8.8 words/sentence
- Explains "acupuncture"
- Evidence quality clear

BAD EXAMPLE (What NOT to do):

INCORRECT PLS (FRE=42, FKG=12):
"A systematic review evaluated the efficacy of inhaled corticosteroids 
in patients with COPD exacerbations, finding that early administration 
significantly reduced hospitalization rates and improved pulmonary 
function parameters."

Problems:
- 1 sentence with 32 words
- Jargon: "exacerbations", "COPD", "parameters"
- College level

CORRECT VERSION (FRE=65, FKG=7.9):
"Some people have COPD. It is a lung disease. It gets worse sometimes. 
Doctors reviewed studies about a medicine. The medicine is inhaled. 
It goes straight to the lungs. The medicine helped people. Fewer people 
went to the hospital. They breathed better. The studies are good."

CHECKLIST BEFORE RESPONDING:
- Are all sentences ≤12 words?
- Did I use 1-2 syllable words 90%+ of the time?
- Did I explain all technical terms?
- Would a 13-year-old understand this?
- Did I use the 4-paragraph structure?
- Are numbers in simple format?
- Did I mention study quality?

If any answer is NO → REWRITE

Respond ONLY with the PLS. No titles, explanations, or metadata."""

USER_PROMPT_TEMPLATE = """Generate a Plain Language Summary of the following Cochrane systematic review:

{texto_original}

Remember: 150-250 words, simple language, preserve main findings."""

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def contar_palabras(texto: str) -> int:
    """
    Cuenta el número de palabras en un texto.
    
    Args:
        texto: Texto a analizar
        
    Returns:
        Número de palabras
    """
    if not texto or not isinstance(texto, str):
        return 0
    return len(texto.split())


def estimar_tokens(num_palabras: int) -> int:
    """
    Estima el número de tokens basándose en el número de palabras.
    Usa ratio de 1.33 palabras por token (aproximación común para inglés/español).
    
    Args:
        num_palabras: Número de palabras
        
    Returns:
        Número estimado de tokens
    """
    return int(num_palabras * 1.33)


def calcular_costo(tokens_input: int, tokens_output: int) -> float:
    """
    Calcula el costo de una llamada a la API.
    
    Args:
        tokens_input: Tokens de entrada
        tokens_output: Tokens de salida
        
    Returns:
        Costo en USD
    """
    costo_input = (tokens_input / 1_000_000) * COST_INPUT_PER_1M
    costo_output = (tokens_output / 1_000_000) * COST_OUTPUT_PER_1M
    return costo_input + costo_output


def calcular_metricas_legibilidad(texto: str) -> Dict[str, float]:
    """
    Calcula métricas de legibilidad para un texto.
    
    Args:
        texto: Texto a analizar
        
    Returns:
        Diccionario con métricas de legibilidad
    """
    if not texto or not isinstance(texto, str):
        return {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0
        }
    
    try:
        flesch_ease = textstat.flesch_reading_ease(texto)
        flesch_grade = textstat.flesch_kincaid_grade(texto)
        
        return {
            'flesch_reading_ease': round(flesch_ease, 2),
            'flesch_kincaid_grade': round(flesch_grade, 2)
        }
    except Exception as e:
        logger.warning(f"Error calculando métricas de legibilidad: {e}")
        return {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0
        }


def log_error(texto_id: int, error: str):
    """
    Registra un error en el archivo de log.
    
    Args:
        texto_id: ID del texto que falló
        error: Descripción del error
    """
    with open(ERROR_LOG, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().isoformat()
        f.write(f"{timestamp} | ID: {texto_id} | Error: {error}\n")

# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

def verificar_api() -> bool:
    """
    Verifica que la conexión con la API de OpenAI funcione correctamente.
    
    Returns:
        True si la conexión es exitosa, False en caso contrario
    """
    logger.info("Verificando conexión con OpenAI API...")
    
    if not OPENAI_API_KEY:
        logger.error(" OPENAI_API_KEY no encontrada en .env")
        return False
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Test simple con un mensaje corto
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Test"}
            ],
            max_tokens=10
        )
        
        logger.info(f" Conexión exitosa con OpenAI API")
        logger.info(f"   Modelo: {MODEL}")
        logger.info(f"   Rate limit: {RATE_LIMIT_REQUESTS_PER_MINUTE} requests/min")
        return True
        
    except Exception as e:
        logger.error(f" Error conectando con OpenAI API: {e}")
        return False


def generar_pls(texto_original: str, client: OpenAI) -> Optional[str]:
    """
    Genera un Plain Language Summary para un texto médico.
    
    Implementa reintentos con exponential backoff en caso de errores.
    
    Args:
        texto_original: Texto técnico a simplificar
        client: Cliente de OpenAI
        
    Returns:
        PLS generado o None si falla después de todos los reintentos
    """
    if not texto_original or not isinstance(texto_original, str) or len(texto_original.strip()) == 0:
        logger.warning("Texto vacío o inválido")
        return None
    
    user_prompt = USER_PROMPT_TEMPLATE.format(texto_original=texto_original)
    
    for intento in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            pls_generado = response.choices[0].message.content.strip()
            return pls_generado
            
        except Exception as e:
            wait_time = RETRY_DELAY * (2 ** intento)  # Exponential backoff
            logger.warning(f"Intento {intento + 1}/{MAX_RETRIES} falló: {e}")
            
            if intento < MAX_RETRIES - 1:
                logger.info(f"Esperando {wait_time}s antes de reintentar...")
                time.sleep(wait_time)
            else:
                logger.error(f"Falló después de {MAX_RETRIES} intentos")
                return None
    
    return None


def cargar_checkpoint(checkpoint_path: Path) -> Optional[pd.DataFrame]:
    """
    Carga un checkpoint previo si existe.
    
    Args:
        checkpoint_path: Ruta al archivo de checkpoint
        
    Returns:
        DataFrame con los datos del checkpoint o None si no existe
    """
    if checkpoint_path.exists():
        try:
            df = pd.read_csv(checkpoint_path)
            logger.info(f" Checkpoint cargado: {len(df)} registros procesados")
            return df
        except Exception as e:
            logger.error(f"Error cargando checkpoint: {e}")
            return None
    return None


def guardar_checkpoint(df: pd.DataFrame, checkpoint_path: Path):
    """
    Guarda un checkpoint del progreso actual.
    
    Args:
        df: DataFrame con los datos a guardar
        checkpoint_path: Ruta donde guardar el checkpoint
    """
    try:
        df.to_csv(checkpoint_path, index=False, encoding='utf-8')
        logger.info(f" Checkpoint guardado: {len(df)} registros")
    except Exception as e:
        logger.error(f"Error guardando checkpoint: {e}")


def procesar_lote(
    textos_df: pd.DataFrame,
    n_textos: int,
    output_path: Path,
    checkpoint_path: Path
) -> pd.DataFrame:
    """
    Procesa un lote de textos y genera PLS para cada uno.
    
    Args:
        textos_df: DataFrame con los textos a procesar
        n_textos: Número de textos a procesar
        output_path: Ruta donde guardar el resultado final
        checkpoint_path: Ruta para guardar checkpoints periódicos
        
    Returns:
        DataFrame con los resultados
    """
    logger.info(f"Iniciando procesamiento de {n_textos} textos")
    
    # Verificar checkpoint previo
    resultados_df = cargar_checkpoint(checkpoint_path)
    
    if resultados_df is not None:
        textos_procesados = len(resultados_df)
        logger.info(f"Reanudando desde texto {textos_procesados + 1}")
    else:
        resultados_df = pd.DataFrame()
        textos_procesados = 0
    
    # Seleccionar textos a procesar
    textos_a_procesar = textos_df.iloc[textos_procesados:n_textos]
    
    if len(textos_a_procesar) == 0:
        logger.info(" No hay más textos por procesar")
        return resultados_df
    
    # Inicializar cliente de OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Variables para tracking
    tiempo_inicio = time.time()
    tiempo_ultima_request = time.time()
    requests_en_minuto = 0
    costo_total = 0.0
    
    # Procesar textos
    resultados = []
    
    with tqdm(total=len(textos_a_procesar), desc="Generando PLS") as pbar:
        for idx, row in textos_a_procesar.iterrows():
            # Control de rate limiting
            tiempo_actual = time.time()
            if tiempo_actual - tiempo_ultima_request < 60:
                requests_en_minuto += 1
                if requests_en_minuto >= RATE_LIMIT_REQUESTS_PER_MINUTE:
                    tiempo_espera = 60 - (tiempo_actual - tiempo_ultima_request)
                    logger.info(f"⏳ Rate limit alcanzado. Esperando {tiempo_espera:.1f}s...")
                    time.sleep(tiempo_espera)
                    requests_en_minuto = 0
                    tiempo_ultima_request = time.time()
            else:
                requests_en_minuto = 0
                tiempo_ultima_request = tiempo_actual
            
            # Generar PLS
            texto_original = row['texto_original']
            pls_generado = generar_pls(texto_original, client)
            
            if pls_generado is None:
                log_error(idx, "Falló después de todos los reintentos")
                pbar.update(1)
                continue
            
            # Calcular métricas
            longitud_original = contar_palabras(texto_original)
            longitud_pls = contar_palabras(pls_generado)
            tokens_input = estimar_tokens(longitud_original + contar_palabras(SYSTEM_PROMPT))
            tokens_output = estimar_tokens(longitud_pls)
            costo = calcular_costo(tokens_input, tokens_output)
            metricas_legibilidad = calcular_metricas_legibilidad(pls_generado)
            ratio_compresion = longitud_pls / longitud_original if longitud_original > 0 else 0
            
            # Guardar resultado
            resultado = {
                'id': idx,
                'texto_original': texto_original,
                'pls_generado': pls_generado,
                'longitud_original': longitud_original,
                'longitud_pls': longitud_pls,
                'tokens_input': tokens_input,
                'tokens_output': tokens_output,
                'costo_estimado': round(costo, 6),
                'flesch_reading_ease': metricas_legibilidad['flesch_reading_ease'],
                'flesch_kincaid_grade': metricas_legibilidad['flesch_kincaid_grade'],
                'ratio_compresion': round(ratio_compresion, 3),
                'timestamp': datetime.now().isoformat()
            }
            
            resultados.append(resultado)
            costo_total += costo
            
            # Actualizar progress bar
            pbar.set_postfix({
                'costo': f'${costo_total:.4f}',
                'palabras_pls': longitud_pls
            })
            pbar.update(1)
            
            # Guardar checkpoint cada N textos
            if len(resultados) % CHECKPOINT_INTERVAL == 0:
                df_temp = pd.concat([resultados_df, pd.DataFrame(resultados)], ignore_index=True)
                guardar_checkpoint(df_temp, checkpoint_path)
                
                # Mostrar estadísticas intermedias
                tiempo_transcurrido = time.time() - tiempo_inicio
                textos_totales_procesados = textos_procesados + len(resultados)
                velocidad = textos_totales_procesados / (tiempo_transcurrido / 60)
                logger.info(f"\n Estadísticas intermedias:")
                logger.info(f"   Textos procesados: {textos_totales_procesados}")
                logger.info(f"   Velocidad: {velocidad:.1f} textos/min")
                logger.info(f"   Costo acumulado: ${costo_total:.4f}")
    
    # Combinar con resultados previos
    if len(resultados) > 0:
        df_nuevos = pd.DataFrame(resultados)
        if not resultados_df.empty:
            resultados_df = pd.concat([resultados_df, df_nuevos], ignore_index=True)
        else:
            resultados_df = df_nuevos
    
    # Guardar resultado final
    resultados_df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f" Resultados guardados en: {output_path}")
    
    # Mostrar estadísticas finales
    tiempo_total = time.time() - tiempo_inicio
    mostrar_estadisticas_finales(resultados_df, costo_total, tiempo_total)
    
    return resultados_df


def mostrar_estadisticas_finales(df: pd.DataFrame, costo_total: float, tiempo_total: float):
    """
    Muestra estadísticas finales del procesamiento.
    
    Args:
        df: DataFrame con los resultados
        costo_total: Costo total en USD
        tiempo_total: Tiempo total en segundos
    """
    logger.info("\n" + "="*60)
    logger.info(" ESTADÍSTICAS FINALES")
    logger.info("="*60)
    
    logger.info(f"\n  TIEMPO:")
    logger.info(f"   Total: {tiempo_total/60:.2f} minutos")
    logger.info(f"   Velocidad: {len(df)/(tiempo_total/60):.1f} textos/min")
    
    logger.info(f"\n COSTOS:")
    logger.info(f"   Total: ${costo_total:.4f}")
    logger.info(f"   Promedio por texto: ${costo_total/len(df):.6f}")
    
    logger.info(f"\n LONGITUD:")
    logger.info(f"   Original promedio: {df['longitud_original'].mean():.0f} palabras")
    logger.info(f"   PLS promedio: {df['longitud_pls'].mean():.0f} palabras")
    logger.info(f"   Ratio compresión: {df['ratio_compresion'].mean():.2f}")
    
    logger.info(f"\n LEGIBILIDAD:")
    logger.info(f"   Flesch Reading Ease: {df['flesch_reading_ease'].mean():.1f}")
    logger.info(f"   Flesch-Kincaid Grade: {df['flesch_kincaid_grade'].mean():.1f}")
    
    logger.info(f"\n Total de PLS generados: {len(df)}")
    logger.info("="*60 + "\n")


def mostrar_estimacion_costos(n_textos: int, longitud_promedio: int = 520):
    """
    Muestra una estimación de costos antes de procesar.
    
    Args:
        n_textos: Número de textos a procesar
        longitud_promedio: Longitud promedio de los textos en palabras
    """
    # Estimaciones
    palabras_prompt_sistema = contar_palabras(SYSTEM_PROMPT)
    tokens_input_por_texto = estimar_tokens(longitud_promedio + palabras_prompt_sistema)
    tokens_output_por_texto = estimar_tokens(200)  # PLS promedio estimado
    costo_por_texto = calcular_costo(tokens_input_por_texto, tokens_output_por_texto)
    costo_total = costo_por_texto * n_textos
    
    logger.info("\n" + "="*60)
    logger.info(" ESTIMACIÓN DE COSTOS")
    logger.info("="*60)
    logger.info(f"Textos a procesar: {n_textos:,}")
    logger.info(f"Tokens input/texto: ~{tokens_input_por_texto:,}")
    logger.info(f"Tokens output/texto: ~{tokens_output_por_texto:,}")
    logger.info(f"Costo por texto: ${costo_por_texto:.6f}")
    logger.info(f"COSTO TOTAL ESTIMADO: ${costo_total:.2f}")
    logger.info("="*60 + "\n")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal del script.
    """
    logger.info("="*60)
    logger.info(" GENERADOR DE PLAIN LANGUAGE SUMMARIES")
    logger.info("="*60)
    
    # Verificar API
    if not verificar_api():
        logger.error("No se pudo conectar con la API. Abortando.")
        return
    
    # Cargar dataset
    logger.info(f"\nCargando dataset: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
        logger.info(f"   Total de registros: {len(df):,}")
    except Exception as e:
        logger.error(f" Error cargando dataset: {e}")
        return
    
    # Filtrar por 'non_pls'
    if 'label' in df.columns:
        df_non_pls = df[df['label'] == 'non_pls'].copy()
        logger.info(f"   Registros 'non_pls': {len(df_non_pls):,}")
    else:
        logger.warning("  Columna 'label' no encontrada. Usando todos los registros.")
        df_non_pls = df.copy()
    
    # Verificar columna 'texto_original'
    if 'texto_original' not in df_non_pls.columns:
        logger.error(" Columna 'texto_original' no encontrada en el dataset")
        return
    
    # Eliminar textos vacíos
    df_non_pls = df_non_pls[df_non_pls['texto_original'].notna()].copy()
    df_non_pls = df_non_pls[df_non_pls['texto_original'].str.strip() != ''].copy()
    logger.info(f"   Registros válidos: {len(df_non_pls):,}")
    
    # Resetear índices
    df_non_pls = df_non_pls.reset_index(drop=True)
    
    # Menú de opciones
    print("\n" + "="*60)
    print("SELECCIONA MODO DE EJECUCIÓN:")
    print("="*60)
    print("1. Modo PRUEBA: Generar 50 PLS")
    print("2. Modo PRODUCCIÓN: Generar 10,000 PLS")
    print("3. Modo PERSONALIZADO: Especificar cantidad")
    print("0. Salir")
    print("="*60)
    
    try:
        opcion = input("\nIngresa tu opción (0-3): ").strip()
    except KeyboardInterrupt:
        logger.info("\n\n Proceso cancelado por el usuario")
        return
    
    # Determinar cantidad de textos y archivo de salida
    if opcion == '1':
        n_textos = min(50, len(df_non_pls))
        output_filename = 'pls_prueba_50.csv'
        checkpoint_filename = 'checkpoint_prueba_50.csv'
    elif opcion == '2':
        n_textos = min(10000, len(df_non_pls))
        output_filename = 'pls_produccion_10k.csv'
        checkpoint_filename = 'checkpoint_produccion_10k.csv'
    elif opcion == '3':
        try:
            n_textos = int(input(f"¿Cuántos PLS deseas generar? (máx {len(df_non_pls):,}): "))
            n_textos = min(n_textos, len(df_non_pls))
            output_filename = f'pls_custom_{n_textos}.csv'
            checkpoint_filename = f'checkpoint_custom_{n_textos}.csv'
        except (ValueError, KeyboardInterrupt):
            logger.error(" Entrada inválida")
            return
    elif opcion == '0':
        logger.info("👋 Saliendo...")
        return
    else:
        logger.error(" Opción inválida")
        return
    
    output_path = OUTPUT_DIR / output_filename
    checkpoint_path = CHECKPOINT_DIR / checkpoint_filename
    
    # Mostrar estimación de costos
    longitud_promedio = df_non_pls['texto_original'].apply(contar_palabras).mean()
    mostrar_estimacion_costos(n_textos, int(longitud_promedio))
    
    # Confirmación
    try:
        confirmacion = input("¿Deseas continuar? (s/n): ").strip().lower()
        if confirmacion not in ['s', 'si', 'sí', 'y', 'yes']:
            logger.info(" Proceso cancelado")
            return
    except KeyboardInterrupt:
        logger.info("\n\n Proceso cancelado por el usuario")
        return
    
    # Procesar lote
    try:
        resultados_df = procesar_lote(
            textos_df=df_non_pls,
            n_textos=n_textos,
            output_path=output_path,
            checkpoint_path=checkpoint_path
        )
        
        logger.info(f"\n Proceso completado exitosamente")
        logger.info(f" Archivo de salida: {output_path}")
        
        if ERROR_LOG.exists():
            logger.warning(f"  Algunos textos fallaron. Ver: {ERROR_LOG}")
        
    except KeyboardInterrupt:
        logger.info("\n\n  Proceso interrumpido por el usuario")
        logger.info(f" Progreso guardado en: {checkpoint_path}")
        logger.info("   Puedes reanudar ejecutando el script nuevamente")
    except Exception as e:
        logger.error(f"\n Error durante el procesamiento: {e}")
        logger.info(f" Progreso guardado en: {checkpoint_path}")

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    main()

