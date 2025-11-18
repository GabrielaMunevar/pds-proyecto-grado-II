"""
Script OPTIMIZADO con async para generar Plain Language Summaries (PLS) sintéticos.

Versión con paralelización usando asyncio para reducir tiempos a la mitad.
Mantiene la misma calidad de respuestas y todas las características del script original.

Mejoras:
- Procesamiento paralelo de hasta 10 textos simultáneos
- Respeta rate limits (50 req/min)
- Reduce tiempo de 6-7h a 3-4h para 10k PLS
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import textstat
from pathlib import Path
from datetime import datetime
from tqdm.asyncio import tqdm
import logging
from dotenv import load_dotenv
import json

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError(" OPENAI_API_KEY no encontrada en .env")

# Modelo y parámetros
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.7
MAX_TOKENS = 500

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 50
CONCURRENT_REQUESTS = 10  # Número de requests paralelas

# Rutas
DATA_PATH = Path('../data/processed/dataset_clean.csv')
OUTPUT_DIR = Path('../data/synthetic_pls')
CHECKPOINT_DIR = Path('../data/checkpoints')

# Crear directorios
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
Path('../logs').mkdir(parents=True, exist_ok=True)

# Logging (UTF-8 para evitar problemas con caracteres especiales)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/pls_generation_async.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# PROMPT (V3 - INGLÉS - MEJOR RESULTADO)
# =============================================================================

SYSTEM_PROMPT = """You are a medical expert specialized in creating EXTREMELY SIMPLE Plain Language Summaries (PLS) of Cochrane systematic reviews.

 PRIMARY GOAL: 
Write so that a 13-14 year old person (8th grade) can understand PERFECTLY without help.

 STRICT SIMPLICITY RULES:

1. SHORT SENTENCES (CRITICAL):
   - MAXIMUM 15 words per sentence
   - If a sentence has >15 words, split it into two
   - Use periods frequently
   - Avoid complex subordinate clauses

2. ULTRA-SIMPLE VOCABULARY:
    NEVER use these words without explaining:
   - Latin medical terms (placebo, in vitro, etc.)
   - Statistics (relative risk, confidence interval)
   - Methodology (randomized, systematic, meta-analysis)
   
    ALWAYS USE:
   - Everyday words of <3 syllables when possible
   - Simple verbs (do, have, be, use, help)
   - Concrete nouns

3. IMMEDIATE EXPLANATIONS:
   If you must use a technical term:
    "Patients received corticosteroid treatment"
    "Patients received corticosteroids (anti-inflammatory medicines)"

4. ULTRA-CLEAR STRUCTURE:
   Paragraph 1 (2-3 sentences): What is the health problem?
   Paragraph 2 (2-3 sentences): What did researchers want to know?
   Paragraph 3 (3-4 sentences): What did they find?
   Paragraph 4 (2 sentences): How reliable is this?

5. SIMPLIFIED NUMBERS:
    "RR 0.72 (95% CI 0.58-0.89)"
    "The treatment reduced the problem in almost 3 out of 10 people"
   
    "p<0.001"
    "This difference is not by chance"

6. EVIDENCE QUALITY (everyday language):
    "Low quality evidence"
    "The studies are small. We are not very sure"
   
    "High quality evidence"  
    "The studies are large and well done. We are quite sure"

7. ACTIVE AND DIRECT VOICE:
    "Treatment efficacy was evaluated"
    "Researchers tested if the treatment works"

8. AVOID ACADEMIC JARGON:
    "A systematic review was conducted"
    "Researchers reviewed all available studies"
   
    "A significant reduction was observed"
    "The treatment helped reduce the problem"

 NUMERICAL TARGETS YOU MUST MEET:
- Length: 150-250 words
- Average words per sentence: <15
- Reading level 8th grade (use simple phrases)
- No technical terms NOT explained

 MENTAL PROCESS BEFORE RESPONDING:
1. Read the original text
2. Identify the 3-4 MOST important points
3. Write each point in ONE simple sentence
4. Review: Would a 13-year-old understand it?
5. If in doubt → SIMPLIFY MORE

RESPOND ONLY WITH THE PLS. No titles or additional explanations."""

USER_PROMPT_TEMPLATE = """Generate a Plain Language Summary of the following Cochrane systematic review:

{texto_original}
"""

# =============================================================================
# FUNCIONES DE CÁLCULO
# =============================================================================

def calcular_metricas(texto_original, pls_generado):
    """Calcula todas las métricas para un par texto-PLS"""
    
    # Longitudes
    longitud_original = len(texto_original.split())
    longitud_pls = len(pls_generado.split())
    
    # Tokens estimados
    tokens_input = int(longitud_original * 1.33)
    tokens_output = int(longitud_pls * 1.33)
    
    # Costo (gpt-4o-mini: $0.15/1M input, $0.60/1M output)
    costo_input = (tokens_input / 1_000_000) * 0.15
    costo_output = (tokens_output / 1_000_000) * 0.60
    costo_total = costo_input + costo_output
    
    # Métricas de legibilidad
    flesch_reading_ease = textstat.flesch_reading_ease(pls_generado)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(pls_generado)
    
    # Ratio de compresión
    ratio_compresion = longitud_pls / longitud_original if longitud_original > 0 else 0
    
    return {
        'longitud_original': longitud_original,
        'longitud_pls': longitud_pls,
        'tokens_input': tokens_input,
        'tokens_output': tokens_output,
        'costo_estimado': costo_total,
        'flesch_reading_ease': flesch_reading_ease,
        'flesch_kincaid_grade': flesch_kincaid_grade,
        'ratio_compresion': ratio_compresion
    }

# =============================================================================
# GENERACIÓN ASYNC CON OPENAI
# =============================================================================

class AsyncRateLimiter:
    """Control de rate limiting para requests asíncronos"""
    
    def __init__(self, max_per_minute):
        self.max_per_minute = max_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Espera si es necesario para respetar rate limit"""
        async with self.lock:
            now = datetime.now()
            # Limpiar requests antiguos (>1 minuto)
            self.requests = [r for r in self.requests if (now - r).seconds < 60]
            
            if len(self.requests) >= self.max_per_minute:
                # Esperar hasta que el request más antiguo tenga >1 minuto
                oldest = self.requests[0]
                wait_time = 60 - (now - oldest).seconds
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self.requests = []
            
            self.requests.append(now)

rate_limiter = AsyncRateLimiter(MAX_REQUESTS_PER_MINUTE)

async def generar_pls_async(session, texto_original, max_reintentos=3):
    """
    Genera un PLS usando OpenAI API de forma asíncrona.
    
    Args:
        session: aiohttp ClientSession
        texto_original: Texto médico original
        max_reintentos: Número máximo de reintentos en caso de error
    
    Returns:
        str: PLS generado o None si falla
    """
    
    for intento in range(max_reintentos):
        try:
            # Respetar rate limit
            await rate_limiter.acquire()
            
            # Preparar request
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(texto_original=texto_original)}
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS
            }
            
            # Hacer request
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    pls = data['choices'][0]['message']['content'].strip()
                    return pls
                else:
                    error_text = await response.text()
                    logger.warning(f"Error {response.status} en intento {intento + 1}: {error_text}")
                    
                    if response.status == 429:  # Rate limit
                        wait_time = 2 ** intento  # Exponential backoff
                        await asyncio.sleep(wait_time)
                    elif response.status >= 500:  # Server error
                        await asyncio.sleep(2 ** intento)
                    else:
                        return None
        
        except asyncio.TimeoutError:
            logger.warning(f"Timeout en intento {intento + 1}")
            await asyncio.sleep(2 ** intento)
        
        except Exception as e:
            logger.error(f"Error inesperado en intento {intento + 1}: {e}")
            await asyncio.sleep(2 ** intento)
    
    logger.error(f"Falló después de {max_reintentos} intentos")
    return None

# =============================================================================
# PROCESAMIENTO POR LOTES
# =============================================================================

async def procesar_lote_async(textos_df, inicio, fin, output_file):
    """
    Procesa un lote de textos de forma asíncrona.
    
    Args:
        textos_df: DataFrame con los textos originales
        inicio: Índice de inicio
        fin: Índice de fin
        output_file: Archivo de salida para resultados
    """
    
    resultados = []
    lote = textos_df.iloc[inicio:fin]
    
    # Crear sesión aiohttp compartida
    async with aiohttp.ClientSession() as session:
        
        # Crear tareas para procesamiento concurrente
        tasks = []
        for idx, row in lote.iterrows():
            task = generar_pls_async(session, row['texto_original'])
            tasks.append((idx, row, task))
        
        # Procesar con barra de progreso
        pbar = tqdm(total=len(tasks), desc=f"Lote {inicio}-{fin}", leave=False)
        
        for idx, row, task in tasks:
            pls_generado = await task
            pbar.update(1)
            
            if pls_generado:
                # Calcular métricas
                metricas = calcular_metricas(row['texto_original'], pls_generado)
                
                # Guardar resultado
                resultado = {
                    'id': idx,
                    'texto_original': row['texto_original'],
                    'pls_generado': pls_generado,
                    **metricas,
                    'timestamp': datetime.now().isoformat()
                }
                resultados.append(resultado)
            else:
                logger.error(f"Falló la generación para ID {idx}")
        
        pbar.close()
    
    # Guardar resultados
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        
        # Agregar o crear archivo
        if output_file.exists():
            df_existente = pd.read_csv(output_file)
            df_final = pd.concat([df_existente, df_resultados], ignore_index=True)
            df_final.to_csv(output_file, index=False)
        else:
            df_resultados.to_csv(output_file, index=False)
        
        logger.info(f"Guardados {len(resultados)} resultados en {output_file}")
    
    return len(resultados)

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

async def main_async():
    """Función principal con procesamiento asíncrono"""
    
    print("\n" + "="*70)
    print("GENERADOR DE PLS SINTÉTICOS (VERSIÓN OPTIMIZADA CON ASYNC)")
    print("="*70)
    
    # Cargar datos
    if not DATA_PATH.exists():
        logger.error(f" No se encuentra el dataset: {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    df_non_pls = df[df['label'] == 'non_pls'].reset_index(drop=True)
    
    print(f"\n Dataset cargado:")
    print(f"   Total textos non_pls: {len(df_non_pls):,}")
    print(f"   Longitud promedio: {df_non_pls['texto_original'].str.split().str.len().mean():.0f} palabras")
    
    # Menú
    print("\n" + "="*70)
    print("OPCIONES:")
    print("="*70)
    print("1. Modo PRUEBA: Generar 50 PLS (~ 2-3 minutos)")
    print("2. Modo PRODUCCIÓN: Generar 10,000 PLS (~ 3-4 horas)")
    print("3. Modo PERSONALIZADO: Especificar cantidad")
    print("="*70)
    
    opcion = input("\nSelecciona una opción (1-3): ").strip()
    
    if opcion == "1":
        n = 50
        output_file = OUTPUT_DIR / "pls_prueba_50.csv"
        print(f"\n Modo PRUEBA: Generando {n} PLS...")
    elif opcion == "2":
        n = 10000
        output_file = OUTPUT_DIR / "pls_produccion_10k.csv"
        print(f"\n Modo PRODUCCIÓN: Generando {n} PLS...")
    elif opcion == "3":
        try:
            n = int(input("Cantidad de PLS a generar: "))
            output_file = OUTPUT_DIR / f"pls_custom_{n}.csv"
            print(f"\n  Modo PERSONALIZADO: Generando {n} PLS...")
        except ValueError:
            print(" Cantidad inválida")
            return
    else:
        print(" Opción inválida")
        return
    
    # Validar cantidad
    if n > len(df_non_pls):
        print(f"  Solo hay {len(df_non_pls)} textos disponibles. Ajustando a {len(df_non_pls)}.")
        n = len(df_non_pls)
    
    # Muestra aleatoria
    muestra = df_non_pls.sample(n=n, random_state=42).reset_index(drop=True)
    
    # Estimación de costos
    longitud_promedio = muestra['texto_original'].str.split().str.len().mean()
    tokens_promedio_input = int(longitud_promedio * 1.33)
    tokens_promedio_output = int(200 * 1.33)  # Estimado 200 palabras PLS
    
    costo_por_pls = (tokens_promedio_input / 1_000_000) * 0.15 + (tokens_promedio_output / 1_000_000) * 0.60
    costo_total_estimado = costo_por_pls * n
    
    tiempo_estimado_min = (n / MAX_REQUESTS_PER_MINUTE) * 1.5  # Factor de seguridad
    
    print(f"\n ESTIMACIÓN DE COSTOS:")
    print(f"   Costo por PLS: ${costo_por_pls:.6f}")
    print(f"   Costo total estimado: ${costo_total_estimado:.2f}")
    print(f"   Tiempo estimado: {tiempo_estimado_min:.1f} minutos ({tiempo_estimado_min/60:.1f} horas)")
    print(f"   Procesamiento paralelo: {CONCURRENT_REQUESTS} requests simultáneos")
    
    confirmar = input("\n¿Continuar? (s/n): ").strip().lower()
    if confirmar != 's':
        print(" Operación cancelada")
        return
    
    # PROCESAR
    print(f"\n Iniciando generación con {CONCURRENT_REQUESTS} workers paralelos...")
    inicio_total = datetime.now()
    
    # Procesar en lotes para respetar concurrencia
    total_procesados = 0
    batch_size = CONCURRENT_REQUESTS
    
    for i in range(0, n, batch_size):
        fin = min(i + batch_size, n)
        procesados = await procesar_lote_async(muestra, i, fin, output_file)
        total_procesados += procesados
        
        # Guardar checkpoint cada 100
        if total_procesados % 100 == 0:
            checkpoint_file = CHECKPOINT_DIR / f"checkpoint_{total_procesados}.csv"
            if output_file.exists():
                df_temp = pd.read_csv(output_file)
                df_temp.to_csv(checkpoint_file, index=False)
                logger.info(f"Checkpoint guardado: {checkpoint_file}")
    
    # RESUMEN FINAL
    fin_total = datetime.now()
    duracion = (fin_total - inicio_total).total_seconds()
    
    print("\n" + "="*70)
    print(" GENERACIÓN COMPLETADA")
    print("="*70)
    
    if output_file.exists():
        df_final = pd.read_csv(output_file)
        
        print(f"\n RESULTADOS:")
        print(f"   Total PLS generados: {len(df_final)}")
        print(f"   Tasa de éxito: {len(df_final)/n*100:.1f}%")
        print(f"   Tiempo total: {duracion/60:.1f} minutos ({duracion/3600:.2f} horas)")
        print(f"   Velocidad: {len(df_final)/(duracion/60):.1f} PLS/minuto")
        
        print(f"\n MÉTRICAS PROMEDIO:")
        print(f"   Longitud PLS: {df_final['longitud_pls'].mean():.0f} palabras")
        print(f"   Flesch Reading Ease: {df_final['flesch_reading_ease'].mean():.1f} (target: 60-70)")
        print(f"   Flesch-Kincaid Grade: {df_final['flesch_kincaid_grade'].mean():.1f} (target: ~8)")
        print(f"   Ratio compresión: {df_final['ratio_compresion'].mean():.2f}")
        
        print(f"\n COSTOS REALES:")
        print(f"   Costo total: ${df_final['costo_estimado'].sum():.4f}")
        print(f"   Costo promedio: ${df_final['costo_estimado'].mean():.6f}")
        
        print(f"\n Archivo guardado: {output_file}")
        
        # Proyección a 10k
        if n < 10000:
            factor = 10000 / n
            print(f"\n PROYECCIÓN A 10,000 PLS:")
            print(f"   Costo estimado: ${df_final['costo_estimado'].sum() * factor:.2f}")
            print(f"   Tiempo estimado: {duracion/60 * factor:.1f} min ({duracion/3600 * factor:.1f} horas)")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Verificar API key
    if not OPENAI_API_KEY:
        print(" Error: OPENAI_API_KEY no configurada en .env")
        exit(1)
    
    # Ejecutar async
    asyncio.run(main_async())

