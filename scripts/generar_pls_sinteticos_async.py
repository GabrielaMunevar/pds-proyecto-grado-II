"""
Script ALTAMENTE OPTIMIZADO con async para generar Plain Language Summaries (PLS) sintéticos.

Versión mejorada con paralelización real usando asyncio.gather() y semáforos.
Mantiene la misma calidad de respuestas y todas las características del script original.

Mejoras v3:
- Procesamiento REALMENTE paralelo usando asyncio.gather() (no secuencial)
- Token limiter para respetar límite de 200k TPM (además de 500 RPM)
- 50 requests simultáneos (ajustado para no exceder TPM)
- Rate limiter optimizado con time.monotonic()
- Guardado en batches para reducir I/O
- Checkpoints cada 500 PLS
- Reduce tiempo de 13h a ~2-3h para 10k PLS (4-5x más rápido)
- Soporte para 20k PLS respetando todos los límites de API

Mejoras v4 (CALIDAD Y ANTI-REPETICIÓN):
- frequency_penalty=0.5 y presence_penalty=0.3 para reducir repeticiones
- Temperatura reducida a 0.5 (de 0.7) para mayor consistencia
- MAX_TOKENS reducido a 300 (de 500) para PLS más concisos
- Prompt V8 con ejemplos few-shot y reglas anti-repetición explícitas
- Estructura simplificada a 3 párrafos (evita redundancia)
- Validación de calidad post-generación (longitud, repeticiones)
- Métricas de diversidad léxica: Type-Token Ratio, Repetition Rate
- Detección automática de frases repetidas y baja diversidad
"""

import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import textstat
import time
import sys
from pathlib import Path
from datetime import datetime
from tqdm.asyncio import tqdm
import logging
from dotenv import load_dotenv
import json
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("⚠️  tiktoken no disponible. Usando estimación aproximada de tokens.")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError(" OPENAI_API_KEY no encontrada en .env")

# Modelo y parámetros
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.6  # Reducido de 0.7 para mayor consistencia en resúmenes médicos
MAX_TOKENS = 380  # Reducido de 500 (target: 140-210 palabras ≈ 187-280 tokens)
FREQUENCY_PENALTY = 0.5  # Penalizar repeticiones frecuentes de tokens
PRESENCE_PENALTY = 0.3  # Penalizar repetir temas ya mencionados

# Rate limiting (gpt-4o-mini límites reales: 500 RPM, 200k TPM)
MAX_REQUESTS_PER_MINUTE = 500  # Límite real de OpenAI para gpt-4o-mini
MAX_TOKENS_PER_MINUTE = 200000  # Límite de tokens por minuto
CONCURRENT_REQUESTS = 50  # Reducido para evitar exceder TPM (era 100)

# Rutas (relativas a la raíz del proyecto)
# Detectar si se ejecuta desde scripts/ o desde la raíz
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'scripts' else SCRIPT_DIR

DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'dataset_clean.csv'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'synthetic_pls'
CHECKPOINT_DIR = PROJECT_ROOT / 'data' / 'checkpoints'

# Crear directorios
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logging (UTF-8 para evitar problemas con caracteres especiales)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'pls_generation_async.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# PROMPT (V8.1 - INGLÉS - ANTI-REPETICIÓN + LONGITUD MÍNIMA + READABILITY)
# =============================================================================
# Versión V8.1 mejorada para:
# 1. Anti-Repetición: Reglas explícitas para evitar repetir información.
# 2. Estructura Simplificada: 3 párrafos en vez de 4 para evitar redundancia.
# 3. Few-Shot Learning: Ejemplo concreto de un buen PLS (150 palabras).
# 4. Legibilidad Extrema: Oraciones muy cortas (8-12 palabras).
# 5. LONGITUD MÍNIMA EXPLÍCITA: Regla crítica de 140 palabras mínimo.
# 6. Cero Alucinaciones: Prohibido inventar datos.
# 7. Interpretación Segura: Uso de "reduced risk" en lugar de números inventados.

SYSTEM_PROMPT = """
You are a medical expert writing for an 11-year-old student (6th grade reading level).

GOAL:
- Explain the medical text simply and clearly.
- Keep facts 100% correct based ONLY on the source.
- Use short sentences and simple words.
- Do NOT repeat the same information twice. Each sentence must add NEW information.

INPUT: One medical text.
OUTPUT: One Plain Language Summary (140–210 words).

CRITICAL: WORD COUNT REQUIREMENT:
- MINIMUM: 140 words (REQUIRED - do not stop before reaching this)
- TARGET: 150-200 words (ideal range)
- MAXIMUM: 210 words (do not exceed)
- Count your words as you write. If you have less than 140 words, add more details.

RULES FOR WRITING (READABILITY IS PRIORITY):
1. SHORT SENTENCES: Aim for 8-12 words per sentence. Maximum 15 words.
   - If a sentence has a comma, try to split it into two sentences.
   - Subject -> Verb -> Object.
2. SHORT WORDS: Use words with 1 or 2 syllables if possible.
   - Use "doctor" not "physician".
   - Use "drug" not "medication".
   - Use "use" not "utilize".
   - Use "start" not "initiate".
3. EXPLAIN HARD WORDS: If you must use a medical term (like "randomized"), explain it simply the first time.
   - Example: "randomized (chosen by chance)".
4. ACTIVE VOICE: Say "The drug reduced pain" (not "Pain was reduced by the drug").
5. SIMPLE CONNECTORS: Use "But", "So", "Also". Do NOT use "However", "Therefore", "Furthermore".

RULES FOR FACTS (ZERO HALLUCINATION):
1. NO INVENTING: If the text does not say it, do not write it.
2. NO FAKE NUMBERS: Do not turn "Risk Ratio 0.5" into "50 out of 100 people". This is false.
3. SAFE INTERPRETATION:
   - Instead of fake numbers, say: "reduced the risk", "fewer people had problems", "made little difference".
   - You CAN say "reduced the risk by half" if the Risk Ratio is 0.5.
4. EXACT NUMBERS: Use the numbers from the text.
   - You can change "Twelve" to "12".
   - You can change "82%" to "82 out of 100".

STRUCTURE (3 short paragraphs - NO REPETITION):
1. BACKGROUND: What is the problem? What did researchers want to know?
2. METHODS & RESULTS: What did they do? What did they find? (Combine methods and results in ONE paragraph)
3. MEANING: What does this mean for patients? (Do NOT repeat numbers from paragraph 2. Focus on practical implications.)

ANTI-REPETITION RULES:
- Each sentence must say something NEW.
- Do NOT rephrase the same fact in different words.
- If you mentioned a result with numbers, do NOT mention it again without numbers.
- The conclusion paragraph should focus on IMPLICATIONS, not repeat results.

EXAMPLE (Good PLS - 150 words):

"Researchers wanted to know if vitamin D helps prevent bone breaks in older adults. Many older people break bones easily. This can cause serious health problems.

They studied 5,000 people over age 60. Half took vitamin D pills daily. Half took fake pills. The study lasted 3 years. People who took vitamin D had 15% fewer bone breaks. The vitamin D group also had stronger bones overall.

This means vitamin D may help protect bones in older adults. It is a simple and safe treatment. Talk to your doctor about taking vitamin D. Your doctor can tell you if it is right for you."

FINAL CHECK (MANDATORY):
- Do I have at least 140 words? (Count them - if NO, add more details)
- Are sentences short? (Yes/No)
- Did I repeat any information? (Yes -> Delete it)
- Did I invent any number? (Yes -> Delete it)
- Did I use simple words? (Yes/No)

Respond ONLY with the Plain Language Summary.
"""

USER_PROMPT_TEMPLATE = """Generate a Plain Language Summary of the following Cochrane systematic review:

{texto_original}
"""

# =============================================================================
# FUNCIONES DE CÁLCULO
# =============================================================================

def calcular_diversidad_lexica(texto):
    """
    Calcula métricas de diversidad léxica para detectar repeticiones.
    
    Args:
        texto: Texto a analizar
    
    Returns:
        dict: Métricas de diversidad
    """
    from collections import Counter
    
    # Tokenizar y limpiar
    palabras = texto.lower().split()
    palabras_unicas = set(palabras)
    
    # Type-Token Ratio (TTR): Proporción de palabras únicas
    ttr = len(palabras_unicas) / len(palabras) if len(palabras) > 0 else 0
    
    # Repetition Rate: Detectar bigramas y trigramas repetidos
    bigramas = [' '.join(palabras[i:i+2]) for i in range(len(palabras)-1)]
    trigramas = [' '.join(palabras[i:i+3]) for i in range(len(palabras)-2)]
    
    # Contar repeticiones
    bigram_counts = Counter(bigramas)
    trigram_counts = Counter(trigramas)
    
    # Calcular tasa de repetición (bigramas que aparecen 2+ veces)
    bigramas_repetidos = sum(1 for count in bigram_counts.values() if count > 1)
    repetition_rate_bigrams = bigramas_repetidos / len(bigramas) if len(bigramas) > 0 else 0
    
    trigramas_repetidos = sum(1 for count in trigram_counts.values() if count > 1)
    repetition_rate_trigrams = trigramas_repetidos / len(trigramas) if len(trigramas) > 0 else 0
    
    return {
        'type_token_ratio': round(ttr, 3),
        'repetition_rate_bigrams': round(repetition_rate_bigrams, 3),
        'repetition_rate_trigrams': round(repetition_rate_trigrams, 3),
        'palabras_unicas': len(palabras_unicas),
        'total_palabras': len(palabras)
    }

def validar_calidad_pls(pls_generado, min_palabras=140, max_palabras=210):
    """
    Valida la calidad de un PLS generado.
    
    Args:
        pls_generado: Texto del PLS
        min_palabras: Mínimo de palabras permitidas
        max_palabras: Máximo de palabras permitidas
    
    Returns:
        dict: Diccionario con validación y problemas detectados
    """
    palabras = pls_generado.split()
    num_palabras = len(palabras)
    
    problemas = []
    warnings = []
    
    # 1. Validar longitud
    if num_palabras < min_palabras:
        problemas.append(f"Muy corto: {num_palabras} palabras (mínimo: {min_palabras})")
    elif num_palabras > max_palabras:
        problemas.append(f"Muy largo: {num_palabras} palabras (máximo: {max_palabras})")
    
    # 2. Detectar frases exactamente repetidas
    oraciones = pls_generado.split('.')
    oraciones_limpias = [s.strip().lower() for s in oraciones if s.strip()]
    
    oraciones_vistas = set()
    for oracion in oraciones_limpias:
        if oracion in oraciones_vistas and len(oracion) > 10:  # Ignorar oraciones muy cortas
            problemas.append(f"Oración repetida: '{oracion[:50]}...'")
        oraciones_vistas.add(oracion)
    
    # 3. Validar diversidad léxica
    diversidad = calcular_diversidad_lexica(pls_generado)
    
    # TTR muy bajo indica mucha repetición de palabras
    if diversidad['type_token_ratio'] < 0.4:
        warnings.append(f"Baja diversidad léxica (TTR: {diversidad['type_token_ratio']:.2f})")
    
    # Muchos trigramas repetidos indica frases similares
    if diversidad['repetition_rate_trigrams'] > 0.3:
        warnings.append(f"Muchas frases similares (repetición trigramas: {diversidad['repetition_rate_trigrams']:.2f})")
    
    # 4. Validación pasó?
    es_valido = len(problemas) == 0
    calidad = "excelente" if es_valido and len(warnings) == 0 else \
              "buena" if es_valido and len(warnings) <= 1 else \
              "aceptable" if es_valido else "pobre"
    
    return {
        'es_valido': es_valido,
        'calidad': calidad,
        'num_palabras': num_palabras,
        'problemas': problemas,
        'warnings': warnings,
        **diversidad
    }

def calcular_metricas(texto_original, pls_generado):
    """Calcula todas las métricas para un par texto-PLS (ahora con validación)"""
    
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
    
    # Validación de calidad y diversidad léxica
    validacion = validar_calidad_pls(pls_generado)
    
    return {
        'longitud_original': longitud_original,
        'longitud_pls': longitud_pls,
        'tokens_input': tokens_input,
        'tokens_output': tokens_output,
        'costo_estimado': costo_total,
        'flesch_reading_ease': flesch_reading_ease,
        'flesch_kincaid_grade': flesch_kincaid_grade,
        'ratio_compresion': ratio_compresion,
        # Nuevas métricas de validación y diversidad
        'es_valido': validacion['es_valido'],
        'calidad': validacion['calidad'],
        'type_token_ratio': validacion['type_token_ratio'],
        'repetition_rate_bigrams': validacion['repetition_rate_bigrams'],
        'repetition_rate_trigrams': validacion['repetition_rate_trigrams'],
        'problemas_detectados': '; '.join(validacion['problemas']) if validacion['problemas'] else '',
        'warnings': '; '.join(validacion['warnings']) if validacion['warnings'] else ''
    }

# =============================================================================
# GENERACIÓN ASYNC CON OPENAI
# =============================================================================

def contar_tokens(texto, modelo="gpt-4o-mini"):
    """Cuenta tokens en un texto usando tiktoken o estimación"""
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(modelo)
            return len(encoding.encode(texto))
        except:
            pass
    # Estimación aproximada: ~4 caracteres por token para inglés
    return len(texto) // 4

class AsyncTokenLimiter:
    """Control de rate limiting para tokens por minuto"""
    
    def __init__(self, max_tokens_per_minute):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.token_usage = []  # Lista de (timestamp, tokens_usados)
        self.lock = asyncio.Lock()
        self.safety_margin = 0.9  # Usar solo 90% del límite para seguridad
    
    async def acquire(self, estimated_tokens):
        """Espera si es necesario para respetar límite de tokens"""
        async with self.lock:
            now = time.monotonic()
            
            # Limpiar tokens antiguos (>1 minuto)
            cutoff_time = now - 60.0
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]
            
            # Calcular tokens usados en el último minuto
            tokens_used = sum(tokens for _, tokens in self.token_usage)
            max_allowed = int(self.max_tokens_per_minute * self.safety_margin)
            
            # Si agregar estos tokens excedería el límite, esperar
            if tokens_used + estimated_tokens > max_allowed:
                if self.token_usage:
                    # Esperar hasta que el token más antiguo tenga >1 minuto
                    oldest_time = min(t for t, _ in self.token_usage)
                    wait_time = 60.0 - (now - oldest_time) + 0.1
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        now = time.monotonic()
                        # Limpiar de nuevo después de esperar
                        cutoff_time = now - 60.0
                        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]
                else:
                    # Si no hay tokens registrados pero aún excederíamos, esperar un poco
                    await asyncio.sleep(0.1)
    
    async def record_usage(self, tokens_used):
        """Registra tokens usados después de un request"""
        async with self.lock:
            now = time.monotonic()
            self.token_usage.append((now, tokens_used))
            # Limpiar tokens antiguos
            cutoff_time = now - 60.0
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]

class AsyncRateLimiter:
    """Control de rate limiting para requests asíncronos (optimizado y seguro)"""
    
    def __init__(self, max_per_minute):
        self.max_per_minute = max_per_minute
        self.requests = []
        self.lock = asyncio.Lock()
        self.min_interval = 60.0 / max_per_minute
        self.last_request_time = 0.0
    
    async def acquire(self):
        """Espera si es necesario para respetar rate limit (distribución uniforme)"""
        async with self.lock:
            now = time.monotonic()
            
            # Limpiar requests antiguos (>1 minuto)
            cutoff_time = now - 60.0
            self.requests = [r for r in self.requests if r > cutoff_time]
            
            # Si ya hay el máximo de requests en el último minuto, esperar
            if len(self.requests) >= self.max_per_minute:
                oldest = self.requests[0]
                wait_time = 60.0 - (now - oldest) + 0.1
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    now = time.monotonic()
                    cutoff_time = now - 60.0
                    self.requests = [r for r in self.requests if r > cutoff_time]
            
            # Distribución uniforme: asegurar intervalo mínimo entre requests
            if self.last_request_time > 0:
                time_since_last = now - self.last_request_time
                if time_since_last < self.min_interval:
                    wait_time = self.min_interval - time_since_last
                    await asyncio.sleep(wait_time)
                    now = time.monotonic()
                    cutoff_time = now - 60.0
                    self.requests = [r for r in self.requests if r > cutoff_time]
            
            # Registrar este request
            self.requests.append(now)
            self.last_request_time = now

rate_limiter = AsyncRateLimiter(MAX_REQUESTS_PER_MINUTE)
token_limiter = AsyncTokenLimiter(MAX_TOKENS_PER_MINUTE)

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
    
    # Estimar tokens de entrada (system prompt + user prompt)
    user_prompt = USER_PROMPT_TEMPLATE.format(texto_original=texto_original)
    input_tokens = contar_tokens(SYSTEM_PROMPT) + contar_tokens(user_prompt)
    # Estimar tokens de salida (máximo permitido)
    estimated_output_tokens = MAX_TOKENS
    total_estimated_tokens = input_tokens + estimated_output_tokens
    
    for intento in range(max_reintentos):
        try:
            # Respetar rate limits (requests y tokens)
            await rate_limiter.acquire()
            await token_limiter.acquire(total_estimated_tokens)
            
            # Preparar request
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "frequency_penalty": FREQUENCY_PENALTY,
                "presence_penalty": PRESENCE_PENALTY
            }
            
            # Hacer request
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=45)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    pls = data['choices'][0]['message']['content'].strip()
                    
                    # Registrar tokens usados (si están disponibles en la respuesta)
                    if 'usage' in data:
                        total_tokens_used = data['usage'].get('total_tokens', total_estimated_tokens)
                    else:
                        # Estimar tokens de salida reales
                        output_tokens = contar_tokens(pls)
                        total_tokens_used = input_tokens + output_tokens
                    
                    await token_limiter.record_usage(total_tokens_used)
                    
                    return pls
                else:
                    error_text = await response.text()
                    logger.warning(f"Error {response.status} en intento {intento + 1}: {error_text}")
                    
                    # Registrar tokens estimados incluso si falla (para mantener el tracking)
                    await token_limiter.record_usage(total_estimated_tokens)
                    
                    if response.status == 429:  # Rate limit
                        wait_time = 2 ** intento  # Exponential backoff
                        await asyncio.sleep(wait_time)
                    elif response.status >= 500:  # Server error
                        await asyncio.sleep(2 ** intento)
                    else:
                        return None
        
        except asyncio.TimeoutError:
            logger.warning(f"Timeout en intento {intento + 1}")
            await token_limiter.record_usage(total_estimated_tokens)
            await asyncio.sleep(2 ** intento)
        
        except Exception as e:
            logger.error(f"Error inesperado en intento {intento + 1}: {e}")
            await token_limiter.record_usage(total_estimated_tokens)
            await asyncio.sleep(2 ** intento)
    
    logger.error(f"Falló después de {max_reintentos} intentos")
    return None

# =============================================================================
# PROCESAMIENTO PARALELO OPTIMIZADO
# =============================================================================

async def procesar_item_async(session, semaphore, idx, texto_original, pbar):
    """
    Procesa un solo item con control de concurrencia.
    
    Args:
        session: aiohttp ClientSession
        semaphore: Semáforo para controlar concurrencia
        idx: Índice del item
        texto_original: Texto médico original
        pbar: Barra de progreso
    
    Returns:
        dict: Resultado con métricas o None si falla
    """
    async with semaphore:
        pls_generado = await generar_pls_async(session, texto_original)
        pbar.update(1)
        
        if pls_generado:
            # Calcular métricas
            metricas = calcular_metricas(texto_original, pls_generado)
            
            # Guardar resultado
            resultado = {
                'id': idx,
                'texto_original': texto_original,
                'pls_generado': pls_generado,
                **metricas,
                'timestamp': datetime.now().isoformat()
            }
            return resultado
        else:
            logger.error(f"Falló la generación para ID {idx}")
            return None

async def procesar_todos_async(textos_df, output_file, checkpoint_interval=500):
    """
    Procesa todos los textos de forma paralela con control de concurrencia.
    
    Args:
        textos_df: DataFrame con los textos originales
        output_file: Archivo de salida para resultados
        checkpoint_interval: Intervalo para guardar checkpoints
    """
    
    # Semáforo para controlar concurrencia
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    # Lista para acumular resultados antes de guardar
    resultados_buffer = []
    total_procesados = 0
    
    # Crear sesión aiohttp compartida
    async with aiohttp.ClientSession() as session:
        
        # Crear todas las tareas
        pbar = tqdm(total=len(textos_df), desc="Generando PLS", unit="PLS")
        
        tasks = []
        for idx, row in textos_df.iterrows():
            task = procesar_item_async(
                session, 
                semaphore, 
                idx, 
                row['texto_original'],
                pbar
            )
            tasks.append(task)
        
        # Procesar todas las tareas en paralelo, pero en chunks para guardar periódicamente
        chunk_size = checkpoint_interval
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i:i + chunk_size]
            resultados_chunk = await asyncio.gather(*chunk, return_exceptions=True)
            
            # Filtrar resultados válidos (None y excepciones se ignoran)
            resultados_validos = [
                r for r in resultados_chunk 
                if r is not None and not isinstance(r, Exception)
            ]
            
            resultados_buffer.extend(resultados_validos)
            total_procesados += len(resultados_validos)
            
            # Guardar checkpoint periódicamente
            if len(resultados_buffer) >= checkpoint_interval:
                guardar_resultados(resultados_buffer, output_file, append=True)
                resultados_buffer = []
                
                # Guardar checkpoint
                checkpoint_file = CHECKPOINT_DIR / f"checkpoint_{total_procesados}.csv"
                if output_file.exists():
                    df_temp = pd.read_csv(output_file)
                    df_temp.to_csv(checkpoint_file, index=False)
                    logger.info(f"Checkpoint guardado: {checkpoint_file}")
        
        pbar.close()
    
    # Guardar resultados restantes
    if resultados_buffer:
        guardar_resultados(resultados_buffer, output_file, append=True)
    
    return total_procesados

def guardar_resultados(resultados, output_file, append=False):
    """Guarda resultados en CSV de forma eficiente"""
    if not resultados:
        return
    
    df_resultados = pd.DataFrame(resultados)
    
    if append and output_file.exists():
        df_existente = pd.read_csv(output_file)
        df_final = pd.concat([df_existente, df_resultados], ignore_index=True)
        df_final.to_csv(output_file, index=False)
    else:
        df_resultados.to_csv(output_file, index=False)
    
    logger.info(f"Guardados {len(resultados)} resultados en {output_file}")

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
    print("1. Modo PRUEBA: Generar 50 PLS (~ 1-2 minutos)")
    print("2. Modo PRODUCCIÓN: Generar 10,000 PLS (~ 2-3 horas)")
    print("3. Modo GRANDE: Generar 20,000 PLS (~ 4-5 horas)")
    print("4. Modo PERSONALIZADO: Especificar cantidad")
    print("="*70)
    
    # Permitir argumentos de línea de comandos
    if len(sys.argv) > 1:
        opcion = sys.argv[1].strip()
        auto_confirm = True
        print(f"\n Modo automático: Opción {opcion} seleccionada")
    else:
        opcion = input("\nSelecciona una opción (1-4): ").strip()
        auto_confirm = False
    
    if opcion == "1":
        n = 50
        output_file = OUTPUT_DIR / "pls_prueba_50_v8.1.csv"
        print(f"\n Modo PRUEBA: Generando {n} PLS...")
    elif opcion == "2":
        n = 10000
        output_file = OUTPUT_DIR / "pls_produccion_10k_v8.1.csv"
        print(f"\n Modo PRODUCCIÓN: Generando {n} PLS...")
    elif opcion == "3":
        n = 20000
        output_file = OUTPUT_DIR / "pls_produccion_20k_v8.1.csv"
        print(f"\n Modo GRANDE: Generando {n} PLS...")
    elif opcion == "4":
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
    
    # Estimación de tiempo mejorada (considerando paralelismo real)
    tiempo_estimado_min = (n / MAX_REQUESTS_PER_MINUTE) * 1.2  # Factor optimizado
    
    print(f"\n ESTIMACIÓN DE COSTOS:")
    print(f"   Costo por PLS: ${costo_por_pls:.6f}")
    print(f"   Costo total estimado: ${costo_total_estimado:.2f}")
    print(f"   Tiempo estimado: {tiempo_estimado_min:.1f} minutos ({tiempo_estimado_min/60:.1f} horas)")
    print(f"   Procesamiento paralelo: {CONCURRENT_REQUESTS} requests simultáneos")
    print(f"   Rate limit: {MAX_REQUESTS_PER_MINUTE} requests/minuto")
    
    if auto_confirm:
        print("\n Continuando automáticamente...")
        confirmar = 's'
    else:
        confirmar = input("\n¿Continuar? (s/n): ").strip().lower()
    
    if confirmar != 's':
        print(" Operación cancelada")
        return
    
    # PROCESAR
    print(f"\n Iniciando generación con {CONCURRENT_REQUESTS} workers paralelos...")
    print(f"   Procesamiento optimizado: todas las tareas en paralelo")
    print(f"   Checkpoints cada 500 PLS generados")
    inicio_total = datetime.now()
    
    # Procesar todos los textos en paralelo
    total_procesados = await procesar_todos_async(muestra, output_file, checkpoint_interval=500)
    
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
        
        print(f"\n MÉTRICAS DE CALIDAD (NUEVAS):")
        if 'es_valido' in df_final.columns:
            tasa_validez = df_final['es_valido'].mean() * 100
            print(f"   Tasa de validez: {tasa_validez:.1f}% (cumplen longitud 140-210 palabras)")
        if 'type_token_ratio' in df_final.columns:
            print(f"   Type-Token Ratio: {df_final['type_token_ratio'].mean():.3f} (diversidad léxica, >0.5 es bueno)")
        if 'repetition_rate_bigrams' in df_final.columns:
            print(f"   Repetición bigramas: {df_final['repetition_rate_bigrams'].mean():.3f} (<0.2 es bueno)")
        if 'repetition_rate_trigrams' in df_final.columns:
            print(f"   Repetición trigramas: {df_final['repetition_rate_trigrams'].mean():.3f} (<0.1 es bueno)")
        if 'calidad' in df_final.columns:
            calidad_counts = df_final['calidad'].value_counts()
            print(f"   Distribución de calidad:")
            for calidad, count in calidad_counts.items():
                print(f"      {calidad}: {count} ({count/len(df_final)*100:.1f}%)")
        
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
