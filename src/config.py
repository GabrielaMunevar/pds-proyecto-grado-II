"""
Configuración centralizada del proyecto PLS Biomédico.

Este módulo centraliza todas las constantes y configuraciones del proyecto,
especialmente el prompt estándar para garantizar consistencia en todos los modelos.
"""

# ============================================================================
# PROMPT ESTÁNDAR PARA SIMPLIFICACIÓN
# ============================================================================

# Prompt optimizado para simplificación de texto biomédico
# Este prompt es usado consistentemente en todo el proyecto para:
# - Entrenamiento de modelos
# - Evaluación
# - Generación de PLS
# - Comparación de modelos
#
# Características del prompt:
# - Específico para dominio biomédico
# - Indica claramente el objetivo (plain language)
# - Formato compatible con T5
# - Balanceado entre descriptivo y conciso
STANDARD_PROMPT = "simplify medical text into plain language: "

# Variantes del prompt (para experimentación futura)
PROMPT_VARIANTS = {
    'standard': "simplify medical text into plain language: ",
    'detailed': "translate this biomedical text into plain language for patients: ",
    'concise': "simplify: ",
    'instruction': "Rewrite this medical text in plain language, avoiding jargon: "
}

# Prompt actualmente en uso (puede cambiarse para experimentación)
CURRENT_PROMPT = STANDARD_PROMPT

# ============================================================================
# FUNCIÓN HELPER PARA APLICAR PROMPT
# ============================================================================

def apply_prompt(text: str, prompt: str = None) -> str:
    """
    Aplica el prompt estándar a un texto.
    
    Args:
        text: Texto técnico a simplificar
        prompt: Prompt a usar. Si es None, usa CURRENT_PROMPT
    
    Returns:
        Texto con el prompt aplicado
    """
    if prompt is None:
        prompt = CURRENT_PROMPT
    
    return prompt + text

def get_prompt() -> str:
    """
    Obtiene el prompt estándar actual.
    
    Returns:
        String con el prompt estándar
    """
    return CURRENT_PROMPT

def set_prompt(prompt: str):
    """
    Cambia el prompt estándar (útil para experimentación).
    
    Args:
        prompt: Nuevo prompt a usar
    """
    global CURRENT_PROMPT
    CURRENT_PROMPT = prompt

# ============================================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================================

# Modelos base disponibles
MODEL_CONFIGS = {
    't5-base': {
        'model_name': 't5-base',
        'max_context': 512,
        'max_length_source': 400,  # Dejando margen para el prompt
        'max_length_target': 256
    },
    't5-small': {
        'model_name': 't5-small',
        'max_context': 512,
        'max_length_source': 400,
        'max_length_target': 256
    },
    'biomedgpt': {
        'model_name': 'microsoft/biogpt',
        'max_context': 1024,
        'max_length_source': 900,
        'max_length_target': 256
    },
    'llama-2-1b': {
        'model_name': 'meta-llama/Llama-2-1b-hf',
        'max_context': 2048,
        'max_length_source': 1900,
        'max_length_target': 256
    },
    'qwen-2.5-0.5b': {
        'model_name': 'Qwen/Qwen2.5-0.5B',
        'max_context': 32000,
        'max_length_source': 31900,
        'max_length_target': 256
    }
}

# Modelo actualmente en uso
CURRENT_MODEL = 't5-base'

def get_model_config(model_name: str = None):
    """
    Obtiene la configuración de un modelo.
    
    Args:
        model_name: Nombre del modelo. Si es None, usa CURRENT_MODEL
    
    Returns:
        Diccionario con configuración del modelo
    """
    if model_name is None:
        model_name = CURRENT_MODEL
    
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['t5-base'])

# ============================================================================
# CONFIGURACIÓN DE CHUNKING
# ============================================================================

# Configuración por defecto para chunking
CHUNKING_CONFIG = {
    'max_tokens': 400,  # Máximo de tokens por chunk
    'overlap': 50,      # Tokens de overlap entre chunks
    'min_chunk_size': 50  # Tamaño mínimo de chunk en caracteres
}

# ============================================================================
# CONFIGURACIÓN DE EVALUACIÓN
# ============================================================================

# Métricas objetivo
EVALUATION_TARGETS = {
    'rouge_l': 0.35,
    'bleu_4': 0.30,
    'sari': 0.40,
    'bertscore_f1': 0.85,
    'flesch_reading_ease_min': 60
}

# ============================================================================
# INFORMACIÓN DEL PROYECTO
# ============================================================================

PROJECT_INFO = {
    'name': 'PLS Biomédico',
    'description': 'Sistema de generación de Plain Language Summaries para textos biomédicos',
    'version': '1.0.0',
    'domain': 'biomedical_text_simplification',
    'target_audience': 'patients_and_general_public'
}

