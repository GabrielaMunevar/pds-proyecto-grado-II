"""
Configuración específica para el modelo T5.

Este módulo centraliza la configuración específica para T5 y proporciona
acceso a la configuración general del proyecto.
"""

from pathlib import Path
import sys

# Importar configuración general del proyecto
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    STANDARD_PROMPT,
    CURRENT_PROMPT,
    PROMPT_VARIANTS,
    apply_prompt,
    get_prompt,
    set_prompt,
    MODEL_CONFIGS,
    CURRENT_MODEL,
    get_model_config,
    CHUNKING_CONFIG,
    EVALUATION_TARGETS,
    PROJECT_INFO
)

# ============================================================================
# CONFIGURACIÓN ESPECÍFICA DE T5
# ============================================================================

# Configuración por defecto para T5
T5_DEFAULT_CONFIG = {
    'model_name': 't5-base',
    'max_context': 512,
    'max_length_source': 400,  
    'max_length_target': 256,
    'chunking': {
        'max_tokens': 400,
        'overlap': 50,
        'min_chunk_size': 50
    },
    'generation': {
        'max_length': 256,
        'num_beams': 4,
        'early_stopping': True,
        'no_repeat_ngram_size': 3
    },
    'training': {
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'gradient_accumulation_steps': 2,
        'fp16': True,  
        'save_total_limit': 3,
        'eval_strategy': 'epoch',
        'save_strategy': 'epoch'
    }
}

# Configuraciones para diferentes tamaños de T5
T5_MODEL_CONFIGS = {
    't5-small': {
        **T5_DEFAULT_CONFIG,
        'model_name': 't5-small',
        'training': {
            **T5_DEFAULT_CONFIG['training'],
            'batch_size': 16,  
        }
    },
    't5-base': {
        **T5_DEFAULT_CONFIG,
        'model_name': 't5-base',
        'training': {
            **T5_DEFAULT_CONFIG['training'],
            'batch_size': 8,
        }
    },
    't5-large': {
        **T5_DEFAULT_CONFIG,
        'model_name': 't5-large',
        'training': {
            **T5_DEFAULT_CONFIG['training'],
            'batch_size': 2, 
            'gradient_accumulation_steps': 16,  
            'eval_batch_size': 1,  
            'eval_accumulation_steps': 8,  
        }
    }
}

# Modelo T5 actualmente en uso
CURRENT_T5_MODEL = 't5-base'

def get_t5_config(model_name: str = None) -> dict:
    """
    Obtiene la configuración para un modelo T5 específico.
    
    Args:
        model_name: Nombre del modelo T5. Si es None, usa CURRENT_T5_MODEL
    
    Returns:
        Diccionario con configuración del modelo T5
    """
    if model_name is None:
        model_name = CURRENT_T5_MODEL
    
    return T5_MODEL_CONFIGS.get(model_name, T5_DEFAULT_CONFIG)

def get_t5_chunking_config() -> dict:
    """
    Obtiene la configuración de chunking para T5.
    
    Returns:
        Diccionario con configuración de chunking
    """
    return get_t5_config()['chunking']

def get_t5_generation_config() -> dict:
    """
    Obtiene la configuración de generación para T5.
    
    Returns:
        Diccionario con configuración de generación
    """
    return get_t5_config()['generation']

def get_t5_training_config() -> dict:
    """
    Obtiene la configuración de entrenamiento para T5.
    
    Returns:
        Diccionario con configuración de entrenamiento
    """
    return get_t5_config()['training']

# Exportar funciones y constantes principales
__all__ = [
    # Configuración general (re-exportada)
    'STANDARD_PROMPT',
    'CURRENT_PROMPT',
    'PROMPT_VARIANTS',
    'apply_prompt',
    'get_prompt',
    'set_prompt',
    'MODEL_CONFIGS',
    'CURRENT_MODEL',
    'get_model_config',
    'CHUNKING_CONFIG',
    'EVALUATION_TARGETS',
    'PROJECT_INFO',
    # Configuración específica de T5
    'T5_DEFAULT_CONFIG',
    'T5_MODEL_CONFIGS',
    'CURRENT_T5_MODEL',
    'get_t5_config',
    'get_t5_chunking_config',
    'get_t5_generation_config',
    'get_t5_training_config'
]

