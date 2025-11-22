"""
Configuración centralizada para entrenamiento y evaluación del modelo T5-BASE.

Este módulo centraliza TODAS las configuraciones para:
- Entrenamiento del modelo (train_t5_base_pls_a100.py)
- Evaluación del modelo (evaluate_t5_base_pls_a100.py)

IMPORTANTE:
- Las CONFIGURACIONES están aquí en config_model.py (versionadas en Git)
- Las CREDENCIALES y rutas específicas van en variables de entorno o se configuran en runtime
- Las configuraciones pueden ser sobrescritas por variables de entorno si es necesario

Uso:
    from config_model import ModelConfig
    
    config = ModelConfig()
    print(f"Model name: {config.MODEL_NAME}")
    print(f"Task prefix: {config.TASK_PREFIX}")
"""

import os
from pathlib import Path
from typing import Optional
import logging

# Configurar logger
logger = logging.getLogger(__name__)


class ModelConfig:
    """
    Configuración centralizada para entrenamiento y evaluación del modelo T5-BASE.
    
    Todas las configuraciones tienen valores por defecto razonables.
    Pueden ser sobrescritas por variables de entorno si es necesario.
    """
    
    # ========================================================================
    # Model Configuration
    # ========================================================================
    MODEL_NAME: str = os.getenv("MODEL_NAME", "t5-base")
    """Nombre del modelo base de Hugging Face"""
    
    TASK_PREFIX: str = os.getenv("TASK_PREFIX", "simplify medical text: ")
    """Prefijo agregado al texto antes de enviarlo al modelo T5"""
    
    # ========================================================================
    # Tokenization Configuration
    # ========================================================================
    MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "512"))
    """Longitud máxima de tokens para input"""
    
    MAX_TARGET_LENGTH: int = int(os.getenv("MAX_TARGET_LENGTH", "256"))
    """Longitud máxima de tokens para output (target)"""
    
    # ========================================================================
    # Chunking Configuration
    # ========================================================================
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
    """Tamaño de chunk para división de texto (en tokens)"""
    
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    """Solapamiento entre chunks (en tokens)"""
    
    SEPARATORS: list = ["\n\n", "\n", ". ", " "]
    """Separadores de texto para chunking, en orden de preferencia"""
    
    # ========================================================================
    # Training Configuration
    # ========================================================================
    NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "3"))
    """Número de épocas de entrenamiento"""
    
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "3e-4"))
    """Learning rate para optimizador"""
    
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "16"))
    """Batch size por dispositivo (optimizado para A100)"""
    
    GRAD_ACCUM_STEPS: int = int(os.getenv("GRAD_ACCUM_STEPS", "2"))
    """Pasos de acumulación de gradiente (effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS)"""
    
    WARMUP_STEPS: int = int(os.getenv("WARMUP_STEPS", "500"))
    """Pasos de warmup para learning rate scheduler"""
    
    WEIGHT_DECAY: float = float(os.getenv("WEIGHT_DECAY", "0.01"))
    """Weight decay para regularización"""
    
    # ========================================================================
    # Evaluation Configuration
    # ========================================================================
    EVAL_STEPS: int = int(os.getenv("EVAL_STEPS", "200"))
    """Pasos entre evaluaciones durante entrenamiento"""
    
    SAVE_STEPS: int = int(os.getenv("SAVE_STEPS", "200"))
    """Pasos entre guardado de checkpoints"""
    
    SAVE_TOTAL_LIMIT: int = int(os.getenv("SAVE_TOTAL_LIMIT", "3"))
    """Número máximo de checkpoints a mantener"""
    
    # ========================================================================
    # Generation Configuration
    # ========================================================================
    NUM_BEAMS: int = int(os.getenv("NUM_BEAMS", "4"))
    """Número de beams para beam search durante generación"""
    
    # ========================================================================
    # Data Configuration
    # ========================================================================
    SEED: int = int(os.getenv("SEED", "42"))
    """Semilla para reproducibilidad (CRÍTICO: debe ser la misma en train y eval)"""
    
    TRAIN_RATIO: float = float(os.getenv("TRAIN_RATIO", "0.8"))
    """Proporción de datos para entrenamiento"""
    
    VAL_RATIO: float = float(os.getenv("VAL_RATIO", "0.1"))
    """Proporción de datos para validación (test = 1 - TRAIN_RATIO - VAL_RATIO)"""
    
    # ========================================================================
    # Paths Configuration (se configuran dinámicamente)
    # ========================================================================
    CSV_PATH: Optional[str] = os.getenv("CSV_PATH", None)
    """Ruta al archivo CSV con los datos (se detecta automáticamente si es None)"""
    
    DRIVE_BASE: str = os.getenv("DRIVE_BASE", "/content/drive/MyDrive/PLS_Project")
    """Ruta base de Google Drive (solo para Colab)"""
    
    MODEL_DIR: Optional[str] = os.getenv("MODEL_DIR", None)
    """Ruta donde se guarda/carga el modelo entrenado"""
    
    RESULTS_DIR: Optional[str] = os.getenv("RESULTS_DIR", None)
    """Ruta donde se guardan los resultados"""
    
    PLOTS_DIR: Optional[str] = os.getenv("PLOTS_DIR", None)
    """Ruta donde se guardan los gráficos"""
    
    # ========================================================================
    # Evaluation-specific Configuration
    # ========================================================================
    BATCH_SIZE_EVAL: int = int(os.getenv("BATCH_SIZE_EVAL", "32"))
    """Batch size para evaluación (puede ser mayor que training batch)"""
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"ModelConfig("
            f"MODEL_NAME={self.MODEL_NAME}, "
            f"TASK_PREFIX={self.TASK_PREFIX}, "
            f"SEED={self.SEED}, "
            f"NUM_EPOCHS={self.NUM_EPOCHS}, "
            f"BATCH_SIZE={self.BATCH_SIZE}"
            f")"
        )
    
    def validate(self) -> bool:
        """
        Valida la configuración.
        
        Returns:
            True si la configuración es válida, False en caso contrario
        """
        errors = []
        
        # Validar longitudes
        if self.MAX_INPUT_LENGTH < 1 or self.MAX_INPUT_LENGTH > 1024:
            errors.append(f"Invalid MAX_INPUT_LENGTH: {self.MAX_INPUT_LENGTH}. Must be between 1 and 1024")
        
        if self.MAX_TARGET_LENGTH < 1 or self.MAX_TARGET_LENGTH > 512:
            errors.append(f"Invalid MAX_TARGET_LENGTH: {self.MAX_TARGET_LENGTH}. Must be between 1 and 512")
        
        # Validar chunking
        if self.CHUNK_SIZE > self.MAX_INPUT_LENGTH:
            errors.append(f"CHUNK_SIZE ({self.CHUNK_SIZE}) cannot be greater than MAX_INPUT_LENGTH ({self.MAX_INPUT_LENGTH})")
        
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            errors.append(f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({self.CHUNK_SIZE})")
        
        # Validar training
        if self.NUM_EPOCHS < 1:
            errors.append(f"Invalid NUM_EPOCHS: {self.NUM_EPOCHS}. Must be >= 1")
        
        if self.BATCH_SIZE < 1:
            errors.append(f"Invalid BATCH_SIZE: {self.BATCH_SIZE}. Must be >= 1")
        
        if self.LEARNING_RATE <= 0:
            errors.append(f"Invalid LEARNING_RATE: {self.LEARNING_RATE}. Must be > 0")
        
        # Validar ratios
        if not (0 < self.TRAIN_RATIO < 1):
            errors.append(f"Invalid TRAIN_RATIO: {self.TRAIN_RATIO}. Must be between 0 and 1")
        
        if not (0 < self.VAL_RATIO < 1):
            errors.append(f"Invalid VAL_RATIO: {self.VAL_RATIO}. Must be between 0 and 1")
        
        if self.TRAIN_RATIO + self.VAL_RATIO >= 1:
            errors.append(f"TRAIN_RATIO + VAL_RATIO ({self.TRAIN_RATIO + self.VAL_RATIO}) must be < 1")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        return True


# Instancia global de configuración
config = ModelConfig()

# Exportar configuraciones para uso directo (compatibilidad con código existente)
TASK_PREFIX: str = config.TASK_PREFIX
MAX_INPUT_LENGTH: int = config.MAX_INPUT_LENGTH
MAX_TARGET_LENGTH: int = config.MAX_TARGET_LENGTH
CHUNK_SIZE: int = config.CHUNK_SIZE
CHUNK_OVERLAP: int = config.CHUNK_OVERLAP
SEPARATORS: list = config.SEPARATORS
NUM_BEAMS: int = config.NUM_BEAMS
SEED: int = config.SEED


def get_config() -> ModelConfig:
    """
    Obtiene la instancia de configuración.
    
    Returns:
        ModelConfig: Instancia de configuración
    """
    return config


def setup_paths(drive_base: Optional[str] = None, model_dir: Optional[str] = None):
    """
    Configura las rutas de forma dinámica.
    
    Args:
        drive_base: Ruta base de Google Drive (solo para Colab)
        model_dir: Ruta donde se guarda/carga el modelo
    """
    if drive_base:
        config.DRIVE_BASE = drive_base
    
    if model_dir:
        config.MODEL_DIR = model_dir
    elif not config.MODEL_DIR:
        # Configurar automáticamente basado en DRIVE_BASE
        config.MODEL_DIR = f"{config.DRIVE_BASE}/models/t5_pls/final"
    
    if not config.RESULTS_DIR:
        config.RESULTS_DIR = f"{config.DRIVE_BASE}/results"
    
    if not config.PLOTS_DIR:
        config.PLOTS_DIR = f"{config.RESULTS_DIR}/plots"
    
    # Crear directorios si no existen
    for path in [config.MODEL_DIR, config.RESULTS_DIR, config.PLOTS_DIR]:
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)

