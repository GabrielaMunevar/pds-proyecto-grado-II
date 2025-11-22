"""
Configuración centralizada para la API de PLS.

Este módulo centraliza TODAS las configuraciones de la API:
- Variables de configuración: definidas aquí con valores por defecto (versionadas en Git)
- Variables de entorno: solo para credenciales y secretos (se cargan desde .env)

IMPORTANTE:
- Las CONFIGURACIONES (valores por defecto, constantes) están aquí en config.py
- Las CREDENCIALES (AWS keys, API keys) van en .env (NO se versionan)
- Las variables de entorno pueden sobrescribir configuraciones si es necesario

Uso:
    from config import settings, TASK_PREFIX, DEFAULT_MAX_LENGTH
    
    logger.info(f"Using device: {settings.DEVICE}")
    logger.info(f"Task prefix: {TASK_PREFIX}")
"""

import os
from pathlib import Path
from typing import Optional
import logging

# Configurar logger
logger = logging.getLogger(__name__)


class APISettings:
    """
    Configuración centralizada de la API.
    
    Todas las configuraciones se cargan desde variables de entorno,
    con valores por defecto razonables.
    """
    
    # ========================================================================
    # Logging Configuration
    # ========================================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    """Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL"""
    
    # ========================================================================
    # Model Configuration
    # ========================================================================
    DEVICE: Optional[str] = os.getenv("DEVICE", None)
    """Device for model inference: 'cuda' (GPU) or 'cpu'. 
    If None, will auto-detect (prefers CUDA if available)"""
    
    MODEL_PATH: Optional[str] = os.getenv("MODEL_PATH", None)
    """Path to T5 model directory. If None, will auto-detect from common locations"""
    
    CLASSIFIER_PATH: Optional[str] = os.getenv("CLASSIFIER_PATH", None)
    """Path to classifier model directory. If None, will auto-detect"""
    
    # ========================================================================
    # DVC/S3 Configuration
    # ========================================================================
    DVC_S3_BUCKET: Optional[str] = os.getenv("DVC_S3_BUCKET", None)
    """S3 bucket where DVC stores models. Required if using DVC in production"""
    
    DVC_S3_PREFIX: str = os.getenv("DVC_S3_PREFIX", "dvcstore")
    """S3 prefix/path for DVC store"""
    
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    """AWS region for S3 access"""
    
    # ========================================================================
    # Alternative: Direct S3 Model Storage
    # ========================================================================
    S3_MODELS_BUCKET: Optional[str] = os.getenv("S3_MODELS_BUCKET", None)
    """S3 bucket for direct model storage (alternative to DVC)"""
    
    # ========================================================================
    # API Configuration
    # ========================================================================
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    """API host (0.0.0.0 for Docker, 127.0.0.1 for local)"""
    
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    """API port"""
    
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    """CORS allowed origins (comma-separated)"""
    
    # ========================================================================
    # Generation Parameters (defaults)
    # ========================================================================
    # These are configuration values, not credentials
    # Can be overridden by environment variables for flexibility
    DEFAULT_MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "256"))
    """Default maximum output length in tokens"""
    
    DEFAULT_NUM_BEAMS: int = int(os.getenv("NUM_BEAMS", "4"))
    """Default number of beams for beam search"""
    
    # ========================================================================
    # Chunking Configuration
    # ========================================================================
    # These are configuration values, not credentials
    # Can be overridden by environment variables for flexibility
    MAX_INPUT_LENGTH: int = int(os.getenv("MAX_INPUT_LENGTH", "512"))
    """Maximum input tokens per chunk"""
    
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
    """Chunk size for text splitting in tokens"""
    
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    """Overlap between chunks in tokens"""
    
    # ========================================================================
    # Task Prefix (Prompt Configuration)
    # ========================================================================
    TASK_PREFIX: str = "simplify medical text into plain language: "
    """Prefix added to input text before sending to T5 model"""
    
    SEPARATORS: list = ["\n\n", "\n", ". ", " "]
    """Text separators used for chunking, in order of preference"""


# ========================================================================
# CONFIGURACIONES DE APLICACIÓN (No son credenciales)
# ========================================================================
# Estas configuraciones están aquí en el código, no en .env
# Pueden ser sobrescritas por variables de entorno si es necesario
# ========================================================================

# Exportar configuraciones para uso directo (además de settings)
# Estas están en la clase APISettings arriba, pero también las exportamos
# directamente para compatibilidad con código existente
    # ========================================================================
    # Development/Testing
    # ========================================================================
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    """Enable debug mode (not recommended for production)"""
    
    PYTHONUNBUFFERED: str = os.getenv("PYTHONUNBUFFERED", "1")
    """Python unbuffered output (recommended for Docker)"""
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"APISettings("
            f"LOG_LEVEL={self.LOG_LEVEL}, "
            f"DEVICE={self.DEVICE}, "
            f"DVC_S3_BUCKET={self.DVC_S3_BUCKET}, "
            f"AWS_REGION={self.AWS_REGION}"
            f")"
        )
    
    def validate(self) -> bool:
        """
        Valida la configuración.
        
        Returns:
            True si la configuración es válida, False en caso contrario
        """
        errors = []
        
        # Validar LOG_LEVEL
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.LOG_LEVEL.upper() not in valid_log_levels:
            errors.append(f"Invalid LOG_LEVEL: {self.LOG_LEVEL}. Must be one of {valid_log_levels}")
        
        # Validar DEVICE si está configurado
        if self.DEVICE and self.DEVICE not in ["cuda", "cpu"]:
            errors.append(f"Invalid DEVICE: {self.DEVICE}. Must be 'cuda' or 'cpu'")
        
        # Validar puerto
        if not (1 <= self.API_PORT <= 65535):
            errors.append(f"Invalid API_PORT: {self.API_PORT}. Must be between 1 and 65535")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        return True


# Instancia global de configuración
settings = APISettings()

# Exportar configuraciones para uso directo (además de settings)
# Estas están en la clase APISettings, pero también las exportamos
# directamente para compatibilidad con código existente
TASK_PREFIX: str = settings.TASK_PREFIX
MAX_INPUT_LENGTH: int = settings.MAX_INPUT_LENGTH
CHUNK_SIZE: int = settings.CHUNK_SIZE
CHUNK_OVERLAP: int = settings.CHUNK_OVERLAP
SEPARATORS: list = settings.SEPARATORS
DEFAULT_MAX_LENGTH: int = settings.DEFAULT_MAX_LENGTH
DEFAULT_NUM_BEAMS: int = settings.DEFAULT_NUM_BEAMS


def get_settings() -> APISettings:
    """
    Obtiene la instancia de configuración.
    
    Returns:
        APISettings: Instancia de configuración
    """
    return settings


def load_env_file(env_path: Optional[Path] = None) -> None:
    """
    Carga variables de entorno desde un archivo .env.
    
    Args:
        env_path: Ruta al archivo .env. Si es None, busca .env en el directorio actual
    """
    if env_path is None:
        # Buscar .env en el directorio actual y directorios padres
        current_dir = Path(__file__).parent
        env_path = current_dir / ".env"
        
        if not env_path.exists():
            # Intentar en el directorio padre
            env_path = current_dir.parent / ".env"
    
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    # Ignorar comentarios y líneas vacías
                    if not line or line.startswith("#"):
                        continue
                    # Parsear KEY=VALUE
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # Solo establecer si no existe
                        if key and key not in os.environ:
                            os.environ[key] = value
        except Exception as e:
            logger.warning(f"Could not load .env file: {e}")
    else:
        logger.debug(f"No .env file found at {env_path} (this is OK if running in production with IAM roles)")


# Intentar cargar .env al importar el módulo
try:
    load_env_file()
except Exception as e:
    logger.debug(f"Could not load .env file: {e}")

