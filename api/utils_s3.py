"""
Utilidades para descargar modelos desde S3
"""

import os
import boto3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def download_models_from_s3(
    bucket_name: Optional[str] = None,
    models_dir: str = "/app/models",
    region: str = "us-east-1"
) -> bool:
    """
    Descarga modelos desde S3 si no existen localmente.
    
    Args:
        bucket_name: Nombre del bucket S3. Si es None, se usa la variable de entorno S3_MODELS_BUCKET
        models_dir: Directorio donde se guardarán los modelos
        region: Región de AWS
    
    Returns:
        True si la descarga fue exitosa o los modelos ya existen, False en caso contrario
    """
    if bucket_name is None:
        bucket_name = os.getenv("S3_MODELS_BUCKET")
    
    if not bucket_name:
        logger.warning("S3_MODELS_BUCKET not configured, using local models")
        return False
    
    try:
        s3 = boto3.client('s3', region_name=region)
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Lista de archivos del modelo T5 que necesitamos
        t5_files = [
            'config.json',
            'generation_config.json',
            'model.safetensors',
            'special_tokens_map.json',
            'spiece.model',
            'tokenizer_config.json',
            'tokenizer.json'
        ]
        
        # Descargar modelo T5
        t5_path = models_path / 't5_base'
        t5_path.mkdir(parents=True, exist_ok=True)
        
        t5_exists = all((t5_path / f).exists() for f in t5_files)
        
        if not t5_exists:
            logger.info("Downloading T5 model from S3...")
            for file in t5_files:
                s3_key = f'models/t5_base/{file}'
                local_file = t5_path / file
                
                try:
                    s3.download_file(bucket_name, s3_key, str(local_file))
                    logger.debug(f"Downloaded: {file}")
                except Exception as e:
                    logger.warning(f"Could not download {file}: {e}")
            
            logger.info("T5 model downloaded successfully")
        else:
            logger.info("T5 model already exists locally")
        
        # Descargar clasificador
        classifier_path = models_path / 'baseline_classifier'
        classifier_path.mkdir(parents=True, exist_ok=True)
        
        classifier_files = ['classifier.pkl', 'vectorizer.pkl']
        classifier_exists = all((classifier_path / f).exists() for f in classifier_files)
        
        if not classifier_exists:
            logger.info("Downloading classifier from S3...")
            for file in classifier_files:
                s3_key = f'models/baseline_classifier/{file}'
                local_file = classifier_path / file
                
                try:
                    s3.download_file(bucket_name, s3_key, str(local_file))
                    logger.debug(f"Downloaded: {file}")
                except Exception as e:
                    logger.warning(f"Could not download {file}: {e}")
            
            logger.info("Classifier downloaded successfully")
        else:
            logger.info("Classifier already exists locally")
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading models from S3: {e}")
        return False

