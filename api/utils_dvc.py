"""
Utilidades para descargar modelos desde DVC/S3
DVC almacena los modelos en s3://pds-pls-data-prod/dvcstore
"""

import os
import boto3
import logging
from pathlib import Path
from typing import Optional
import subprocess

logger = logging.getLogger(__name__)


def download_models_from_dvc_s3(
    bucket_name: Optional[str] = None,
    dvc_prefix: Optional[str] = None,
    models_dir: str = "/app/models",
    region: str = "us-east-1"
) -> bool:
    """
    Descarga modelos desde el bucket S3 de DVC.
    
    DVC almacena los archivos con estructura:
    s3://bucket/dvcstore/{hash_prefix}/{hash}
    
    Para los modelos, necesitamos usar dvc pull o descargar directamente
    desde la estructura de DVC.
    
    Args:
        bucket_name: Nombre del bucket S3 de DVC
        dvc_prefix: Prefijo/path del DVC store (default: dvcstore)
        models_dir: Directorio donde se guardarán los modelos
        region: Región de AWS
    
    Returns:
        True si la descarga fue exitosa, False en caso contrario
    """
    if bucket_name is None:
        bucket_name = os.getenv("DVC_S3_BUCKET", "pds-pls-data-prod")
    
    if dvc_prefix is None:
        dvc_prefix = os.getenv("DVC_S3_PREFIX", "dvcstore")
    
    try:
        s3 = boto3.client('s3', region_name=region)
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Intentar usar dvc pull si está disponible
        if _try_dvc_pull(models_path):
            logger.info("Models downloaded using DVC pull")
            return True
        
        # Fallback: download directly from S3 using known structure
        logger.info(f"Downloading models from S3: s3://{bucket_name}/{dvc_prefix}")
        
        # Listar objetos en el bucket de DVC para encontrar los modelos
        # DVC almacena archivos con estructura: {hash_prefix}/{hash}
        # Necesitamos encontrar los archivos de los modelos
        
        # Para modelos T5, buscar archivos comunes
        t5_path = models_path / 't5_base'
        t5_path.mkdir(parents=True, exist_ok=True)
        
        # Archivos necesarios para T5
        t5_files = [
            'config.json',
            'generation_config.json',
            'model.safetensors',
            'special_tokens_map.json',
            'spiece.model',
            'tokenizer_config.json',
            'tokenizer.json'
        ]
        
        # Intentar descargar desde estructura conocida de DVC
        # Si los modelos están trackeados con DVC, podemos usar dvc.list()
        # o buscar en el bucket directamente
        
        # Método alternativo: buscar archivos por nombre en el bucket
        # (esto es menos eficiente pero funciona si conocemos los nombres)
        
        logger.warning("Direct download from DVC S3 requires additional configuration")
        logger.info("Recommended: use 'dvc pull' locally and copy models to Docker image")
        logger.info("Or configure DVC in container to use dvc pull")
        
        return False
        
    except Exception as e:
        logger.error(f"Error downloading models from DVC S3: {e}")
        return False


def _try_dvc_pull(models_path: Path) -> bool:
    """
    Intenta usar dvc pull para descargar modelos.
    
    Returns:
        True si dvc pull fue exitoso, False en caso contrario
    """
    try:
        # Verificar si dvc está disponible
        result = subprocess.run(
            ['dvc', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return False
        
        # Try dvc pull
        logger.info("Attempting to download models with DVC pull...")
        result = subprocess.run(
            ['dvc', 'pull'],
            cwd=str(models_path.parent),
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            return True
        else:
            logger.warning(f"DVC pull failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        logger.debug("DVC is not installed in container")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("DVC pull exceeded time limit")
        return False
    except Exception as e:
        logger.debug(f"Error attempting dvc pull: {e}")
        return False


def download_models_direct_from_s3(
    bucket_name: str,
    s3_models_path: str,
    local_models_dir: str = "/app/models",
    region: str = "us-east-1"
) -> bool:
    """
    Descarga modelos directamente desde una ruta específica en S3.
    Útil si los modelos están en una ruta conocida (no en estructura DVC).
    
    Args:
        bucket_name: Nombre del bucket S3
        s3_models_path: Ruta en S3 donde están los modelos (ej: "models/t5_base")
        local_models_dir: Directorio local donde guardar
        region: Región de AWS
    
    Returns:
        True si exitoso, False en caso contrario
    """
    try:
        s3 = boto3.client('s3', region_name=region)
        local_path = Path(local_models_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Listar objetos en la ruta
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_models_path)
        
        downloaded = 0
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                # Crear estructura de directorios local
                relative_path = key.replace(s3_models_path, '').lstrip('/')
                local_file = local_path / relative_path
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Descargar archivo
                s3.download_file(bucket_name, key, str(local_file))
                downloaded += 1
                logger.debug(f"Downloaded: {relative_path}")
        
        if downloaded > 0:
            logger.info(f"Downloaded {downloaded} files from S3")
            return True
        else:
            logger.warning("No files found to download")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading models from S3: {e}")
        return False

