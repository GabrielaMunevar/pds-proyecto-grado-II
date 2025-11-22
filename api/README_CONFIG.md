# Configuración de la API

Este documento describe todas las opciones de configuración disponibles para la API de PLS.

## Arquitectura de Configuración

La API usa una arquitectura de configuración en dos niveles:

1. **Configuraciones en `config.py`**: Valores por defecto y constantes (versionadas en Git)
2. **Credenciales en `.env`**: Secretos y credenciales (NO versionadas, solo para desarrollo local)

### ¿Qué va en cada lugar?

- **`api/config.py`**: Configuraciones de la aplicación (puertos, límites, prefijos, etc.)
- **`.env`**: Solo credenciales (AWS keys, API keys, tokens)

## Configuraciones en config.py

Todas las configuraciones de la aplicación están definidas en `api/config.py` con valores por defecto razonables. Estas pueden ser sobrescritas por variables de entorno si es necesario.

### Configuraciones Disponibles

#### Logging
- `LOG_LEVEL`: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - Valor por defecto: `"INFO"`

#### Modelo
- `DEVICE`: Dispositivo para inferencia (cuda o cpu)
  - Valor por defecto: `None` (auto-detecta)
- `MODEL_PATH`: Ruta al modelo T5
  - Valor por defecto: `None` (auto-detecta)
- `CLASSIFIER_PATH`: Ruta al clasificador
  - Valor por defecto: `None` (auto-detecta)

#### DVC/S3
- `DVC_S3_BUCKET`: Bucket S3 donde DVC almacena modelos
  - Valor por defecto: `None`
- `DVC_S3_PREFIX`: Prefijo/ruta del DVC store
  - Valor por defecto: `"dvcstore"`
- `AWS_REGION`: Región de AWS para S3
  - Valor por defecto: `"us-east-1"`

#### API
- `API_HOST`: Host de la API
  - Valor por defecto: `"0.0.0.0"`
- `API_PORT`: Puerto de la API
  - Valor por defecto: `8000`
- `CORS_ORIGINS`: Orígenes permitidos para CORS
  - Valor por defecto: `["*"]`

#### Generación (Valores por Defecto)
- `DEFAULT_MAX_LENGTH`: Longitud máxima de salida en tokens
  - Valor por defecto: `256`
- `DEFAULT_NUM_BEAMS`: Número de beams para beam search
  - Valor por defecto: `4`

#### Chunking
- `MAX_INPUT_LENGTH`: Máximo de tokens de entrada por chunk
  - Valor por defecto: `512`
- `CHUNK_SIZE`: Tamaño de chunk para división de texto
  - Valor por defecto: `400`
- `CHUNK_OVERLAP`: Solapamiento entre chunks
  - Valor por defecto: `50`

#### Prompt
- `TASK_PREFIX`: Prefijo agregado al texto antes de enviarlo al modelo
  - Valor por defecto: `"simplify medical text into plain language: "`
- `SEPARATORS`: Separadores de texto para chunking
  - Valor por defecto: `["\n\n", "\n", ". ", " "]`

## Credenciales en .env

El archivo `.env` debe contener **SOLO credenciales y secretos**, no configuraciones.

### Configuración Rápida

1. Crea el archivo `.env` en la raíz del proyecto:
   ```bash
   # En Windows (PowerShell)
   New-Item -ItemType File -Path .env
   
   # En Linux/Mac
   touch .env
   ```

2. Agrega SOLO credenciales:
   ```bash
   # .env - SOLO CREDENCIALES
   AWS_ACCESS_KEY_ID=your-key
   AWS_SECRET_ACCESS_KEY=your-secret
   AWS_SESSION_TOKEN=your-token  # Si usas sesiones temporales
   AWS_DEFAULT_REGION=us-east-1
   OPENAI_API_KEY=your-key
   ```

3. La API cargará automáticamente las credenciales desde `.env` al iniciar.

### Variables de Credenciales

| Variable | Descripción | Requerido |
|----------|-------------|-----------|
| `AWS_ACCESS_KEY_ID` | AWS Access Key | Solo desarrollo local |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Key | Solo desarrollo local |
| `AWS_SESSION_TOKEN` | AWS Session Token (si usas sesiones temporales) | Solo desarrollo local |
| `AWS_DEFAULT_REGION` | Región de AWS | Solo desarrollo local |
| `OPENAI_API_KEY` | Clave de API de OpenAI | Si usas OpenAI |

**Nota**: En producción (AWS ECS), las credenciales de AWS se obtienen automáticamente del IAM Role, no se necesitan en `.env`.

## Sobrescribir Configuraciones con Variables de Entorno

Aunque las configuraciones están en `config.py`, puedes sobrescribirlas con variables de entorno si es necesario:

```bash
# .env - Solo para sobrescribir configuraciones si es necesario
LOG_LEVEL=DEBUG
DVC_S3_BUCKET=my-custom-bucket
```

## Configuración en Producción

En producción (AWS ECS), las configuraciones se pasan a través de Terraform. Ver [infrastructure/README_ENV_VARS.md](../infrastructure/README_ENV_VARS.md) para más detalles.

## Uso en el Código

### Importar Configuraciones

```python
from config import (
    settings,
    TASK_PREFIX,
    DEFAULT_MAX_LENGTH,
    DEFAULT_NUM_BEAMS,
    MAX_INPUT_LENGTH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS
)

# Usar configuraciones
logger.info(f"Log level: {settings.LOG_LEVEL}")
logger.info(f"Task prefix: {TASK_PREFIX}")
logger.info(f"Default max length: {DEFAULT_MAX_LENGTH}")
```

### Acceder a Credenciales

Las credenciales se cargan automáticamente desde `.env` y están disponibles como variables de entorno:

```python
import os

aws_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
```

## Cambiar Configuraciones

### Desarrollo Local

1. Edita `api/config.py` directamente
2. O sobrescribe con variables de entorno en `.env`

### Producción

1. Edita `infrastructure/terraform/terraform.tfvars`
2. Ejecuta `terraform apply`

## Resumen

- **Configuraciones**: `api/config.py` (versionado)
- **Credenciales**: `.env` (NO versionado, solo desarrollo local)
- **Producción**: Terraform pasa configuraciones a ECS
- **Flexibilidad**: Variables de entorno pueden sobrescribir configuraciones
