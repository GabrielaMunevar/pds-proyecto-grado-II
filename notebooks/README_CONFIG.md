# Configuración del Modelo T5-BASE

Este documento describe la configuración centralizada para entrenamiento y evaluación del modelo T5-BASE.

## Arquitectura de Configuración

La configuración del modelo usa el mismo patrón que la API:

1. **Configuraciones en `config_model.py`**: Valores por defecto y constantes (versionadas en Git)
2. **Variables de entorno**: Para sobrescribir configuraciones si es necesario
3. **Rutas dinámicas**: Se configuran automáticamente o mediante variables de entorno

### ¿Qué va en cada lugar?

- **`notebooks/config_model.py`**: Todas las configuraciones del modelo (versionado)
- **Variables de entorno**: Solo para sobrescribir configuraciones específicas

## Configuraciones Disponibles

### Modelo Base

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `MODEL_NAME` | Nombre del modelo de Hugging Face | `t5-base` |
| `TASK_PREFIX` | Prefijo agregado al texto antes del modelo | `simplify medical text: ` |

### Tokenización

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `MAX_INPUT_LENGTH` | Longitud máxima de tokens para input | `512` |
| `MAX_TARGET_LENGTH` | Longitud máxima de tokens para output | `256` |

### Chunking

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `CHUNK_SIZE` | Tamaño de chunk para división de texto (tokens) | `400` |
| `CHUNK_OVERLAP` | Solapamiento entre chunks (tokens) | `50` |
| `SEPARATORS` | Separadores de texto para chunking | `["\n\n", "\n", ". ", " "]` |

### Entrenamiento

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `NUM_EPOCHS` | Número de épocas de entrenamiento | `3` |
| `LEARNING_RATE` | Learning rate para optimizador | `3e-4` |
| `BATCH_SIZE` | Batch size por dispositivo (A100) | `16` |
| `GRAD_ACCUM_STEPS` | Pasos de acumulación de gradiente | `2` |
| `WARMUP_STEPS` | Pasos de warmup para scheduler | `500` |
| `WEIGHT_DECAY` | Weight decay para regularización | `0.01` |
| `EVAL_STEPS` | Pasos entre evaluaciones | `200` |
| `SAVE_STEPS` | Pasos entre guardado de checkpoints | `200` |
| `SAVE_TOTAL_LIMIT` | Número máximo de checkpoints | `3` |

### Generación

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `NUM_BEAMS` | Número de beams para beam search | `4` |

### Datos

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `SEED` | Semilla para reproducibilidad | `42` |
| `TRAIN_RATIO` | Proporción de datos para entrenamiento | `0.8` |
| `VAL_RATIO` | Proporción de datos para validación | `0.1` |

### Evaluación

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `BATCH_SIZE_EVAL` | Batch size para evaluación | `32` |

### Rutas (Configuración Dinámica)

| Variable | Descripción | Valor por Defecto |
|----------|-------------|-------------------|
| `CSV_PATH` | Ruta al archivo CSV (se detecta automáticamente) | `None` |
| `DRIVE_BASE` | Ruta base de Google Drive (Colab) | `/content/drive/MyDrive/PLS_Project` |
| `MODEL_DIR` | Ruta donde se guarda/carga el modelo | `{DRIVE_BASE}/models/t5_pls/final` |
| `RESULTS_DIR` | Ruta donde se guardan resultados | `{DRIVE_BASE}/results` |
| `PLOTS_DIR` | Ruta donde se guardan gráficos | `{RESULTS_DIR}/plots` |

## Uso en los Scripts

### Entrenamiento (`train_t5_base_pls_a100.py`)

```python
from config_model import ModelConfig, setup_paths

# Crear instancia de configuración
config = ModelConfig()

# Configurar rutas
setup_paths(drive_base="/content/drive/MyDrive/PLS_Project")

# Usar configuraciones
print(f"Model: {config.MODEL_NAME}")
print(f"Task prefix: {config.TASK_PREFIX}")
print(f"Batch size: {config.BATCH_SIZE}")
print(f"Seed: {config.SEED}")
```

### Evaluación (`evaluate_t5_base_pls_a100.py`)

```python
from config_model import ModelConfig, setup_paths

# Crear instancia de configuración
config = ModelConfig()

# Para evaluación, usar batch size mayor
config.BATCH_SIZE = config.BATCH_SIZE_EVAL

# Configurar rutas
setup_paths(drive_base="/content/drive/MyDrive/PLS_Project")

# Usar configuraciones
print(f"Model: {config.MODEL_NAME}")
print(f"Seed: {config.SEED} (debe ser igual a entrenamiento)")
```

## Sobrescribir Configuraciones

Puedes sobrescribir configuraciones usando variables de entorno:

```python
import os

# Antes de importar config_model
os.environ['NUM_EPOCHS'] = '5'
os.environ['LEARNING_RATE'] = '5e-4'
os.environ['BATCH_SIZE'] = '32'

# Ahora importar
from config_model import ModelConfig
config = ModelConfig()

# Las configuraciones estarán sobrescritas
print(config.NUM_EPOCHS)  # 5
print(config.LEARNING_RATE)  # 5e-4
print(config.BATCH_SIZE)  # 32
```

## Validación de Configuración

El archivo `config_model.py` incluye validación automática:

```python
from config_model import ModelConfig

config = ModelConfig()

# Validar configuración
if not config.validate():
    print("Error: Configuración inválida")
    exit(1)
```

## Compatibilidad

Los scripts mantienen compatibilidad hacia atrás:

- Si `config_model.py` no está disponible, los scripts usan configuración local (fallback)
- Las configuraciones se pueden usar directamente: `config.TASK_PREFIX`, `config.SEED`, etc.

## Importante: Semilla (SEED)

**CRÍTICO**: La semilla (`SEED`) debe ser la misma en entrenamiento y evaluación para garantizar reproducibilidad:

- Entrenamiento: usa `SEED=42` para el split de datos
- Evaluación: debe usar `SEED=42` para reconstruir el mismo split

Si cambias la semilla en entrenamiento, **debes cambiarla también en evaluación**.

## Resumen

- **Configuraciones**: `notebooks/config_model.py` (versionado)
- **Variables de entorno**: Para sobrescribir configuraciones
- **Rutas**: Se configuran automáticamente o mediante `setup_paths()`
- **Validación**: Automática con `config.validate()`
- **Compatibilidad**: Fallback si `config_model.py` no está disponible

