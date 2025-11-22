# Generador de Resúmenes en Lenguaje Sencillo (PLS) Médicos

Este proyecto tiene como objetivo generar Resúmenes en Lenguaje Sencillo (PLS) a partir de textos biomédicos técnicos para mejorar la comprensión de los pacientes. Utiliza un modelo Transformer basado en T5 fine-tuneado en un conjunto de datos de textos médicos.

## Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

- **`notebooks/`**: Contiene notebooks de Jupyter para análisis de datos, entrenamiento de modelos y evaluación.
- **`models/`**: Almacena los archivos del modelo entrenado (por ejemplo, `.pkl`, `.h5`, `.pt`, `.safetensors`) y configuraciones del tokenizador.
- **`data/`**: Contiene muestras y estructura de los datos utilizados para entrenamiento y pruebas.
  - `raw/`: Datasets originales.
  - `processed/`: Datos limpiados y preprocesados.
- **`api/`**: Código fuente para la API REST (FastAPI) y configuraciones de despliegue.
  - **`config.py`**: Configuración centralizada de la API usando variables de entorno.
  - **`.env`**: Archivo de configuración local (crear manualmente, ver `README_CONFIG.md`).
  - **`README_CONFIG.md`**: Documentación completa de configuración con todas las variables disponibles.
- **`src/`**: Código fuente para el dashboard de Streamlit (`src/dashboard/`) y scripts de utilidades.
- **`params.yaml`**: Archivo de configuración central del proyecto (parametrización).
- **`src/config.py`**: Configuración centralizada del proyecto (prompts, modelos, etc.).

## Entregables y Enlaces

- **Repositorio GitHub**: [Enlace al Repositorio GitHub](LINK_TO_GITHUB_REPO)
- **Aplicación Desplegada**: [Enlace a la Aplicación Desplegada](LINK_TO_DEPLOYED_APP)

*(Nota: Por favor actualiza los enlaces anteriores con las URLs reales)*

## Instrucciones de Uso

### Prerrequisitos

- Python 3.8+
- Docker (opcional, para despliegue containerizado)
- GPU (recomendado para inferencia)

### Instalación

1. Clonar el repositorio:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Ejecutar la Aplicación

#### 1. Dashboard Streamlit (Interfaz Interactiva)
Para lanzar el dashboard fácil de usar:
```bash
streamlit run src/dashboard/app.py
```
El dashboard permite ingresar texto técnico y ver el resumen generado, junto con métricas de evaluación.

#### 2. API REST (FastAPI)
Para iniciar el servidor de la API:
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```
La documentación estará disponible en `http://localhost:8000/docs`.

## Despliegue

El proyecto incluye configuración para despliegue usando Docker y AWS.

### Docker
Construir y ejecutar el contenedor:
```bash
cd api
docker build -t pls-api .
docker run -p 8000:8000 pls-api
```

Para instrucciones más detalladas de despliegue en AWS, consulta `deploy/README_AWS.md`.

## Parametrización

El sistema está diseñado para ser fácilmente configurable. Los parámetros clave están definidos en **`params.yaml`** y pueden ajustarse sin cambiar el código:

- **Datos**: Reglas de preprocesamiento (longitud mín/máx, deduplicación).
- **Modelo**: Arquitectura (`t5-base`), parámetros de búsqueda por haz, longitudes máximas de tokens.
- **Evaluación**: Selección de métricas (ROUGE, BERTScore, Flesch-Kincaid).
- **Entrenamiento**: Tasas de aprendizaje, tamaños de lote, épocas.

## Ejemplo de Uso

### Ejemplo de Cliente Python

```python
import requests

url = "http://localhost:8000/generate"
payload = {
    "technical_text": "Systemic arterial hypertension is a chronic medical condition characterized by persistent elevation of blood pressure...",
    "max_length": 256,
    "num_beams": 4
}

response = requests.post(url, json=payload)
print(response.json()["generated_pls"])
# Salida: "High blood pressure is a long-term health problem..."
```

## Credenciales

Si la aplicación requiere autenticación (por ejemplo, para rutas de administrador específicas), utiliza las siguientes credenciales de ejemplo para pruebas:

- **Usuario**: `admin`
- **Contraseña**: `pls_project_2025`

*(Nota: Estas son credenciales de ejemplo para fines de evaluación si aplica)*

## Autores

- [Nombres de Miembros del Equipo]
- Proyecto de Grado - Maestría
