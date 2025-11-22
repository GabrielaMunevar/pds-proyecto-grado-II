# Generador de Resúmenes en Lenguaje Sencillo (PLS) Médicos

Este proyecto tiene como objetivo generar Resúmenes en Lenguaje Sencillo (PLS) a partir de textos biomédicos técnicos para mejorar la comprensión de los pacientes. Utiliza un modelo Transformer basado en T5 fine-tuneado en un conjunto de datos de textos médicos.

## Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

- **`notebooks/`**: Contiene notebooks de Jupyter para análisis de datos, entrenamiento de modelos y evaluación.
- **`models/`**: Almacena los archivos del modelo entrenado (por ejemplo, `.pkl`, `.h5`, `.pt`, `.safetensors`) y configuraciones del tokenizador.
- **`data/`**: Contiene muestras y estructura de los datos utilizados para entrenamiento y pruebas.
  - `raw/`: Datasets originales.
  - `processed/`: Datos limpiados y preprocesados.
- **`api/`**: Código fuente para la API REST (FastAPI) con frontend integrado y configuraciones de despliegue.
  - **`main.py`**: Servidor FastAPI con endpoints REST y servidor de archivos estáticos.
  - **`static/`**: Frontend web (HTML, CSS, JavaScript) integrado en la API.
  - **`config.py`**: Configuración centralizada de la API usando variables de entorno.
  - **`.env`**: Archivo de configuración local (crear manualmente, ver `README_CONFIG.md`).
  - **`README_CONFIG.md`**: Documentación completa de configuración con todas las variables disponibles.
- **`src/`**: Código fuente del proyecto organizado en módulos:
  - **`models/`**: Módulos para entrenamiento y evaluación de modelos (T5, clasificadores).
  - **`data/`**: Scripts para procesamiento y creación de datasets.
  - **`utils/`**: Utilidades compartidas (métricas de evaluación, chunking de texto, análisis de longitud).
  - **`config.py`**: Configuración centralizada del proyecto (prompts, modelos, etc.).
- **`params.yaml`**: Archivo de configuración central del proyecto (parametrización).

## Entregables y Enlaces

- **Repositorio GitHub**: [https://github.com/GabrielaMunevar/pds-proyecto-grado-II](https://github.com/GabrielaMunevar/pds-proyecto-grado-II)
- **Aplicación Desplegada**: [Enlace a la Aplicación Desplegada](LINK_TO_DEPLOYED_APP)

## Instrucciones de Uso

### Prerrequisitos

- Python 3.8+
- Docker (opcional, para despliegue containerizado)
- GPU (recomendado para inferencia)

### Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/GabrielaMunevar/pds-proyecto-grado-II.git
   cd pds-proyecto-grado-II
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Ejecutar la Aplicación

#### API REST con Frontend Integrado (FastAPI)
Para iniciar el servidor de la API con la interfaz web integrada:
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

Una vez iniciado, puedes acceder a:
- **Interfaz Web**: `http://localhost:8000/static/index.html` - Frontend interactivo para generar PLS y clasificar textos
- **Documentación API**: `http://localhost:8000/docs` - Documentación interactiva de Swagger
- **API REST**: Endpoints disponibles en `http://localhost:8000/generate`, `http://localhost:8000/evaluate`, etc.

La interfaz web permite:
- Clasificar textos (técnico vs. lenguaje sencillo)
- Generar resúmenes en lenguaje sencillo (PLS) desde texto técnico biomédico
- Evaluar métricas de calidad (ROUGE, BERTScore, Flesch-Kincaid)
- Ajustar parámetros de generación (longitud máxima, beam search)

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

## Autores

- Jean Munevar
- Gabriela Munevar
- Carlos Chaparro
- Erika Cardenas
- Proyecto de Grado - Maestría
