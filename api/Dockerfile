# ============================================================================
# Dockerfile para API de PLS - Producción
# Multi-stage build para optimizar tamaño de imagen
# ============================================================================

# Stage 1: Builder - Instalar dependencias y compilar
FROM python:3.10-slim as builder

WORKDIR /build

# Instalar dependencias del sistema necesarias para compilar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python en un directorio temporal
RUN pip install --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Imagen final optimizada
FROM python:3.10-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Instalar solo dependencias runtime necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copiar dependencias instaladas desde builder
COPY --from=builder /root/.local /home/appuser/.local

# Copiar código de la API
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser utils.py .
COPY --chown=appuser:appuser utils_dvc.py .

# Copiar frontend (archivos estáticos)
COPY --chown=appuser:appuser static/ ./static/

# Crear directorios para modelos
RUN mkdir -p /app/models/t5_base /app/models/baseline_classifier && \
    chown -R appuser:appuser /app/models

# Cambiar a usuario no-root
USER appuser

# Asegurar que el PATH incluya el directorio local de Python
ENV PATH=/home/appuser/.local/bin:$PATH

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando para ejecutar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
