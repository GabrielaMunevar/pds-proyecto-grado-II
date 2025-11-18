#!/bin/bash

echo "🚀 Iniciando API de PLS..."
echo ""

# Verificar que el modelo existe
if [ ! -d "../models/t5_base" ]; then
    echo "❌ Error: No se encontró el modelo en ../models/t5_base/"
    echo "   Por favor, asegúrate de que el modelo esté en la ubicación correcta."
    exit 1
fi

echo "✅ Modelo encontrado"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 no está instalado"
    exit 1
fi

echo "✅ Python 3 encontrado"
echo ""

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔄 Activando entorno virtual..."
source venv/bin/activate

# Instalar dependencias
echo "📥 Instalando dependencias..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✅ Dependencias instaladas"
echo ""

# Ejecutar API
echo "🎯 Iniciando API en http://localhost:8000"
echo "📚 Documentación: http://localhost:8000/docs"
echo ""
echo "Presiona Ctrl+C para detener"
echo ""

python main.py

