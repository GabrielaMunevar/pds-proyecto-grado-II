#!/bin/bash
# Script de configuración DVC + S3
# Este script configura el remote storage de DVC en S3

echo "🔧 Configurando DVC con S3..."

# Verificar que DVC está instalado
if ! command -v dvc &> /dev/null; then
    echo "❌ DVC no está instalado. Instalando..."
    pip install dvc[s3]
fi

# Configurar remote storage S3
echo "📦 Configurando remote storage..."
dvc remote add -d myremote s3://pds-pls-data-prod/dvcstore --force

# Configurar región
dvc remote modify myremote region us-east-1

# Verificar configuración
echo "✅ Configuración completada:"
dvc remote list -v

echo ""
echo "📝 Próximos pasos:"
echo "1. Configura tus credenciales AWS:"
echo "   - Copia .env.template a .env"
echo "   - Completa con tus credenciales"
echo "   - O usa: aws configure"
echo ""
echo "2. Agrega datos al tracking:"
echo "   dvc add data/raw"
echo ""
echo "3. Sube datos a S3:"
echo "   dvc push"
echo ""
echo "4. Ejecuta el pipeline:"
echo "   dvc repro"

