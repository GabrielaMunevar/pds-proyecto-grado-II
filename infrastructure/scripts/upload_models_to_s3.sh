#!/bin/bash

# ============================================================================
# Script para subir modelos a S3
# ============================================================================

set -e

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo -e "${YELLOW}Subiendo modelos a S3${NC}"
echo "================================"

# Verificar que existe el directorio de modelos
if [ ! -d "$MODELS_DIR" ]; then
    echo -e "${RED}Error: No se encontró el directorio models/${NC}"
    exit 1
fi

# Obtener nombre del bucket desde Terraform o pedirlo
if [ -z "$S3_BUCKET" ]; then
    echo -e "${YELLOW}Ingresa el nombre del bucket S3 (o déjalo vacío para usar el de Terraform):${NC}"
    read -p "Bucket: " S3_BUCKET
fi

# Si aún está vacío, intentar obtenerlo de Terraform
if [ -z "$S3_BUCKET" ]; then
    TERRAFORM_DIR="$PROJECT_ROOT/infrastructure/terraform"
    if [ -d "$TERRAFORM_DIR" ]; then
        cd "$TERRAFORM_DIR"
        S3_BUCKET=$(terraform output -raw s3_models_bucket 2>/dev/null || echo "")
    fi
fi

if [ -z "$S3_BUCKET" ]; then
    echo -e "${RED}Error: No se especificó el bucket S3${NC}"
    echo "Usa: export S3_BUCKET=tu-bucket-name"
    exit 1
fi

echo -e "${GREEN}Bucket: $S3_BUCKET${NC}"

# Subir modelo T5
if [ -d "$MODELS_DIR/t5_base" ]; then
    echo -e "\n${YELLOW}Subiendo modelo T5...${NC}"
    aws s3 sync "$MODELS_DIR/t5_base" "s3://$S3_BUCKET/models/t5_base" --region "$AWS_REGION"
    echo -e "${GREEN}Modelo T5 subido${NC}"
else
    echo -e "${RED}Advertencia: No se encontró models/t5_base${NC}"
fi

# Subir clasificador
if [ -d "$MODELS_DIR/baseline_classifier" ]; then
    echo -e "\n${YELLOW}Subiendo clasificador...${NC}"
    aws s3 sync "$MODELS_DIR/baseline_classifier" "s3://$S3_BUCKET/models/baseline_classifier" --region "$AWS_REGION"
    echo -e "${GREEN}Clasificador subido${NC}"
else
    echo -e "${RED}Advertencia: No se encontró models/baseline_classifier${NC}"
fi

echo -e "\n${GREEN}Modelos subidos exitosamente a s3://$S3_BUCKET/models/${NC}"

