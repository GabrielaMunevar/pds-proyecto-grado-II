#!/bin/bash

# ============================================================================
# Script para descargar modelos desde DVC usando dvc pull
# ============================================================================

set -e

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${YELLOW}Descargando modelos desde DVC${NC}"
echo "================================"

cd "$PROJECT_ROOT"

# Verificar que DVC está instalado
if ! command -v dvc &> /dev/null; then
    echo -e "${RED}Error: DVC no está instalado${NC}"
    echo "Instala DVC con: pip install dvc[s3]"
    exit 1
fi

# Verificar configuración de DVC
if [ ! -f ".dvc/config" ]; then
    echo -e "${RED}Error: DVC no está configurado${NC}"
    echo "Configura DVC con: make setup-dvc"
    exit 1
fi

# Verificar credenciales AWS
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo -e "${RED}Error: Credenciales AWS no configuradas${NC}"
    echo "Configura con: aws configure"
    exit 1
fi

echo -e "${GREEN}Descargando modelos desde DVC...${NC}"
dvc pull models/t5_base models/baseline_classifier

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Modelos descargados exitosamente${NC}"
    echo ""
    echo "Los modelos están en:"
    echo "  - $PROJECT_ROOT/models/t5_base"
    echo "  - $PROJECT_ROOT/models/baseline_classifier"
else
    echo -e "${RED}Error al descargar modelos${NC}"
    exit 1
fi

