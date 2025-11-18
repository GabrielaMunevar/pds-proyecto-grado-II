#!/bin/bash

echo "=========================================="
echo "  Ejecutando Tests de la API PLS"
echo "=========================================="
echo ""

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    echo -e "${BLUE}🔄 Activando entorno virtual...${NC}"
    source venv/bin/activate
fi

# Verificar que pytest está instalado
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}📦 Instalando dependencias de testing...${NC}"
    pip install -q pytest pytest-cov pytest-asyncio httpx
fi

echo ""
echo -e "${BLUE}🧪 Ejecutando tests unitarios...${NC}"
echo ""

# Ejecutar tests con coverage
pytest test_api.py -v --cov=. --cov-report=term-missing --cov-report=html

TEST_EXIT_CODE=$?

echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Todos los tests pasaron!${NC}"
else
    echo -e "${YELLOW}⚠️  Algunos tests fallaron (código: $TEST_EXIT_CODE)${NC}"
fi

# Mostrar ubicación del reporte HTML
if [ -d "htmlcov" ]; then
    echo ""
    echo -e "${BLUE}📊 Reporte de cobertura HTML generado en: htmlcov/index.html${NC}"
    echo ""
    
    # Intentar abrir el reporte en el navegador (Linux/Mac)
    if command -v xdg-open &> /dev/null; then
        echo "Abriendo reporte en navegador..."
        xdg-open htmlcov/index.html &
    elif command -v open &> /dev/null; then
        echo "Abriendo reporte en navegador..."
        open htmlcov/index.html &
    fi
fi

exit $TEST_EXIT_CODE

