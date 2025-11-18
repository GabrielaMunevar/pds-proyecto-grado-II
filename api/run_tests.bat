@echo off
echo ==========================================
echo   Ejecutando Tests de la API PLS
echo ==========================================
echo.

REM Activar entorno virtual si existe
if exist "venv\" (
    echo [OK] Activando entorno virtual...
    call venv\Scripts\activate.bat
)

REM Verificar que pytest está instalado
python -c "import pytest" 2>nul
if errorlevel 1 (
    echo [INFO] Instalando dependencias de testing...
    pip install -q pytest pytest-cov pytest-asyncio httpx
)

echo.
echo [INFO] Ejecutando tests unitarios...
echo.

REM Ejecutar tests con coverage
pytest test_api.py -v --cov=. --cov-report=term-missing --cov-report=html

if %errorlevel% equ 0 (
    echo.
    echo [OK] Todos los tests pasaron!
) else (
    echo.
    echo [WARN] Algunos tests fallaron (codigo: %errorlevel%)
)

REM Mostrar ubicación del reporte HTML
if exist "htmlcov\" (
    echo.
    echo [INFO] Reporte de cobertura HTML generado en: htmlcov\index.html
    echo.
    
    REM Abrir reporte en el navegador
    start htmlcov\index.html
)

pause

