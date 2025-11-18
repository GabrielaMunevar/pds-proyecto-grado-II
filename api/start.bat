@echo off
echo ========================================
echo   API de PLS - Inicio
echo ========================================
echo.

REM Verificar que el modelo existe
if not exist "..\models\t5_base\" (
    echo [ERROR] No se encontro el modelo en ..\models\t5_base\
    echo Por favor, asegurate de que el modelo este en la ubicacion correcta.
    pause
    exit /b 1
)

echo [OK] Modelo encontrado
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no esta instalado
    pause
    exit /b 1
)

echo [OK] Python encontrado
echo.

REM Crear entorno virtual si no existe
if not exist "venv\" (
    echo Creando entorno virtual...
    python -m venv venv
)

REM Activar entorno virtual
echo Activando entorno virtual...
call venv\Scripts\activate.bat

REM Instalar dependencias
echo Instalando dependencias...
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo.
echo [OK] Dependencias instaladas
echo.

REM Ejecutar API
echo Iniciando API en http://localhost:8000
echo Documentacion: http://localhost:8000/docs
echo.
echo Presiona Ctrl+C para detener
echo.

python main.py

pause

