# ============================================================================
# Script para descargar modelos desde DVC usando dvc pull (PowerShell)
# ============================================================================

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)

Write-Host "Descargando modelos desde DVC" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow

Push-Location $ProjectRoot

# Verificar que DVC está instalado
if (-not (Get-Command dvc -ErrorAction SilentlyContinue)) {
    Write-Host "Error: DVC no está instalado" -ForegroundColor Red
    Write-Host "Instala DVC con: pip install dvc[s3]" -ForegroundColor Yellow
    exit 1
}

# Verificar configuración de DVC
if (-not (Test-Path ".dvc\config")) {
    Write-Host "Error: DVC no está configurado" -ForegroundColor Red
    Write-Host "Configura DVC con: make setup-dvc" -ForegroundColor Yellow
    exit 1
}

# Verificar credenciales AWS
try {
    aws sts get-caller-identity | Out-Null
} catch {
    Write-Host "Error: Credenciales AWS no configuradas" -ForegroundColor Red
    Write-Host "Configura con: aws configure" -ForegroundColor Yellow
    exit 1
}

Write-Host "Descargando modelos desde DVC..." -ForegroundColor Green
dvc pull models/t5_base models/baseline_classifier

if ($LASTEXITCODE -eq 0) {
    Write-Host "Modelos descargados exitosamente" -ForegroundColor Green
    Write-Host ""
    Write-Host "Los modelos están en:"
    Write-Host "  - $ProjectRoot\models\t5_base"
    Write-Host "  - $ProjectRoot\models\baseline_classifier"
} else {
    Write-Host "Error al descargar modelos" -ForegroundColor Red
    exit 1
}

Pop-Location

