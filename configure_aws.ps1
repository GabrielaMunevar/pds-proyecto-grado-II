# Script para configurar credenciales AWS de forma segura
# Lee desde aws_credentials.txt y configura variables de entorno

param(
    [switch]$Permanent = $false
)

Write-Host "[INFO] Configurando credenciales AWS..." -ForegroundColor Cyan

# Leer credenciales desde aws_credentials.txt
$credFile = "aws_credentials.txt"

if (-not (Test-Path $credFile)) {
    Write-Host "[ERROR] No se encontro $credFile" -ForegroundColor Red
    Write-Host "Por favor crea el archivo con tus credenciales AWS." -ForegroundColor Yellow
    exit 1
}

Write-Host "[INFO] Leyendo credenciales desde $credFile..." -ForegroundColor Yellow

$credentials = @{}
Get-Content $credFile | ForEach-Object {
    if ($_ -match '^aws_access_key_id=(.+)$') {
        $credentials['AWS_ACCESS_KEY_ID'] = $matches[1]
    }
    elseif ($_ -match '^aws_secret_access_key=(.+)$') {
        $credentials['AWS_SECRET_ACCESS_KEY'] = $matches[1]
    }
    elseif ($_ -match '^aws_session_token=(.+)$') {
        $credentials['AWS_SESSION_TOKEN'] = $matches[1]
    }
    elseif ($_ -match '^s3:\s*(.+)$') {
        $credentials['S3_BUCKET'] = $matches[1]
    }
}

# Verificar que se leyeron las credenciales principales
if (-not $credentials['AWS_ACCESS_KEY_ID']) {
    Write-Host "[ERROR] No se encontro aws_access_key_id" -ForegroundColor Red
    exit 1
}
if (-not $credentials['AWS_SECRET_ACCESS_KEY']) {
    Write-Host "[ERROR] No se encontro aws_secret_access_key" -ForegroundColor Red
    exit 1
}

# Configurar variables de entorno
Write-Host "[OK] Configurando variables de entorno..." -ForegroundColor Green

$env:AWS_ACCESS_KEY_ID = $credentials['AWS_ACCESS_KEY_ID']
$env:AWS_SECRET_ACCESS_KEY = $credentials['AWS_SECRET_ACCESS_KEY']
$env:AWS_DEFAULT_REGION = "us-east-1"

if ($credentials['AWS_SESSION_TOKEN']) {
    $env:AWS_SESSION_TOKEN = $credentials['AWS_SESSION_TOKEN']
    Write-Host "[WARN] Credenciales temporales detectadas (session_token)" -ForegroundColor Yellow
    Write-Host "       Estas credenciales expiraran despues de algunas horas." -ForegroundColor Yellow
}

if ($credentials['S3_BUCKET']) {
    $env:S3_BUCKET = $credentials['S3_BUCKET']
    Write-Host "[INFO] Bucket S3: $($credentials['S3_BUCKET'])" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "[OK] Credenciales configuradas correctamente para esta sesion!" -ForegroundColor Green
Write-Host ""

# Verificar conexion
Write-Host "[INFO] Verificando conexion con AWS..." -ForegroundColor Cyan
try {
    $result = aws sts get-caller-identity 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Conexion exitosa con AWS!" -ForegroundColor Green
        $result | ConvertFrom-Json | Format-List
    }
    else {
        Write-Host "[WARN] No se pudo verificar la conexion (puede que AWS CLI no este instalado)" -ForegroundColor Yellow
        Write-Host "       Las variables de entorno estan configuradas de todos modos." -ForegroundColor Yellow
    }
}
catch {
    Write-Host "[WARN] AWS CLI no esta instalado o no esta en el PATH" -ForegroundColor Yellow
    Write-Host "       Las variables de entorno estan configuradas de todos modos." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[INFO] Proximos pasos:" -ForegroundColor Cyan
Write-Host "1. Ejecuta: .\setup_dvc_s3.ps1" -ForegroundColor White
Write-Host "2. Luego: dvc pull" -ForegroundColor White
Write-Host ""
Write-Host "[IMPORTANTE] Estas variables son temporales para esta sesion de PowerShell." -ForegroundColor Yellow
Write-Host "             Si cierras la ventana, deberas ejecutar este script nuevamente." -ForegroundColor Yellow
Write-Host ""
Write-Host "[TIP] Para configuracion permanente, considera usar AWS CLI:" -ForegroundColor Cyan
Write-Host "      aws configure" -ForegroundColor Gray
