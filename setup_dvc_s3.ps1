# Script de configuracion DVC + S3 para Windows PowerShell
# Este script configura el remote storage de DVC en S3

Write-Host "[INFO] Configurando DVC con S3..." -ForegroundColor Cyan

# Verificar que DVC esta instalado
$dvcInstalled = Get-Command dvc -ErrorAction SilentlyContinue
if (-not $dvcInstalled) {
    Write-Host "[ERROR] DVC no esta instalado. Instalando..." -ForegroundColor Red
    pip install dvc[s3]
}

# Configurar remote storage S3
Write-Host "[INFO] Configurando remote storage..." -ForegroundColor Yellow
dvc remote add -d myremote s3://pds-pls-data-prod/dvcstore --force

# Configurar region
dvc remote modify myremote region us-east-1

# Verificar configuracion
Write-Host "[OK] Configuracion completada:" -ForegroundColor Green
dvc remote list -v

Write-Host ""
Write-Host "[INFO] Proximos pasos:" -ForegroundColor Cyan
Write-Host "1. Configura tus credenciales AWS:" -ForegroundColor White
Write-Host '   Ejecuta: .\configure_aws.ps1' -ForegroundColor Gray
Write-Host ""
Write-Host "2. O configura manualmente:" -ForegroundColor White
Write-Host '   $env:AWS_ACCESS_KEY_ID="tu_access_key"' -ForegroundColor Gray
Write-Host '   $env:AWS_SECRET_ACCESS_KEY="tu_secret_key"' -ForegroundColor Gray
Write-Host '   $env:AWS_SESSION_TOKEN="tu_session_token"' -ForegroundColor Gray
Write-Host '   $env:AWS_DEFAULT_REGION="us-east-1"' -ForegroundColor Gray
Write-Host ""
Write-Host "3. Agrega datos al tracking:" -ForegroundColor White
Write-Host "   dvc add data/raw" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Sube datos a S3:" -ForegroundColor White
Write-Host "   dvc push" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Ejecuta el pipeline:" -ForegroundColor White
Write-Host "   dvc repro" -ForegroundColor Gray
