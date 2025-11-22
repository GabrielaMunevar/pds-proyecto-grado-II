# ============================================================================
# Script para subir modelos a S3 (PowerShell)
# ============================================================================

param(
    [string]$S3Bucket = $env:S3_BUCKET,
    [string]$AwsRegion = if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-1" }
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$ModelsDir = Join-Path $ProjectRoot "models"

Write-Host "Subiendo modelos a S3" -ForegroundColor Yellow
Write-Host "================================" -ForegroundColor Yellow

# Verificar que existe el directorio de modelos
if (-not (Test-Path $ModelsDir)) {
    Write-Host "Error: No se encontró el directorio models/" -ForegroundColor Red
    exit 1
}

# Obtener nombre del bucket
if (-not $S3Bucket) {
    $S3Bucket = Read-Host "Ingresa el nombre del bucket S3 (o déjalo vacío para usar el de Terraform)"
}

# Si aún está vacío, intentar obtenerlo de Terraform
if (-not $S3Bucket) {
    $TerraformDir = Join-Path $ProjectRoot "infrastructure" "terraform"
    if (Test-Path $TerraformDir) {
        Push-Location $TerraformDir
        try {
            $S3Bucket = terraform output -raw s3_models_bucket 2>$null
        } catch {
            # Ignorar error
        }
        Pop-Location
    }
}

if (-not $S3Bucket) {
    Write-Host "Error: No se especificó el bucket S3" -ForegroundColor Red
    Write-Host "Usa: `$env:S3_BUCKET='tu-bucket-name'" -ForegroundColor Yellow
    exit 1
}

Write-Host "Bucket: $S3Bucket" -ForegroundColor Green

# Subir modelo T5
$T5Path = Join-Path $ModelsDir "t5_base"
if (Test-Path $T5Path) {
    Write-Host "`nSubiendo modelo T5..." -ForegroundColor Yellow
    aws s3 sync $T5Path "s3://$S3Bucket/models/t5_base" --region $AwsRegion
    Write-Host "Modelo T5 subido" -ForegroundColor Green
} else {
    Write-Host "Advertencia: No se encontró models/t5_base" -ForegroundColor Red
}

# Subir clasificador
$ClassifierPath = Join-Path $ModelsDir "baseline_classifier"
if (Test-Path $ClassifierPath) {
    Write-Host "`nSubiendo clasificador..." -ForegroundColor Yellow
    aws s3 sync $ClassifierPath "s3://$S3Bucket/models/baseline_classifier" --region $AwsRegion
    Write-Host "Clasificador subido" -ForegroundColor Green
} else {
    Write-Host "Advertencia: No se encontró models/baseline_classifier" -ForegroundColor Red
}

Write-Host "`nModelos subidos exitosamente a s3://$S3Bucket/models/" -ForegroundColor Green

