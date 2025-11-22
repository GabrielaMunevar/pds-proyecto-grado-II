# ============================================================================
# Script de Despliegue para AWS - Medical PLS API (PowerShell)
# ============================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet("build", "push", "infra", "update", "all", "url")]
    [string]$Command = "all"
)

$ErrorActionPreference = "Stop"

# Configuración
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$ApiDir = Join-Path $ProjectRoot "api"
$TerraformDir = Join-Path $ProjectRoot "infrastructure" "terraform"
$AwsRegion = if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-1" }
$ProjectName = if ($env:PROJECT_NAME) { $env:PROJECT_NAME } else { "medical-pls-api" }
$Environment = if ($env:ENVIRONMENT) { $env:ENVIRONMENT } else { "prod" }

# Funciones
function Write-Header {
    param([string]$Message)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor Blue
}

# Verificar dependencias
function Test-Dependencies {
    Write-Header "Verificando Dependencias"
    
    $missing = @()
    
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        $missing += "docker"
    }
    if (-not (Get-Command terraform -ErrorAction SilentlyContinue)) {
        $missing += "terraform"
    }
    if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
        $missing += "aws-cli"
    }
    
    if ($missing.Count -gt 0) {
        Write-Error "Faltan las siguientes dependencias: $($missing -join ', ')"
        exit 1
    }
    
    Write-Success "Todas las dependencias están instaladas"
}

# Verificar credenciales AWS
function Test-AwsCredentials {
    Write-Header "Verificando Credenciales AWS"
    
    try {
        $identity = aws sts get-caller-identity 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Error"
        }
        $accountId = (aws sts get-caller-identity --query Account --output text)
        Write-Success "Credenciales AWS válidas (Account: $accountId)"
    } catch {
        Write-Error "No se encontraron credenciales AWS válidas"
        Write-Info "Configura tus credenciales con: aws configure"
        exit 1
    }
}

# Build Docker image
function Build-Image {
    Write-Header "Construyendo Imagen Docker"
    
    Push-Location $ApiDir
    
    Write-Info "Construyendo imagen..."
    docker build -t "${ProjectName}:latest" .
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Error al construir la imagen"
        exit 1
    }
    
    Write-Success "Imagen construida exitosamente"
    
    Pop-Location
}

# Login a ECR
function Connect-Ecr {
    Write-Header "Autenticando con ECR"
    
    $accountId = aws sts get-caller-identity --query Account --output text
    $ecrUrl = "$accountId.dkr.ecr.$AwsRegion.amazonaws.com"
    
    Write-Info "Haciendo login a ECR..."
    $password = aws ecr get-login-password --region $AwsRegion
    $password | docker login --username AWS --password-stdin $ecrUrl
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Error al hacer login a ECR"
        exit 1
    }
    
    Write-Success "Login exitoso"
}

# Push imagen a ECR
function Push-Image {
    Write-Header "Subiendo Imagen a ECR"
    
    $accountId = aws sts get-caller-identity --query Account --output text
    $ecrUrl = "$accountId.dkr.ecr.$AwsRegion.amazonaws.com"
    $ecrRepo = "$ecrUrl/$ProjectName-api"
    
    Write-Info "Etiquetando imagen..."
    docker tag "${ProjectName}:latest" "$ecrRepo:latest"
    
    Write-Info "Subiendo imagen a ECR..."
    docker push "$ecrRepo:latest"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Error al subir la imagen"
        exit 1
    }
    
    Write-Success "Imagen subida exitosamente"
}

# Aplicar Terraform
function Apply-Terraform {
    Write-Header "Aplicando Infraestructura con Terraform"
    
    Push-Location $TerraformDir
    
    if (-not (Test-Path ".terraform")) {
        Write-Info "Inicializando Terraform..."
        terraform init
    }
    
    Write-Info "Generando plan de Terraform..."
    terraform plan -out=tfplan `
        -var="aws_region=$AwsRegion" `
        -var="environment=$Environment" `
        -var="project_name=$ProjectName"
    
    $response = Read-Host "Presiona Enter para aplicar los cambios o Ctrl+C para cancelar"
    
    Write-Info "Aplicando cambios..."
    terraform apply tfplan
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Error al aplicar Terraform"
        exit 1
    }
    
    Write-Success "Infraestructura desplegada exitosamente"
    
    Pop-Location
}

# Actualizar servicio ECS
function Update-EcsService {
    Write-Header "Actualizando Servicio ECS"
    
    $clusterName = "$ProjectName-cluster"
    $serviceName = "$ProjectName-api-service"
    
    Write-Info "Forzando nueva implementación del servicio..."
    aws ecs update-service `
        --cluster $clusterName `
        --service $serviceName `
        --force-new-deployment `
        --region $AwsRegion | Out-Null
    
    Write-Success "Servicio actualizado, nueva tarea iniciando..."
    Write-Info "Puedes monitorear el progreso en la consola de AWS ECS"
}

# Obtener URL de la API
function Get-ApiUrl {
    Write-Header "Información de Despliegue"
    
    Push-Location $TerraformDir
    
    try {
        $albDns = terraform output -raw alb_dns_name 2>$null
        
        if ($albDns) {
            Write-Host "`nAPI desplegada exitosamente!`n" -ForegroundColor Green
            Write-Host "URL de la API: http://$albDns" -ForegroundColor Cyan
            Write-Host "Documentación: http://$albDns/docs" -ForegroundColor Cyan
            Write-Host "Health Check: http://$albDns/health`n" -ForegroundColor Cyan
        } else {
            Write-Warning "No se pudo obtener la URL del ALB"
        }
    } catch {
        Write-Warning "No se pudo obtener la URL del ALB"
    }
    
    Pop-Location
}

# Función principal
function Main {
    switch ($Command) {
        "build" {
            Test-Dependencies
            Build-Image
        }
        "push" {
            Test-Dependencies
            Test-AwsCredentials
            Connect-Ecr
            Push-Image
        }
        "infra" {
            Test-Dependencies
            Test-AwsCredentials
            Apply-Terraform
        }
        "update" {
            Test-Dependencies
            Test-AwsCredentials
            Update-EcsService
        }
        "all" {
            Test-Dependencies
            Test-AwsCredentials
            Build-Image
            Connect-Ecr
            Push-Image
            Apply-Terraform
            Update-EcsService
            Get-ApiUrl
        }
        "url" {
            Get-ApiUrl
        }
        default {
            Write-Host "Uso: .\deploy.ps1 {build|push|infra|update|all|url}"
            Write-Host ""
            Write-Host "Comandos:"
            Write-Host "  build   - Construir imagen Docker"
            Write-Host "  push    - Subir imagen a ECR"
            Write-Host "  infra   - Aplicar infraestructura con Terraform"
            Write-Host "  update  - Actualizar servicio ECS"
            Write-Host "  all     - Ejecutar todo el proceso de despliegue"
            Write-Host "  url     - Mostrar URL de la API"
            exit 1
        }
    }
}

# Ejecutar
Main

