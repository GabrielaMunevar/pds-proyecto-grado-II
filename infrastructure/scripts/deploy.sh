#!/bin/bash

# ============================================================================
# Script de Despliegue para AWS - Medical PLS API
# ============================================================================

set -e  # Exit on error

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
API_DIR="$PROJECT_ROOT/api"
TERRAFORM_DIR="$PROJECT_ROOT/infrastructure/terraform"
AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_NAME="${PROJECT_NAME:-medical-pls-api}"
ENVIRONMENT="${ENVIRONMENT:-prod}"

# Funciones
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Verificar dependencias
check_dependencies() {
    print_header "Verificando Dependencias"
    
    local missing_deps=()
    
    command -v docker >/dev/null 2>&1 || missing_deps+=("docker")
    command -v terraform >/dev/null 2>&1 || missing_deps+=("terraform")
    command -v aws >/dev/null 2>&1 || missing_deps+=("aws-cli")
    command -v jq >/dev/null 2>&1 || missing_deps+=("jq")
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Faltan las siguientes dependencias: ${missing_deps[*]}"
        exit 1
    fi
    
    print_success "Todas las dependencias están instaladas"
}

# Verificar credenciales AWS
check_aws_credentials() {
    print_header "Verificando Credenciales AWS"
    
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        print_error "No se encontraron credenciales AWS válidas"
        print_info "Configura tus credenciales con: aws configure"
        exit 1
    fi
    
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    print_success "Credenciales AWS válidas (Account: $account_id)"
}

# Build Docker image
build_image() {
    print_header "Construyendo Imagen Docker"
    
    cd "$API_DIR"
    
    print_info "Construyendo imagen..."
    docker build -t "$PROJECT_NAME:latest" .
    
    print_success "Imagen construida exitosamente"
}

# Login a ECR
ecr_login() {
    print_header "Autenticando con ECR"
    
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local ecr_url="$account_id.dkr.ecr.$AWS_REGION.amazonaws.com"
    
    print_info "Haciendo login a ECR..."
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ecr_url"
    
    print_success "Login exitoso"
}

# Push imagen a ECR
push_image() {
    print_header "Subiendo Imagen a ECR"
    
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local ecr_url="$account_id.dkr.ecr.$AWS_REGION.amazonaws.com"
    local ecr_repo="$ecr_url/$PROJECT_NAME-api"
    
    # Tag imagen
    print_info "Etiquetando imagen..."
    docker tag "$PROJECT_NAME:latest" "$ecr_repo:latest"
    
    # Push
    print_info "Subiendo imagen a ECR..."
    docker push "$ecr_repo:latest"
    
    print_success "Imagen subida exitosamente"
}

# Aplicar Terraform
apply_terraform() {
    print_header "Aplicando Infraestructura con Terraform"
    
    cd "$TERRAFORM_DIR"
    
    # Inicializar Terraform
    if [ ! -d ".terraform" ]; then
        print_info "Inicializando Terraform..."
        terraform init
    fi
    
    # Plan
    print_info "Generando plan de Terraform..."
    terraform plan -out=tfplan \
        -var="aws_region=$AWS_REGION" \
        -var="environment=$ENVIRONMENT" \
        -var="project_name=$PROJECT_NAME"
    
    # Aplicar
    read -p "$(echo -e ${YELLOW}Presiona Enter para aplicar los cambios o Ctrl+C para cancelar...${NC})"
    
    print_info "Aplicando cambios..."
    terraform apply tfplan
    
    print_success "Infraestructura desplegada exitosamente"
}

# Actualizar servicio ECS
update_ecs_service() {
    print_header "Actualizando Servicio ECS"
    
    local cluster_name="$PROJECT_NAME-cluster"
    local service_name="$PROJECT_NAME-api-service"
    
    print_info "Forzando nueva implementación del servicio..."
    aws ecs update-service \
        --cluster "$cluster_name" \
        --service "$service_name" \
        --force-new-deployment \
        --region "$AWS_REGION" >/dev/null
    
    print_success "Servicio actualizado, nueva tarea iniciando..."
    print_info "Puedes monitorear el progreso en la consola de AWS ECS"
}

# Obtener URL de la API
get_api_url() {
    print_header "Información de Despliegue"
    
    cd "$TERRAFORM_DIR"
    
    local alb_dns=$(terraform output -raw alb_dns_name 2>/dev/null || echo "N/A")
    
    if [ "$alb_dns" != "N/A" ]; then
        echo -e "\n${GREEN}API desplegada exitosamente!${NC}\n"
        echo -e "URL de la API: ${BLUE}http://$alb_dns${NC}"
        echo -e "Documentación: ${BLUE}http://$alb_dns/docs${NC}"
        echo -e "Health Check: ${BLUE}http://$alb_dns/health${NC}\n"
    else
        print_warning "No se pudo obtener la URL del ALB"
    fi
}

# Función principal
main() {
    local command="${1:-all}"
    
    case "$command" in
        "build")
            check_dependencies
            build_image
            ;;
        "push")
            check_dependencies
            check_aws_credentials
            ecr_login
            push_image
            ;;
        "infra")
            check_dependencies
            check_aws_credentials
            apply_terraform
            ;;
        "update")
            check_dependencies
            check_aws_credentials
            update_ecs_service
            ;;
        "all")
            check_dependencies
            check_aws_credentials
            build_image
            ecr_login
            push_image
            apply_terraform
            update_ecs_service
            get_api_url
            ;;
        "url")
            get_api_url
            ;;
        *)
            echo "Uso: $0 {build|push|infra|update|all|url}"
            echo ""
            echo "Comandos:"
            echo "  build   - Construir imagen Docker"
            echo "  push    - Subir imagen a ECR"
            echo "  infra   - Aplicar infraestructura con Terraform"
            echo "  update  - Actualizar servicio ECS"
            echo "  all     - Ejecutar todo el proceso de despliegue"
            echo "  url     - Mostrar URL de la API"
            exit 1
            ;;
    esac
}

# Ejecutar
main "$@"

