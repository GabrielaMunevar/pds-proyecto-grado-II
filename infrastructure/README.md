# Infraestructura AWS para Medical PLS API

Esta carpeta contiene la configuración de infraestructura como código para desplegar la API de Medical PLS en AWS usando Docker, ECS Fargate, y Terraform.

## Arquitectura

La infraestructura incluye:

- **VPC**: Red virtual con subnets públicas y privadas
- **ECS Fargate**: Contenedores sin servidores gestionados
- **ECR**: Repositorio de imágenes Docker
- **Application Load Balancer**: Balanceador de carga para la API
- **CloudWatch**: Logs y monitoreo
- **Auto Scaling**: Escalado automático basado en CPU
- **S3** (opcional): Almacenamiento para modelos

## Prerrequisitos

1. **AWS CLI** instalado y configurado
   ```bash
   aws configure
   ```

2. **Docker** instalado y funcionando
   ```bash
   docker --version
   ```

3. **Terraform** >= 1.0 instalado
   ```bash
   terraform version
   ```

4. **jq** (para scripts bash, opcional)
   ```bash
   # Linux/Mac
   sudo apt-get install jq  # o brew install jq
   ```

## Estructura de Archivos

```
infrastructure/
├── terraform/
│   ├── main.tf              # Recursos principales
│   ├── variables.tf        # Variables
│   ├── outputs.tf          # Outputs
│   └── terraform.tfvars.example  # Ejemplo de variables
├── scripts/
│   ├── deploy.sh           # Script de despliegue (Linux/Mac)
│   └── deploy.ps1          # Script de despliegue (Windows)
└── README_ENV_VARS.md      # Documentación de variables de entorno
```

**Importante**: Ver [README_ENV_VARS.md](README_ENV_VARS.md) para entender cómo se configuran las variables de entorno en producción.

## Configuración Inicial

### 1. Configurar Variables de Terraform

Copia el archivo de ejemplo y personaliza:

```bash
cd infrastructure/terraform
cp terraform.tfvars.example terraform.tfvars
```

Edita `terraform.tfvars` con tus valores:

```hcl
aws_region = "us-east-1"
environment = "prod"
project_name = "medical-pls-api"

# Ajusta según tus necesidades
task_cpu = 2048
task_memory = 4096
desired_count = 1
```

### 2. Preparar Modelos

**Los modelos están almacenados en DVC usando S3** (`s3://pds-pls-data-prod/dvcstore`).

Tienes dos opciones:

#### Opción A: Incluir modelos en la imagen Docker (más simple)

1. Descarga los modelos localmente usando DVC:
   ```bash
   # Desde la raíz del proyecto
   dvc pull models/t5_base models/baseline_classifier
   ```

2. Modifica el Dockerfile para copiar los modelos:
   ```dockerfile
   COPY models/ /app/models/
   ```

**Nota**: Esto aumentará significativamente el tamaño de la imagen.

#### Opción B: Descargar desde DVC S3 al iniciar (recomendado)

La infraestructura ya está configurada para usar el bucket de DVC. Los modelos se descargarán automáticamente desde `s3://pds-pls-data-prod/dvcstore` al iniciar el contenedor.

**Ventajas:**
- Imagen Docker más pequeña
- Actualización de modelos sin reconstruir imagen
- Usa el mismo bucket S3 que DVC
- Acceso privado desde VPC (VPC Endpoint)

## Despliegue

### Método 1: Script Automatizado (Recomendado)

#### Linux/Mac:

```bash
cd infrastructure/scripts
chmod +x deploy.sh
./deploy.sh all
```

#### Windows (PowerShell):

```powershell
cd infrastructure/scripts
.\deploy.ps1 all
```

El script ejecutará:
1. Build de la imagen Docker
2. Login a ECR
3. Push de la imagen
4. Aplicación de infraestructura con Terraform
5. Actualización del servicio ECS

### Método 2: Pasos Manuales

#### 1. Construir Imagen Docker

```bash
cd api
docker build -t medical-pls-api:latest .
```

#### 2. Autenticarse con ECR

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com
```

#### 3. Subir Imagen a ECR

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
ECR_URL=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

docker tag medical-pls-api:latest $ECR_URL/medical-pls-api-api:latest
docker push $ECR_URL/medical-pls-api-api:latest
```

#### 4. Aplicar Infraestructura con Terraform

```bash
cd infrastructure/terraform
terraform init
terraform plan
terraform apply
```

#### 5. Actualizar Servicio ECS

```bash
aws ecs update-service \
  --cluster medical-pls-api-cluster \
  --service medical-pls-api-api-service \
  --force-new-deployment \
  --region us-east-1
```

## Descarga de Modelos desde DVC S3

**La descarga automática desde DVC S3 ya está implementada.** El código en `api/main.py` y `api/utils_dvc.py` se encarga de:

1. Verificar si las variables de entorno `DVC_S3_BUCKET` y `DVC_S3_PREFIX` están configuradas
2. Intentar descargar modelos desde el bucket de DVC usando `dvc pull` (si DVC está disponible)
3. Usar modelos locales si ya están descargados

**Configuración automática:**
- La infraestructura de Terraform configura automáticamente:
  - `DVC_S3_BUCKET = "pds-pls-data-prod"`
  - `DVC_S3_PREFIX = "dvcstore"`
- Los permisos IAM permiten acceso de lectura al bucket de DVC

**Nota:** Para que `dvc pull` funcione en el contenedor, necesitas:
1. Instalar DVC en el Dockerfile: `RUN pip install dvc[s3]`
2. Copiar `.dvc/config` al contenedor
3. O usar descarga directa desde S3 (implementación alternativa en `utils_dvc.py`)

**Recomendación:** Para producción, es más eficiente incluir los modelos en la imagen Docker después de hacer `dvc pull` localmente, o configurar una descarga directa desde S3 sin usar DVC en el contenedor.

## Monitoreo

### Ver Logs

```bash
# CloudWatch Logs
aws logs tail /ecs/medical-pls-api --follow --region us-east-1

# O desde la consola de AWS
```

### Ver Estado del Servicio

```bash
aws ecs describe-services \
  --cluster medical-pls-api-cluster \
  --services medical-pls-api-api-service \
  --region us-east-1
```

### Ver Métricas

- CloudWatch Console → ECS → Clusters → medical-pls-api-cluster
- CloudWatch Container Insights (si está habilitado)

## Escalado

El auto-scaling está configurado para escalar basado en CPU:

- **Mínimo**: 1 tarea
- **Máximo**: 5 tareas (configurable)
- **Target**: 70% CPU

Para ajustar, modifica las variables en `terraform.tfvars`:

```hcl
min_capacity = 1
max_capacity = 10
cpu_target_value = 70.0
```

Luego aplica:

```bash
terraform apply
```

## Costos Estimados

Para una configuración básica (1 tarea, 2 vCPU, 4GB RAM):

- **ECS Fargate**: ~$50-70/mes (dependiendo del uso)
- **ALB**: ~$16/mes
- **ECR**: Gratis (primeros 500MB/mes)
- **CloudWatch**: ~$5-10/mes
- **NAT Gateway** (si se usa): ~$32/mes + datos

**Total estimado**: ~$70-130/mes (sin NAT Gateway)

## Troubleshooting

### La API no responde

1. Verifica que el servicio ECS esté corriendo:
   ```bash
   aws ecs describe-services --cluster medical-pls-api-cluster --services medical-pls-api-api-service
   ```

2. Revisa los logs:
   ```bash
   aws logs tail /ecs/medical-pls-api --follow
   ```

3. Verifica el health check:
   ```bash
   curl http://<ALB_DNS>/health
   ```

### Error al cargar modelos

1. Verifica que los modelos estén en la ruta correcta
2. Revisa los permisos de IAM para acceder a S3 (si usas S3)
3. Verifica los logs del contenedor

### Imagen muy grande

1. Usa multi-stage builds (ya incluido en el Dockerfile)
2. Considera usar S3 para modelos en lugar de incluirlos en la imagen
3. Usa compresión de imágenes

## Limpieza

Para eliminar toda la infraestructura:

```bash
cd infrastructure/terraform
terraform destroy
```

**Advertencia**: Esto eliminará todos los recursos creados.

## Seguridad

### Mejoras Recomendadas para Producción

1. **HTTPS**: Configura un certificado SSL en el ALB usando ACM
2. **WAF**: Agrega AWS WAF para protección adicional
3. **VPC Endpoints**: Para acceso privado a S3 (reduce costos de NAT)
4. **Secrets Manager**: Para credenciales y configuraciones sensibles
5. **IAM Roles**: Limita permisos al mínimo necesario
6. **Security Groups**: Restringe acceso solo a lo necesario

## Soporte

Para problemas o preguntas, consulta:
- [Documentación de Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Documentación de ECS](https://docs.aws.amazon.com/ecs/)
- [Documentación de ECR](https://docs.aws.amazon.com/ecr/)

