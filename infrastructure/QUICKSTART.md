# Guía Rápida de Despliegue

Esta guía te ayudará a desplegar la API de Medical PLS en AWS en menos de 30 minutos.

## Prerrequisitos Rápidos

1. **AWS CLI configurado**
   ```bash
   aws configure
   ```

2. **Docker instalado**
   ```bash
   docker --version
   ```

3. **Terraform instalado**
   ```bash
   terraform version
   ```

## Pasos Rápidos

### 1. Configurar Variables (2 minutos)

```bash
cd infrastructure/terraform
cp terraform.tfvars.example terraform.tfvars
# Edita terraform.tfvars con tus valores (opcional, los defaults funcionan)
```

### 2. Preparar Modelos (5 minutos)

**Los modelos están en DVC S3** (`s3://pds-pls-data-prod/dvcstore`)

**Opción A: Incluir en imagen Docker** (más simple, imagen más grande)

1. Descarga modelos localmente:
   ```bash
   dvc pull models/t5_base models/baseline_classifier
   ```

2. Modifica `api/Dockerfile` y agrega antes del `USER appuser`:
   ```dockerfile
   # Copiar modelos
   COPY --chown=appuser:appuser ../models/t5_base /app/models/t5_base
   COPY --chown=appuser:appuser ../models/baseline_classifier /app/models/baseline_classifier
   ```

**Opción B: Descargar desde DVC S3 al iniciar** (recomendado)

La infraestructura ya está configurada para usar el bucket de DVC. Los modelos se descargarán automáticamente al iniciar el contenedor desde `s3://pds-pls-data-prod/dvcstore`.

### 3. Desplegar (15-20 minutos)

**Linux/Mac:**
```bash
cd infrastructure/scripts
chmod +x deploy.sh
./deploy.sh all
```

**Windows:**
```powershell
cd infrastructure\scripts
.\deploy.ps1 all
```

El script hará todo automáticamente:
- Construir imagen Docker
- Subir a ECR
- Crear infraestructura AWS
- Desplegar servicio ECS

### 4. Obtener URL (1 minuto)

Al finalizar, el script mostrará la URL. También puedes obtenerla:

```bash
cd infrastructure/terraform
terraform output api_url
```

## Verificar Despliegue

```bash
# Health check
curl http://<ALB_DNS>/health

# Documentación
# Abre en navegador: http://<ALB_DNS>/docs
```

## Comandos Útiles

```bash
# Ver logs
aws logs tail /ecs/medical-pls-api --follow

# Actualizar servicio (después de cambios)
cd infrastructure/scripts
./deploy.sh update

# Ver estado
aws ecs describe-services \
  --cluster medical-pls-api-cluster \
  --services medical-pls-api-api-service
```

## Troubleshooting Rápido

**Error: "No se encontró el modelo"**
- Verifica que los modelos estén en `models/t5_base` y `models/baseline_classifier`
- Si usas S3, verifica que `S3_MODELS_BUCKET` esté configurado

**Error: "ECR login failed"**
- Verifica credenciales: `aws sts get-caller-identity`
- Verifica región: `aws configure get region`

**Error: "Terraform apply failed"**
- Verifica permisos IAM
- Revisa los logs de Terraform

## Costos

Configuración básica (1 tarea, 2 vCPU, 4GB):
- **~$70-100/mes** (sin NAT Gateway)
- **~$100-130/mes** (con NAT Gateway)

## Siguiente Paso

Lee [README.md](README.md) para más detalles sobre:
- Configuración avanzada
- Seguridad
- Monitoreo
- Escalado

