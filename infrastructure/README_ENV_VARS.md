# Configuración de Variables de Entorno en Producción

Este documento explica cómo se configuran las variables de entorno en diferentes entornos de despliegue.

## Resumen

Las variables de entorno se configuran de manera diferente según el entorno:

- **Desarrollo Local**: Archivo `.env` en la raíz del proyecto
- **Docker Compose**: Sección `environment` en `docker-compose.yml`
- **Producción (AWS ECS)**: Variables en Terraform (`main.tf` y `terraform.tfvars`)

## Desarrollo Local

### Ubicación
Archivo `.env` en la raíz del proyecto.

### Cómo funciona
El código en `api/config.py` busca automáticamente el archivo `.env` y carga las variables.

### Ejemplo
```bash
# .env (en la raíz del proyecto)
LOG_LEVEL=INFO
DVC_S3_BUCKET=pds-pls-data-prod
DVC_S3_PREFIX=dvcstore
```

## Docker Compose (Desarrollo)

### Ubicación
Archivo `api/docker-compose.yml`, sección `environment`.

### Cómo funciona
Docker Compose pasa las variables directamente al contenedor.

### Ejemplo
```yaml
services:
  pls-api:
    environment:
      - LOG_LEVEL=INFO
      - DVC_S3_BUCKET=pds-pls-data-prod
      - DVC_S3_PREFIX=dvcstore
```

## Producción (AWS ECS con Terraform)

### Ubicación
Variables configuradas en:
1. `infrastructure/terraform/main.tf` - Definición de variables de entorno en la task definition
2. `infrastructure/terraform/variables.tf` - Variables de Terraform
3. `infrastructure/terraform/terraform.tfvars` - Valores específicos del entorno

### Cómo funciona
Terraform crea la ECS Task Definition con todas las variables de entorno configuradas. Estas se pasan al contenedor cuando ECS inicia el servicio.

### Variables Configuradas en Terraform

Las siguientes variables se configuran automáticamente en la task definition:

| Variable | Valor | Fuente |
|----------|-------|--------|
| `PYTHONUNBUFFERED` | `1` | Hardcoded |
| `LOG_LEVEL` | `var.log_level` | `terraform.tfvars` |
| `DVC_S3_BUCKET` | `var.dvc_s3_bucket` | `terraform.tfvars` |
| `DVC_S3_PREFIX` | `var.dvc_s3_prefix` | `terraform.tfvars` |
| `AWS_REGION` | `var.aws_region` | `terraform.tfvars` |
| `API_HOST` | `0.0.0.0` | Hardcoded |
| `API_PORT` | `8000` | Hardcoded |
| `CORS_ORIGINS` | `var.cors_origins` | `terraform.tfvars` |
| `MAX_LENGTH` | `var.max_length` | `terraform.tfvars` |
| `NUM_BEAMS` | `var.num_beams` | `terraform.tfvars` |
| `MAX_INPUT_LENGTH` | `var.max_input_length` | `terraform.tfvars` |
| `CHUNK_SIZE` | `var.chunk_size` | `terraform.tfvars` |
| `CHUNK_OVERLAP` | `var.chunk_overlap` | `terraform.tfvars` |
| `DEBUG` | `var.debug` | `terraform.tfvars` |

### Configuración en terraform.tfvars

```hcl
# infrastructure/terraform/terraform.tfvars

# DVC S3 Configuration
dvc_s3_bucket = "pds-pls-data-prod"
dvc_s3_prefix = "dvcstore"
aws_region = "us-east-1"

# Logging
log_level = "INFO"

# API Configuration
cors_origins = "*"
max_length = 256
num_beams = 4
max_input_length = 512
chunk_size = 400
chunk_overlap = 50
debug = false
```

### Variables Adicionales

Si necesitas agregar variables adicionales que no están en la lista estándar, puedes usar `additional_env_vars`:

```hcl
# infrastructure/terraform/terraform.tfvars
additional_env_vars = [
  {
    name  = "CUSTOM_VAR"
    value = "custom_value"
  },
  {
    name  = "ANOTHER_VAR"
    value = "another_value"
  }
]
```

## Credenciales de AWS

### En Producción (ECS)
Las credenciales de AWS **NO** se pasan como variables de entorno. En su lugar:

1. **ECS Task Role**: El contenedor usa el IAM Role asignado a la task (`aws_iam_role.ecs_task`)
2. **Credenciales temporales**: AWS proporciona credenciales temporales automáticamente
3. **Sin archivos .env**: No se copian archivos `.env` al contenedor en producción

### En Desarrollo Local
Si necesitas credenciales de AWS localmente, puedes:
- Usar `aws configure` (se usan automáticamente)
- O agregar al `.env`:
  ```bash
  AWS_ACCESS_KEY_ID=your-key
  AWS_SECRET_ACCESS_KEY=your-secret
  AWS_SESSION_TOKEN=your-token  # Si usas sesiones temporales
  AWS_DEFAULT_REGION=us-east-1
  ```

## Flujo de Configuración

### Desarrollo Local
```
.env (raíz) → api/config.py → Carga automática → Variables disponibles
```

### Docker Compose
```
docker-compose.yml → Docker → Contenedor → Variables disponibles
```

### Producción (AWS ECS)
```
terraform.tfvars → Terraform → ECS Task Definition → Contenedor → Variables disponibles
```

## Verificación

### Desarrollo Local
```bash
# Verificar que las variables se cargan
cd api
python -c "from config import settings; print(settings.LOG_LEVEL)"
```

### Docker Compose
```bash
# Verificar variables en el contenedor
docker exec pls-api-dev env | grep LOG_LEVEL
```

### Producción (ECS)
```bash
# Verificar en CloudWatch Logs
aws logs tail /aws/ecs/medical-pls-api --follow

# O verificar en la consola de AWS:
# ECS → Clusters → Tasks → Ver detalles → Environment variables
```

## Seguridad

### Buenas Prácticas

1. **No commitear `.env`**: El archivo `.env` está en `.gitignore`
2. **Usar IAM Roles en producción**: No pasar credenciales como variables de entorno
3. **Usar AWS Secrets Manager**: Para secretos sensibles (opcional, no implementado actualmente)
4. **Variables en Terraform**: Usar `terraform.tfvars` que NO se commitea

### Importante

- El archivo `.env` local **NO se copia** al contenedor Docker en producción
- Las credenciales de AWS en `.env` son solo para desarrollo local
- En producción, ECS usa IAM Roles automáticamente

## Troubleshooting

### Las variables no se cargan en producción

1. Verifica que `terraform.tfvars` tenga los valores correctos
2. Ejecuta `terraform plan` para ver qué variables se pasarán
3. Verifica en CloudWatch Logs que el contenedor recibe las variables
4. Revisa la task definition en la consola de AWS ECS

### Las variables no se cargan localmente

1. Verifica que el archivo `.env` esté en la raíz del proyecto
2. Verifica que no tenga errores de sintaxis
3. Revisa los logs de la API al iniciar (debe mostrar "Loading environment variables from...")

