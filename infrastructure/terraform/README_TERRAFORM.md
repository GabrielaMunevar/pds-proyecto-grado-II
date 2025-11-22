# Configuración Terraform - Medical PLS API

## Descripción

Esta configuración de Terraform crea toda la infraestructura necesaria para desplegar la API de Medical PLS en AWS usando:

- **ECS Fargate**: Para ejecutar contenedores sin gestionar servidores
- **ECR**: Repositorio de imágenes Docker
- **Application Load Balancer**: Para balancear carga y exponer la API
- **VPC**: Red virtual con subnets públicas/privadas
- **CloudWatch**: Para logs y monitoreo
- **Auto Scaling**: Escalado automático basado en métricas
- **S3** (opcional): Para almacenar modelos

## Uso Básico

### 1. Inicializar Terraform

```bash
cd infrastructure/terraform
terraform init
```

### 2. Revisar Plan

```bash
terraform plan
```

### 3. Aplicar Cambios

```bash
terraform apply
```

### 4. Ver Outputs

```bash
terraform output
```

## Variables Importantes

### Recursos de Computación

- `task_cpu`: CPU para cada tarea ECS (256, 512, 1024, 2048, 4096)
- `task_memory`: Memoria en MB (512, 1024, 2048, 4096, 8192, 16384)
- `desired_count`: Número inicial de tareas

**Recomendaciones:**
- Desarrollo: 1024 CPU, 2048 MB, 1 tarea
- Producción: 2048 CPU, 4096 MB, 2+ tareas

### Escalado

- `min_capacity`: Mínimo de tareas (1 recomendado)
- `max_capacity`: Máximo de tareas (5-10 para producción)
- `cpu_target_value`: Target de CPU para auto-scaling (70% recomendado)

### Red

- `vpc_cidr`: CIDR de la VPC (10.0.0.0/16 por defecto)
- `enable_nat_gateway`: Habilitar NAT Gateway para subnets privadas (aumenta costos)

### Costos

**Sin NAT Gateway:**
- ECS Fargate (2 vCPU, 4GB, 1 tarea): ~$50-70/mes
- ALB: ~$16/mes
- CloudWatch: ~$5-10/mes
- **Total: ~$70-100/mes**

**Con NAT Gateway:**
- Agrega ~$32/mes + datos transferidos
- **Total: ~$100-130/mes**

## Estructura de Recursos

```
VPC (10.0.0.0/16)
├── Internet Gateway
├── Public Subnets (2)
│   └── Application Load Balancer
├── Private Subnets (2, opcional)
│   └── ECS Tasks (si NAT Gateway habilitado)
└── Security Groups
    ├── ALB Security Group
    └── ECS Security Group

ECS Cluster
└── ECS Service
    └── ECS Tasks (Fargate)
        └── Container (pls-api)

ECR Repository
└── Docker Images

S3 Bucket (opcional)
└── models/
    ├── t5_base/
    └── baseline_classifier/

CloudWatch
└── Log Group (/ecs/medical-pls-api)
```

## Comandos Útiles

### Ver Estado

```bash
terraform show
terraform state list
```

### Modificar Recursos

```bash
# Cambiar número de tareas
terraform apply -var="desired_count=3"

# Cambiar recursos de CPU/Memoria
terraform apply -var="task_cpu=4096" -var="task_memory=8192"
```

### Destruir Infraestructura

```bash
terraform destroy
```

**ADVERTENCIA**: Esto eliminará TODOS los recursos creados.

## Outputs Disponibles

- `ecr_repository_url`: URL del repositorio ECR
- `alb_dns_name`: DNS del Load Balancer
- `api_url`: URL completa de la API
- `api_docs_url`: URL de la documentación
- `ecs_cluster_name`: Nombre del cluster ECS
- `s3_models_bucket`: Bucket S3 para modelos (si se creó)

## Troubleshooting

### Error: "Insufficient capacity"

El tipo de instancia Fargate solicitado no está disponible en la región. Solución:
- Esperar unos minutos
- Cambiar a una región diferente
- Reducir `task_cpu` o `task_memory`

### Error: "Resource limit exceeded"

Has alcanzado el límite de recursos en tu cuenta AWS. Solución:
- Solicitar aumento de límites en AWS Support
- Eliminar recursos no utilizados

### Error: "InvalidParameterException"

Parámetros inválidos en la task definition. Verifica:
- `task_cpu` y `task_memory` son valores válidos
- Combinaciones válidas (ver documentación de ECS)

## Mejores Prácticas

1. **Usar Backend Remoto**: Configura un backend S3 para el estado de Terraform
2. **Versionar Estado**: Habilita versionado en el bucket S3
3. **Usar Workspaces**: Para diferentes ambientes (dev, staging, prod)
4. **Revisar Plan**: Siempre revisa `terraform plan` antes de aplicar
5. **Tags**: Todos los recursos tienen tags para mejor organización

## Seguridad

### Recomendaciones

1. **HTTPS**: Configura certificado SSL en el ALB
2. **WAF**: Agrega AWS WAF para protección adicional
3. **VPC Endpoints**: Para acceso privado a S3 (reduce costos)
4. **Secrets Manager**: Para credenciales sensibles
5. **IAM Roles**: Limita permisos al mínimo necesario

## Próximos Pasos

Después de desplegar la infraestructura:

1. Sube los modelos a S3 (si usas S3)
2. Construye y sube la imagen Docker a ECR
3. Actualiza el servicio ECS para usar la nueva imagen
4. Configura un dominio personalizado (opcional)
5. Configura SSL/TLS con ACM

