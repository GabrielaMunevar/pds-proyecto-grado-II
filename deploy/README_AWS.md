# Desplegar Dashboard Generador PLS en AWS

## Resumen

Este documento describe cómo desplegar el dashboard PLS Generator en AWS EC2.

## Prerrequisitos

1. **Cuenta AWS** con permisos para crear instancias EC2
2. **Modelo entrenado** en `models/t5_generator/`
3. **Datos** en `data/processed/synthetic_pairs/`

## Inicio Rápido

### Opción A: Despliegue Automatizado (Recomendado)

```bash
# 1. Crear instancia EC2
# - AMI: Ubuntu 22.04 LTS
# - Tipo de Instancia: t2.small (2GB RAM) o t2.medium (4GB RAM)
# - Grupo de Seguridad: Permitir puerto 8501

# 2. Conectar a la instancia
ssh -i your-key.pem ubuntu@ec2-XX-XXX-XXX-XX.compute-1.amazonaws.com

# 3. Clonar repositorio
git clone <your-repo-url>
cd <project-dir>

# 4. Ejecutar script de despliegue
chmod +x deploy/deploy_aws.sh
./deploy/deploy_aws.sh
```

### Opción B: Despliegue Manual

```bash
# 1. Instalar dependencias
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar paquetes
pip install streamlit torch transformers

# 4. Ejecutar dashboard
streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

## Configuración

### Configuración de Grupo de Seguridad

En AWS Console → EC2 → Security Groups:

```
Reglas de Entrada:
  Tipo: Custom TCP
  Puerto: 8501
  Origen: 0.0.0.0/0 (o tu IP específica)
  Descripción: Streamlit Dashboard
```

### Variables de Entorno

Crear `.env` o exportar:

```bash
export MODEL_PATH="models/t5_generator"
export DATA_PATH="data/processed"
```

## Requisitos de Recursos

### Mínimo
- **Tipo de Instancia**: t2.small
- **RAM**: 2GB
- **Almacenamiento**: 20GB
- **Costo**: ~$15/mes

### Recomendado
- **Tipo de Instancia**: t2.medium
- **RAM**: 4GB
- **Almacenamiento**: 30GB
- **Costo**: ~$30/mes

### Producción
- **Tipo de Instancia**: t3.large
- **RAM**: 8GB
- **Almacenamiento**: 50GB
- **Costo**: ~$60/mes

## Seguridad

### 1. HTTPS (Producción)

```bash
# Instalar Nginx
sudo apt-get install nginx

# Configurar SSL con Let's Encrypt
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 2. Autenticación

Agregar autenticación en `src/dashboard/app.py`:

```python
import streamlit_authenticator as stauth

# Agregar autenticación
authenticator = stauth.Authenticate(
    {'usernames': {'user': 'hashed_password'}},
    'cookie_name',
    'signature_key'
)
```

## Solución de Problemas

### Problema: Puerto ya en uso
```bash
sudo lsof -i :8501
sudo kill -9 <PID>
```

### Problema: Modelo no encontrado
```bash
# Verificar estructura
ls -la models/t5_generator/
```

### Problema: Sin memoria
```bash
# Usar modelo más pequeño o aumentar instancia
# Configurar swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Monitoreo

### Uso de Recursos del Sistema
```bash
# CPU y RAM
htop

# Disco
df -h

# GPU (si aplica)
nvidia-smi
```

### Logs de la Aplicación
```bash
# Ver logs
sudo journalctl -u streamlit -f

# Ver últimas 50 líneas
sudo journalctl -u streamlit -n 50
```

## Optimización de Costos

1. **Detener instancia** cuando no se use
2. **Usar Spot Instances** para pruebas
3. **Reserved Instances** para producción
4. **Auto-escalado** si hay alta demanda

## Actualizaciones

```bash
# Actualizar código
git pull origin main

# Reiniciar servicio
sudo systemctl restart streamlit
```

## Soporte

Para problemas o preguntas:
- Revisar logs: `sudo journalctl -u streamlit`
- Verificar modelo: `python -c "from transformers import T5Tokenizer; print('OK')"`
- Probar dashboard local: `streamlit run src/dashboard/app.py`

## Acceso

Después del despliegue:
```
http://<EC2-PUBLIC-IP>:8501
```

Ejemplo:
```
http://54.123.45.67:8501
```
