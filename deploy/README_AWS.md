# Deploy PLS Generator Dashboard to AWS

## 🎯 Overview

Este documento describe cómo desplegar el dashboard PLS Generator en AWS EC2.

## 📋 Prerequisites

1. **AWS Account** con permisos para crear instancias EC2
2. **Modelo entrenado** en `models/t5_generator/`
3. **Datos** en `data/processed/synthetic_pairs/`

## 🚀 Quick Start

### Opción A: Automated Deployment (Recommended)

```bash
# 1. Crear instancia EC2
# - AMI: Ubuntu 22.04 LTS
# - Instance Type: t2.small (2GB RAM) o t2.medium (4GB RAM)
# - Security Group: Allow port 8501

# 2. Conectar a la instancia
ssh -i your-key.pem ubuntu@ec2-XX-XXX-XXX-XX.compute-1.amazonaws.com

# 3. Clonar repositorio
git clone <your-repo-url>
cd <project-dir>

# 4. Ejecutar script de deployment
chmod +x deploy/deploy_aws.sh
./deploy/deploy_aws.sh
```

### Opción B: Manual Deployment

```bash
# 1. Instalar dependencias
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# 2. Crear virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Instalar paquetes
pip install streamlit torch transformers

# 4. Ejecutar dashboard
streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

## ⚙️ Configuration

### Security Group Settings

En AWS Console → EC2 → Security Groups:

```
Inbound Rules:
  Type: Custom TCP
  Port: 8501
  Source: 0.0.0.0/0 (o tu IP específica)
  Description: Streamlit Dashboard
```

### Environment Variables

Crear `.env` o exportar:

```bash
export MODEL_PATH="models/t5_generator"
export DATA_PATH="data/processed"
```

## 📊 Resource Requirements

### Minimum
- **Instance Type**: t2.small
- **RAM**: 2GB
- **Storage**: 20GB
- **Cost**: ~$15/month

### Recommended
- **Instance Type**: t2.medium
- **RAM**: 4GB
- **Storage**: 30GB
- **Cost**: ~$30/month

### Production
- **Instance Type**: t3.large
- **RAM**: 8GB
- **Storage**: 50GB
- **Cost**: ~$60/month

## 🔒 Security

### 1. HTTPS (Production)

```bash
# Instalar Nginx
sudo apt-get install nginx

# Configurar SSL con Let's Encrypt
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 2. Authentication

Agregar autenticación en `src/dashboard/app.py`:

```python
import streamlit_authenticator as stauth

# Add authentication
authenticator = stauth.Authenticate(
    {'usernames': {'user': 'hashed_password'}},
    'cookie_name',
    'signature_key'
)
```

## 🐛 Troubleshooting

### Issue: Port already in use
```bash
sudo lsof -i :8501
sudo kill -9 <PID>
```

### Issue: Model not found
```bash
# Verificar estructura
ls -la models/t5_generator/
```

### Issue: Out of memory
```bash
# Usar modelo más pequeño o aumentar instancia
# Configurar swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 📈 Monitoring

### System Resource Usage
```bash
# CPU y RAM
htop

# Disk
df -h

# GPU (si aplica)
nvidia-smi
```

### Application Logs
```bash
# Ver logs
sudo journalctl -u streamlit -f

# Ver últimas 50 líneas
sudo journalctl -u streamlit -n 50
```

## 💰 Cost Optimization

1. **Stop instance** cuando no se use
2. **Use Spot Instances** para pruebas
3. **Reserved Instances** para producción
4. **Auto-scaling** si hay alta demanda

## 🔄 Updates

```bash
# Actualizar código
git pull origin main

# Reiniciar servicio
sudo systemctl restart streamlit
```

## 📞 Support

Para problemas o preguntas:
- Revisar logs: `sudo journalctl -u streamlit`
- Verificar modelo: `python -c "from transformers import T5Tokenizer; print('OK')"`
- Test dashboard local: `streamlit run src/dashboard/app.py`

## 🌐 Access

After deployment:
```
http://<EC2-PUBLIC-IP>:8501
```

Example:
```
http://54.123.45.67:8501
```


