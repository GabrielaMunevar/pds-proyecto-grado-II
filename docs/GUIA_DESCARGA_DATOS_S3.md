# Guía Rápida: Descarga de Datos desde S3 con DVC

## Objetivo
Descargar los **65,941 archivos** del proyecto desde Amazon S3 usando DVC.

**Tiempo estimado:** 15-30 minutos

---

## Pre-requisitos
- Python 3.8+
- Git
- 15 GB espacio libre
- Credenciales AWS del proyecto

---

## Configuración Rápida

### 1. Entorno Virtual
```powershell
# Crear y activar entorno virtual
python -m venv venv
.\venv\Scripts\Activate.ps1

# Si hay error de execution policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Instalar Dependencias
```powershell
pip install -r requirements.txt
```

### 3. Verificar DVC
```powershell
dvc version
# Debe mostrar soporte para S3

dvc remote list
# Debe mostrar: myremote s3://pds-pls-data-prod/dvcstore (default)
```

---

## Configuración de Credenciales AWS

### 1. Crear Archivo de Credenciales
Crea `aws_credentials.txt` en la raíz del proyecto:

```ini
[default]
aws_access_key_id=TU_ACCESS_KEY_ID
aws_secret_access_key=TU_SECRET_ACCESS_KEY
aws_session_token=TU_SESSION_TOKEN
s3: pds-pls-data-prod
```

### 2. Configurar Variables de Entorno
```powershell
.\configure_aws.ps1
```

**Salida esperada:**
```
[OK] Credenciales configuradas correctamente para esta sesion!
[OK] Conexion exitosa con AWS!
```

---

## Descarga de Datos

### 1. Verificar Estado
```powershell
dvc status
```

### 2. Descargar Datos
```powershell
dvc pull
```

**Proceso:**
- Collecting: ~17 segundos
- Downloading: 20 minutos - 3 horas (depende de internet)
- Building workspace: ~2 minutos

### 3. Verificar Descarga
```powershell
dvc status
# Debe mostrar: "Data and pipelines are up to date"

# Contar archivos
(Get-ChildItem -Path data/raw -Recurse -File).Count
# Debe mostrar: 65941
```

---

## Troubleshooting

### Error "Access Denied"
- Verificar credenciales en `aws_credentials.txt`
- Re-ejecutar `.\configure_aws.ps1`
- Las credenciales temporales expiran cada 4-12 horas

### Descarga Muy Lenta
```powershell
# Aumentar conexiones paralelas
dvc remote modify myremote jobs 8
dvc pull
```


