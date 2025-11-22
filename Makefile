# Makefile para el proyecto PLS Biomédico
# Facilita la ejecución de comandos comunes

.PHONY: help install setup-dvc pull push repro clean test lint format

# Variables
PYTHON := python
PIP := pip
DVC := dvc
S3_BUCKET := pds-pls-data-prod
REGION := us-east-1

help:
	@echo "Comandos disponibles:"
	@echo ""
	@echo "  make install      - Instalar dependencias"
	@echo "  make setup-dvc    - Configurar DVC con S3"
	@echo "  make pull         - Descargar datos desde S3"
	@echo "  make push         - Subir datos/modelos a S3"
	@echo "  make upload-s3    - Subir datos directamente a S3 (sin DVC)"
	@echo "  make repro        - Ejecutar pipeline completo"
	@echo "  make repro-f      - Ejecutar pipeline forzando re-ejecución"
	@echo ""
	@echo "  make preprocess   - Solo preprocesamiento"
	@echo "  make split        - Solo split de datos"
	@echo "  make train-cls    - Entrenar clasificador"
	@echo "  make train-gen    - Entrenar generador"
	@echo "  make evaluate     - Evaluar modelos"
	@echo ""
	@echo "  make metrics      - Ver métricas"
	@echo "  make status       - Ver estado de DVC"
	@echo "  make clean        - Limpiar archivos temporales"
	@echo "  make clean-all    - Limpiar todo (incluye cache DVC)"
	@echo ""
	@echo "  make lint         - Verificar código con flake8"
	@echo "  make format       - Formatear código con black"
	@echo "  make test         - Ejecutar tests"

# Instalación
install:
	@echo "Instalando dependencias..."
	$(PIP) install -r requirements.txt
	@echo "Dependencias instaladas"

install-dev:
	@echo "📦 Instalando dependencias de desarrollo..."
	$(PIP) install -r requirements.txt
	$(PIP) install black flake8 pytest ipykernel
	@echo "Dependencias instaladas"

# DVC Setup
setup-dvc:
	@echo "Configurando DVC con S3..."
	$(DVC) remote add -d myremote s3://$(S3_BUCKET)/dvcstore --force
	$(DVC) remote modify myremote region $(REGION)
	@echo "DVC configurado"
	@$(DVC) remote list -v

# DVC Operations
pull:
	@echo "Descargando datos desde S3..."
	$(DVC) pull

push:
	@echo "Subiendo datos/modelos a S3..."
	$(DVC) push

# Upload directamente a S3 (sin DVC)
upload-s3:
	@echo "Subiendo datos directamente a S3..."
	@if [ -f "scripts/upload_data_to_s3.ps1" ]; then \
		powershell -ExecutionPolicy Bypass -File scripts/upload_data_to_s3.ps1 -BucketName $(S3_BUCKET) -Region $(REGION); \
	elif [ -f "scripts/upload_data_to_s3.sh" ]; then \
		bash scripts/upload_data_to_s3.sh $(S3_BUCKET) $(REGION); \
	else \
		echo "ERROR: Script de subida no encontrado"; \
	fi

status:
	@echo "Estado de DVC:"
	$(DVC) status

# Pipeline
repro:
	@echo "🔄 Ejecutando pipeline DVC..."
	$(DVC) repro

repro-f:
	@echo "🔄 Ejecutando pipeline DVC (forzado)..."
	$(DVC) repro --force

# Etapas individuales
preprocess:
	@echo "🔄 Ejecutando preprocesamiento..."
	$(DVC) repro preprocess

split:
	@echo "🔄 Ejecutando split..."
	$(DVC) repro split

train-cls:
	@echo "🔄 Entrenando clasificador..."
	$(DVC) repro train_classifier

train-gen:
	@echo "🔄 Entrenando generador..."
	$(DVC) repro train_generator

semi-supervised:
	@echo "🔄 Ejecutando bucle semi-supervisado..."
	$(DVC) repro semi_supervised_loop

evaluate:
	@echo "🔄 Evaluando modelos..."
	$(DVC) repro evaluate

# Métricas
metrics:
	@echo "Métricas actuales:"
	$(DVC) metrics show

metrics-diff:
	@echo "Diferencias de métricas:"
	$(DVC) metrics diff

# Limpieza
clean:
	@echo "Limpiando archivos temporales..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.log" -delete 2>/dev/null || true
	@echo "Limpieza completada"

clean-all: clean
	@echo "Limpiando todo (incluye cache DVC)..."
	$(DVC) gc --workspace --cloud
	@echo "Limpieza completa"

# Calidad de código
lint:
	@echo "Verificando código con flake8..."
	flake8 src/ --max-line-length=120 --ignore=E203,W503

format:
	@echo "Formateando código con black..."
	black src/ --line-length=120

# Tests
test:
	@echo "🧪 Ejecutando tests..."
	pytest tests/ -v

# Notebooks
jupyter:
	@echo "📓 Iniciando Jupyter Lab..."
	jupyter lab

notebook:
	@echo "📓 Iniciando Jupyter Notebook..."
	jupyter notebook notebooks/

# Git + DVC workflow
commit:
	@echo "Agregando cambios a Git..."
	git add params.yaml dvc.yaml dvc.lock src/
	@echo "Escribe el mensaje de commit:"
	@read -p "Message: " msg; git commit -m "$$msg"

sync: push
	@echo "🔄 Sincronizando con remoto..."
	git push

# Info
info:
	@echo "📋 Información del proyecto:"
	@echo ""
	@echo "Python: $$(python --version)"
	@echo "DVC: $$(dvc version)"
	@echo "Git: $$(git --version)"
	@echo ""
	@echo "Remotes DVC:"
	@$(DVC) remote list -v
	@echo ""
	@echo "Remotes Git:"
	@git remote -v

# Verificación completa
check: lint test status metrics
	@echo "Verificación completa"
