# 📊 Resumen del Proyecto: PLS Biomédico con DVC y S3

## 🎯 ¿Qué es este Proyecto?

Un sistema de **Machine Learning** que genera **Plain Language Summaries (PLS)** de textos biomédicos complejos, usando:
- **Aprendizaje semi-supervisado** para aprovechar datos no etiquetados
- **DVC** para versionado de datos y experimentos
- **AWS S3** como almacenamiento remoto escalable

---

## 📐 Planteamiento del Proyecto

### Problema a Resolver

Los textos biomédicos (ensayos clínicos, estudios) son muy técnicos y difíciles de entender para el público general. Necesitamos un sistema que los traduzca a **lenguaje sencillo** (PLS).

### Desafío Principal

- **Datos disponibles**: ~182,700 documentos
- **Con PLS (etiquetados)**: Solo ~27% (~49k)
- **Sin PLS (no etiquetados)**: ~73% (~133k)

**Solución**: Usar **semi-supervisión** para aprovechar los datos no etiquetados.

---

## 🏗️ Arquitectura del Sistema

```
                          DATOS CRUDOS (S3)
                                 │
                                 │ dvc pull
                                 ▼
┌────────────────────────────────────────────────────────────┐
│                    data/raw/ (Local)                       │
│  • ClinicalTrials.gov (train/test)                         │
│  • Cochrane (pls/non_pls)                                  │
│  • Pfizer (incluye PDFs)                                   │
│  • Trial Summaries                                         │
│                                                            │
│  Total: 65,941 archivos versionados con DVC               │
└────────────────────────────────────────────────────────────┘
                                 │
                                 │ dvc repro preprocess
                                 ▼
┌────────────────────────────────────────────────────────────┐
│              1. PREPROCESAMIENTO                           │
│  • Limpieza y normalización                                │
│  • Detección de idioma                                     │
│  • Deduplicación                                           │
│  • Filtros de calidad                                      │
│  → Output: data/processed/dataset.parquet                  │
└────────────────────────────────────────────────────────────┘
                                 │
                                 │ dvc repro split
                                 ▼
┌────────────────────────────────────────────────────────────┐
│               2. SPLIT ESTRATIFICADO                       │
│  • Train: 80%                                              │
│  • Dev: 10%                                                │
│  • Test: 10%                                               │
│  → Outputs: train.parquet, dev.parquet, test.parquet      │
└────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │  3. CLASIFICADOR │      │  4. GENERADOR    │
        │   PLS/non-PLS    │      │     DE PLS       │
        │                  │      │                  │
        │  DistilBERT      │      │  BART/T5/LED     │
        │  F1_macro ≥ 0.85 │      │  ROUGE-L ≥ 0.35  │
        │                  │      │  BERTScore ≥ 0.85│
        └──────────────────┘      └──────────────────┘
                                           │
                                           │
                                           ▼
                            ┌──────────────────────────┐
                            │  5. SEMI-SUPERVISADO     │
                            │                          │
                            │  A. Teacher genera PLS   │
                            │     para non_pls         │
                            │  B. Filtros de calidad   │
                            │  C. Re-entrenar con LoRA │
                            │  D. Mejorar modelo       │
                            └──────────────────────────┘
                                           │
                                           ▼
                            ┌──────────────────────────┐
                            │  6. EVALUACIÓN FINAL     │
                            │                          │
                            │  • ROUGE, BERTScore      │
                            │  • Flesch (legibilidad)  │
                            │  • Reportes por fuente   │
                            └──────────────────────────┘
                                           │
                                           │ dvc push
                                           ▼
                             [Modelos y Resultados en S3]
```

---

## 🔄 Rol de DVC en el Proyecto

### ¿Por Qué DVC?

**Problemas que resuelve:**

1. **Datos grandes**: 65k+ archivos no caben en Git
2. **Versionado**: Necesitamos trackear diferentes versiones de datos
3. **Colaboración**: Múltiples personas trabajando con los mismos datos
4. **Reproducibilidad**: Ejecutar el mismo pipeline en diferentes máquinas
5. **Experimentos**: Comparar resultados de diferentes configuraciones

### ¿Cómo Funciona DVC en Este Proyecto?

```
┌─────────────────────────────────────────────────────────────┐
│                        GIT (Código)                         │
│  • Scripts de Python (src/)                                 │
│  • Notebooks (notebooks/)                                   │
│  • Configuración (params.yaml, dvc.yaml)                    │
│  • Archivos DVC (*.dvc) ← Apuntan a datos en S3            │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                     DVC (Orquestación)                      │
│  • Pipeline: preprocess → split → train → evaluate         │
│  • Tracking de métricas                                     │
│  • Cache local (.dvc/cache/)                                │
│  • Gestión de dependencias entre etapas                     │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│              S3 (Almacenamiento de Datos)                   │
│  s3://pds-pls-data-prod/dvcstore/                           │
│  • Datos raw (65,941 archivos)                              │
│  • Datos procesados                                         │
│  • Modelos entrenados                                       │
│  • Outputs y evaluaciones                                   │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de Trabajo con DVC

**Para el líder del proyecto (tú):**
```bash
# 1. Modificar datos o código
# 2. Ejecutar pipeline
dvc repro

# 3. Subir resultados a S3
dvc push

# 4. Commit cambios al código
git add params.yaml dvc.yaml *.dvc
git commit -m "Experiment: improved preprocessing"
git push
```

**Para colaboradores:**
```bash
# 1. Clonar repo
git clone <url>

# 2. Configurar credenciales AWS
.\configure_aws.ps1

# 3. Descargar datos
dvc pull

# 4. Trabajar localmente
dvc repro

# 5. Subir sus cambios
dvc push
git push
```

---

## 📁 Estructura de Archivos: ¿Qué Va Dónde?

### En Git (código y configuración)
```
.
├── src/                      # ✅ Git
│   ├── data/
│   ├── models/
│   └── loops/
├── notebooks/                # ✅ Git
├── docs/                     # ✅ Git
├── params.yaml               # ✅ Git (parámetros)
├── dvc.yaml                  # ✅ Git (pipeline)
├── requirements.txt          # ✅ Git
├── data/raw.dvc              # ✅ Git (puntero a datos)
├── data/processed/*.dvc      # ✅ Git (punteros)
└── models/*.dvc              # ✅ Git (punteros)
```

### En DVC/S3 (datos y modelos grandes)
```
s3://pds-pls-data-prod/dvcstore/
├── files/md5/...             # ❌ Git, ✅ S3
│   ├── [hash]/               # Datos raw
│   ├── [hash]/               # Datos procesados
│   ├── [hash]/               # Modelos entrenados
│   └── [hash]/               # Outputs
```

### Local (mientras trabajas)
```
.
├── data/                     # ❌ Git, ✅ Local cache
│   ├── raw/                  # Desde S3 via dvc pull
│   ├── processed/            # Generado localmente
│   └── outputs/              # Generado localmente
├── models/                   # ❌ Git, ✅ Local cache
│   ├── pls_classifier/       # Generado localmente
│   └── generator_sft/        # Generado localmente
└── .dvc/cache/               # ❌ Git, ✅ Local cache DVC
```

---

## 🎯 Plan de Ejecución (6 Semanas)

### ✅ Completado (Semana 0)
- [x] Configurar DVC con S3
- [x] Subir datos raw a S3 (65,941 archivos)
- [x] Crear documentación para colaboradores
- [x] Configurar `params.yaml` con todos los parámetros

### 📅 Semana 1: Fundamentos
- [ ] EDA completo (distribución, estadísticas)
- [ ] Ejecutar preprocesamiento: `dvc repro preprocess`
- [ ] Validar `data/processed/dataset.parquet`
- [ ] Ejecutar split: `dvc repro split`
- [ ] Validar splits estratificados

**Comandos:**
```bash
jupyter notebook notebooks/01_EDA.ipynb
dvc repro preprocess
dvc repro split
dvc push
```

### 📅 Semana 2: Clasificador
- [ ] Entrenar baseline (TF-IDF + LogReg)
- [ ] Entrenar DistilBERT
- [ ] Calibrar umbral
- [ ] Evaluar por fuente
- [ ] Target: F1_macro ≥ 0.85

**Comandos:**
```bash
dvc repro train_classifier
dvc metrics show
dvc push
```

### 📅 Semana 3: Generador Supervisado
- [ ] Entrenar BART con pares reales
- [ ] Entrenar T5 para comparar
- [ ] Evaluar en dev y test
- [ ] Target: ROUGE-L ≥ 0.35

**Comandos:**
```bash
dvc repro train_generator
dvc metrics show
dvc push
```

### 📅 Semana 4: Semi-Supervisado Ronda 1
- [ ] Implementar filtros
- [ ] Generar PLS sintéticos
- [ ] Re-entrenar con LoRA
- [ ] Evaluar mejora

### 📅 Semana 5: Refinamiento
- [ ] Ronda 2 semi-supervisado
- [ ] Ablation studies
- [ ] Comparar BART vs T5

### 📅 Semana 6: Finalización
- [ ] Evaluación exhaustiva en test
- [ ] Dashboard Streamlit
- [ ] Documentación final
- [ ] Reporte técnico

---

## 💡 Ventajas de Esta Configuración

### Para Ti (Líder del Proyecto)

✅ **Control total**: Todos los datos en S3 bajo tu control  
✅ **Reproducible**: Cualquiera puede ejecutar `dvc repro` y obtener los mismos resultados  
✅ **Trazabilidad**: Sabes exactamente qué versión de datos produjo qué resultados  
✅ **Backup automático**: S3 mantiene respaldo de todo  

### Para Colaboradores

✅ **Setup rápido**: Solo `git clone` + `dvc pull`  
✅ **No necesitan datos en Git**: Descarga selectiva desde S3  
✅ **Independencia**: Cada uno trabaja localmente sin conflictos  
✅ **Sincronización fácil**: `dvc pull` y `dvc push`  

### Para el Proyecto en General

✅ **Escalable**: S3 puede manejar TB de datos  
✅ **Versionado**: Volver a cualquier versión anterior  
✅ **Experimentos**: Comparar múltiples configuraciones  
✅ **Profesional**: Estándar de la industria  

---

## 📊 Ejemplo de Flujo Completo

### Escenario: Un colaborador quiere mejorar el preprocesamiento

```bash
# DÍA 1: Setup inicial
git clone <url>
cd "PROYECTO DE GRADO"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Configurar AWS
# (Solicitar credenciales al admin)
.\configure_aws.ps1

# Descargar datos
dvc pull                      # Descarga data/raw desde S3

# DÍA 2: Exploración
jupyter notebook notebooks/01_EDA.ipynb
# Identificar problema en limpieza de datos

# DÍA 3: Modificación
# Editar src/data/make_dataset.py
# Cambiar parámetros en params.yaml

# Ejecutar solo preprocesamiento
dvc repro preprocess          # Regenera data/processed/dataset.parquet

# Verificar resultado
python -c "import pandas as pd; print(pd.read_parquet('data/processed/dataset.parquet').info())"

# DÍA 4: Validar que funciona todo el pipeline
dvc repro                     # Ejecuta todo desde preprocess

# Si mejora las métricas:
dvc push                      # Sube resultados a S3
git add src/data/make_dataset.py params.yaml data/processed/dataset.parquet.dvc
git commit -m "Improved data cleaning: removed HTML artifacts"
git push

# Otros colaboradores:
git pull                      # Obtienen el código nuevo
dvc pull                      # Obtienen los datos nuevos
```

---

## 🎓 Conceptos Clave para Entender

### 1. ¿Qué es un archivo .dvc?

Un archivo `.dvc` es un **puntero** a datos en S3. Ejemplo de `data/raw.dvc`:

```yaml
outs:
- md5: a3b5c7d9e1f2...
  size: 15728640000
  nfiles: 65941
  path: data/raw
```

Git guarda este archivo pequeño (KB), DVC usa el hash para descargar datos grandes (GB) desde S3.

### 2. ¿Qué es el pipeline de DVC?

`dvc.yaml` define **dependencias** entre etapas:

```yaml
stages:
  preprocess:
    cmd: python src/data/make_dataset.py
    deps:                         # Necesita estos archivos
      - src/data/make_dataset.py
      - data/raw
    params:                       # Usa estos parámetros
      - data
    outs:                         # Genera este output
      - data/processed/dataset.parquet
```

Si cambias `params.yaml` o `make_dataset.py`, DVC sabe que debe re-ejecutar `preprocess`.

### 3. ¿Qué es params.yaml?

Centraliza **todos los hiperparámetros**:

```yaml
data:
  min_chars: 30
classifier:
  lr: 2e-5
generator:
  base_model: "facebook/bart-base"
```

Cambiar un parámetro → DVC re-ejecuta solo las etapas afectadas.

---

## ✅ Checklist Final

- [x] DVC configurado con S3
- [x] 65,941 archivos en S3
- [x] Credenciales AWS funcionando
- [x] Pipeline definido en dvc.yaml
- [x] Parámetros en params.yaml
- [x] Documentación para colaboradores
- [x] Scripts de setup automatizados
- [ ] Primer experimento ejecutado
- [ ] Modelos entrenados
- [ ] Dashboard de visualización

---

## 📚 Documentación Disponible

1. **`docs/GUIA_DESCARGA_DATOS_S3.md`** ⭐: Guía técnica completa paso a paso para descargar datos
2. **`README_SETUP_RAPIDO.md`**: Setup rápido en 5 pasos (~20 min)
3. **`docs/PLAN_PROYECTO.md`**: Plan detallado de 6 semanas con timeline
4. **`RESUMEN_PROYECTO_Y_DVC.md`**: Este archivo - explicación completa del proyecto
5. **`ARQUITECTURA.txt`**: Diseño técnico detallado del sistema
6. **`README.md`**: Overview general del proyecto
7. **`params.yaml`**: Parámetros configurables del proyecto
8. **`dvc.yaml`**: Definición del pipeline de datos

---

## 🚀 Próximos Pasos Inmediatos

1. **Commit la configuración de DVC**
```bash
git add .dvc/ data/.gitignore data/raw.dvc .gitignore
git commit -m "Configure DVC with S3 and track raw data"
git push
```

2. **Ejecutar EDA**
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

3. **Primera ejecución del pipeline**
```bash
dvc repro preprocess
dvc push
```

---

**¡El proyecto está completamente configurado y listo para desarrollar! 🎉**

