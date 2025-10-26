# 📋 Plan de Proyecto: Sistema PLS Biomédico con Semi-Supervisión

## 🎯 Objetivo General
Desarrollar un sistema de Machine Learning que genere Plain Language Summaries (PLS) de textos biomédicos complejos, usando técnicas de aprendizaje semi-supervisado para aprovechar el gran volumen de datos no etiquetados.

---

## 📊 Contexto del Proyecto

### Datos Disponibles
- **Total**: ~182.7k documentos
- **Con PLS (etiquetados)**: ~27% (≈49k documentos)
- **Sin PLS (no etiquetados)**: ~73% (≈133k documentos)

### Fuentes de Datos
1. **ClinicalTrials.gov**: Ensayos clínicos
2. **Cochrane**: Revisiones sistemáticas
3. **Pfizer**: Estudios clínicos (incluye PDFs)
4. **Trial Summaries**: Resúmenes de ensayos

### Desafío Principal
Gran desbalance entre datos etiquetados y no etiquetados → **Necesidad de semi-supervisión**

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     DATOS CRUDOS (S3)                       │
│  ClinicalTrials │ Cochrane │ Pfizer │ Trial Summaries      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               1. PREPROCESAMIENTO                           │
│  • Normalización y limpieza                                 │
│  • Detección de idioma (filtrar inglés)                     │
│  • Deduplicación por hash                                   │
│  • Filtros de calidad (longitud, formato)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               2. SPLIT ESTRATIFICADO                        │
│  Train: 80% │ Dev: 10% │ Test: 10%                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          3. CLASIFICADOR PLS/non-PLS                        │
│  Modelo: DistilBERT / BioBERT                               │
│  Objetivo: Detectar si texto ya es PLS                      │
│  Métricas: F1_macro ≥ 0.85                                  │
│  • Maneja desbalance con focal loss                         │
│  • Evita generar cuando ya es PLS (gate)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
         ▼                                   ▼
    [ES PLS]                           [NO ES PLS]
  Skip generación                    Pasar a generador
         │                                   │
         └─────────────────┬─────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          4. GENERADOR PLS (Supervisado)                     │
│  Modelo: BART / T5 / LED                                    │
│  Datos: Solo pares reales (texto → PLS)                     │
│  Training: Full Fine-Tuning o LoRA                          │
│  Métricas:                                                  │
│    • ROUGE-L ≥ 0.35-0.40                                    │
│    • BERTScore F1 ≥ 0.85                                    │
│    • ΔFlesch ≥ +15                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│       5. BUCLE SEMI-SUPERVISADO (Opcional)                  │
│                                                             │
│  A. Teacher Model genera PLS para non-PLS                   │
│     ↓                                                       │
│  B. Filtros de Calidad (automáticos)                        │
│     • Legibilidad: Flesch ≥ 60, Δ ≥ +15                    │
│     • Factualidad: BERTScore ≥ 0.85                         │
│     • Sin números nuevos                                    │
│     • Ratio de compresión razonable                         │
│     ↓                                                       │
│  C. Pares Sintéticos Aceptados                              │
│     ↓                                                       │
│  D. Re-entrenamiento con Mix 70:30 (real:sintético)         │
│     • Usar LoRA para eficiencia                             │
│     • Early stopping en dev                                 │
│     ↓                                                       │
│  E. Evaluación → ¿Mejora? → Siguiente ronda                │
│                                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               6. EVALUACIÓN FINAL                           │
│  Con ground truth:                                          │
│    • ROUGE-L, BERTScore, Flesch, Compresión                 │
│  Sin ground truth:                                          │
│    • Legibilidad, longitud, heurísticas                     │
│  Reportes:                                                  │
│    • Por fuente (Cochrane, Pfizer, etc.)                    │
│    • Comparación de modelos                                 │
│    • Ejemplos cualitativos                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         7. DASHBOARD DE VISUALIZACIÓN                       │
│  • Comparador de modelos                                    │
│  • Métricas interactivas                                    │
│  • Ejemplos originales vs generados                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📅 Plan de Ejecución (6 Semanas)

### **Semana 1: Fundamentos y Datos**
- [x] Configurar DVC con S3
- [ ] Ejecutar preprocesamiento completo
- [ ] Análisis exploratorio de datos (EDA)
- [ ] Validar distribución por fuente
- [ ] Confirmar splits estratificados

**Entregables**:
- `data/processed/dataset.parquet`
- `data/processed/{train,dev,test}.parquet`
- Notebook `notebooks/01_EDA.ipynb` actualizado

---

### **Semana 2: Clasificador PLS/non-PLS**
- [ ] Entrenar baseline (TF-IDF + Logistic Regression)
- [ ] Entrenar DistilBERT con focal loss
- [ ] Calibrar umbral de decisión
- [ ] Evaluación por fuente (matriz de confusión)
- [ ] Tunear hiperparámetros

**Entregables**:
- `models/pls_classifier/`
- `data/processed/classified.parquet`
- Reporte: `data/evaluation/classifier_report.json`

**Métricas Objetivo**:
- F1_macro ≥ 0.85
- Recall de PLS (minoritaria) ≥ 0.80

---

### **Semana 3: Generador Supervisado**
- [ ] Preparar pares reales (texto → PLS)
- [ ] Entrenar BART-base con SFT
- [ ] Entrenar T5-base (comparación)
- [ ] Evaluar en dev y test
- [ ] Seleccionar mejor checkpoint

**Entregables**:
- `models/generator_sft/`
- `data/outputs/supervised/`
- Notebook de análisis cualitativo

**Métricas Objetivo**:
- ROUGE-L ≥ 0.35
- BERTScore F1 ≥ 0.85
- ΔFlesch ≥ +15

---

### **Semana 4: Semi-Supervisado - Ronda 1**
- [ ] Implementar filtros de aceptación
- [ ] Generar PLS sintéticos con teacher model
- [ ] Aplicar filtros (legibilidad, factualidad)
- [ ] Analizar tasa de aceptación
- [ ] Re-entrenar con LoRA (mix 70:30)
- [ ] Evaluar en dev

**Entregables**:
- `data/outputs/synthetic/round1/`
- `models/generator_lora_round1/`
- Análisis de calidad sintéticos vs reales

**Control de Calidad**:
- Tasa de aceptación ≥ 30%
- BERTScore sintéticos ≥ 0.83

---

### **Semana 5: Semi-Supervisado - Ronda 2 y Ablations**
- [ ] Ronda 2 de semi-supervisado (si ronda 1 fue exitosa)
- [ ] Experimentos de ablación:
  - Solo reales vs reales+sintéticos
  - Full-FT vs LoRA
  - BART vs T5
  - Con/sin clasificador gate

**Entregables**:
- `models/generator_lora_round2/`
- Tabla comparativa de experimentos
- Notebook de ablations

---

### **Semana 6: Evaluación Final y Documentación**
- [ ] Evaluación exhaustiva en test set
- [ ] Análisis por fuente (Cochrane, Pfizer, etc.)
- [ ] Seleccionar 50 ejemplos cualitativos
- [ ] Dashboard Streamlit/Gradio
- [ ] Documentación técnica completa
- [ ] Reporte final

**Entregables**:
- `data/evaluation/final_report.json`
- Dashboard interactivo
- Reporte técnico (PDF/Markdown)
- Presentación de resultados

---

## 🔧 Stack Tecnológico

### Core ML
- **PyTorch**: Framework principal
- **Transformers (Hugging Face)**: Modelos pre-entrenados
- **PEFT**: LoRA para entrenamiento eficiente
- **Accelerate**: Distributed training

### Procesamiento de Datos
- **Pandas**: Manipulación de datos
- **PyArrow/Parquet**: Formato eficiente
- **scikit-learn**: Métricas y splits

### Métricas y Evaluación
- **rouge-score**: ROUGE metrics
- **bert-score**: Similarity semántica
- **textstat**: Legibilidad (Flesch)
- **NLTK**: Análisis de texto

### MLOps
- **DVC**: Versionado de datos y modelos
- **AWS S3**: Almacenamiento remoto
- **Git**: Versionado de código
- **Weights & Biases** (opcional): Tracking de experimentos

### Visualización
- **Streamlit**: Dashboard interactivo
- **Matplotlib/Seaborn**: Gráficos estáticos
- **Plotly**: Gráficos interactivos

---

## 📏 Métricas de Éxito

### Clasificador
| Métrica | Objetivo | Descripción |
|---------|----------|-------------|
| F1_macro | ≥ 0.85 | Balance entre PLS y non-PLS |
| Recall PLS | ≥ 0.80 | No perder textos que SÍ son PLS |
| Precision non-PLS | ≥ 0.90 | Evitar generar cuando no es necesario |

### Generador
| Métrica | Objetivo | Descripción |
|---------|----------|-------------|
| ROUGE-L | ≥ 0.35-0.40 | Overlap de secuencias más largas |
| BERTScore F1 | ≥ 0.85 | Similaridad semántica |
| ΔFlesch | ≥ +15 | Incremento en legibilidad |
| Compresión | 0.3-0.8 | Ratio longitud PLS/original |

### Semi-Supervisado
| Métrica | Objetivo | Descripción |
|---------|----------|-------------|
| Tasa aceptación | ≥ 30% | % sintéticos que pasan filtros |
| BERTScore sintéticos | ≥ 0.83 | Calidad de pares sintéticos |
| Mejora en dev | +2-3% | Incremento sobre solo supervisado |

---

## 🚨 Riesgos y Mitigaciones

### Riesgo 1: Desbalance extremo
**Mitigación**:
- Focal loss con γ=2.0
- Class weights balanceados
- Threshold tuning

### Riesgo 2: Sintéticos de baja calidad
**Mitigación**:
- Filtros multi-criterio estrictos
- Validación con BERTScore
- Control de drift por ronda

### Riesgo 3: Modelos muy grandes (GPU limitada)
**Mitigación**:
- Usar modelos -base en vez de -large
- LoRA para entrenamiento eficiente
- Gradient accumulation
- Mixed precision (fp16)

### Riesgo 4: PDFs de Pfizer no procesados
**Mitigación**:
- Priorizar ClinicalTrials y Cochrane (ya en TXT/JSONL)
- PDFs como mejora futura (OCR/parsing)

---

## 🎓 Criterios de Evaluación Académica

### Aspectos Técnicos (60%)
- Implementación correcta del pipeline completo ✓
- Manejo adecuado del desbalance ✓
- Técnicas de semi-supervisión bien aplicadas ✓
- Evaluación rigurosa con múltiples métricas ✓

### Experimentación (20%)
- Ablation studies comparando alternativas ✓
- Justificación de decisiones de diseño ✓
- Análisis de resultados por fuente ✓

### Documentación (15%)
- Código limpio y documentado ✓
- Reproducibilidad (DVC + seeds) ✓
- Reporte técnico completo ✓

### Innovación (5%)
- Dashboard interactivo ✓
- Filtros automáticos de calidad ✓
- Análisis cualitativo profundo ✓

---

## 📚 Referencias y Recursos

### Papers Relevantes
- **PLS Generation**: "Lay Summarization of Biomedical Research Articles" (EMNLP 2020)
- **Semi-Supervised NLP**: "Meta Pseudo Labels" (CVPR 2021)
- **Medical Text Summarization**: "Attention is All You Need" + BART/T5

### Datasets
- Cochrane Plain Language Summaries
- ClinicalTrials.gov
- PubMed abstracts

### Modelos Pre-entrenados
- `facebook/bart-base`: Generación general
- `t5-base`: Text-to-text
- `allenai/led-base-16384`: Documentos largos
- `dmis-lab/biobert-base`: Dominio biomédico

---

## 🔄 Workflow DVC Recomendado

```bash
# 1. Pull datos desde S3
dvc pull

# 2. Modificar parámetros
nano params.yaml

# 3. Ejecutar pipeline (solo etapas modificadas)
dvc repro

# 4. Revisar métricas
dvc metrics show
dvc metrics diff

# 5. Push resultados a S3
dvc push

# 6. Commit cambios
git add params.yaml dvc.lock
git commit -m "Experiment: increased batch size"
git push
```

---

## 📞 Contacto y Soporte

**Documentación**:
- `docs/ARCHITECTURE.md`: Detalles técnicos
- `docs/DVC_S3_SETUP.md`: Configuración DVC
- `docs/SETUP.md`: Instalación y ambiente

**Scripts útiles**:
- `setup_dvc_s3.ps1`: Configurar DVC en Windows
- `Makefile`: Comandos frecuentes

---

## ✅ Checklist de Inicio Rápido

- [ ] Instalar dependencias: `pip install -r requirements.txt`
- [ ] Configurar credenciales AWS (ver `docs/DVC_S3_SETUP.md`)
- [ ] Inicializar DVC remote: `.\setup_dvc_s3.ps1`
- [ ] Pull datos: `dvc pull`
- [ ] Ejecutar EDA: `jupyter notebook notebooks/01_EDA.ipynb`
- [ ] Revisar `params.yaml` y ajustar si necesario
- [ ] Ejecutar primera etapa: `dvc repro preprocess`

---

