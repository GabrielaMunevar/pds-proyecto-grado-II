# Plan de Trabajo MLOps: PLS Biomédico

## Tabla de Evaluación por Fases

| Fase | Score | Hallazgos | Archivos/Rutas Clave |
|------|-------|-----------|---------------------|
| **preprocess** | 3/3 |  Pipeline completo con 67,672 registros limpios<br/> Estadísticas JSON generadas<br/> Filtros de calidad implementados (longitud, idioma, OCR) | `src/data/make_dataset.py`<br/>`data/processed/dataset_clean.csv`<br/>`data/processed/dataset_clean_stats.json` |
| **split** | 3/3 |  División estratificada 80/20 reproducible<br/> Parámetros en `params.yaml`<br/> Outputs: train.csv (54,138), test.csv (13,534) | `src/data/split_dataset.py`<br/>`data/processed/train.csv`<br/>`data/processed/test.csv` |
| **classifier** | 3/3 |  TF-IDF + Logistic Regression entrenado<br/> F1 Macro: 0.9952 (supera target 0.85)<br/> Métricas y modelos persistidos en `models/baseline_classifier/` | `src/models/train_classifier.py`<br/>`models/baseline_classifier/classifier.pkl`<br/>`models/baseline_classifier/metrics.json` |
| **generator** | 1/3 |  Código T5 verificado y testeado<br/> Pares sintéticos creados (21,527 pares)<br/> **CALIDAD INSUFICIENTE**: ROUGE-L 0.361 (target 0.38+)<br/> Modelo entrenado pero resultados pobres para proyecto | `src/models/train_t5_generator.py`<br/>`data/processed/synthetic_pairs/synthetic_pairs.jsonl`<br/>`models/t5_generator/` |
| **semi_supervised** | 0/3 |  No implementado<br/> Falta lógica de iteración<br/> Sin estrategia de selección de datos no etiquetados | `src/models/semi_supervised.py` (no existe)<br/>`params.yaml` (sect: semi_supervised configurado) |
| **evaluation** | 2/3 |  Script ejecutado con éxito<br/> Métricas ROUGE reportadas (0.361 L)<br/> Calidad del modelo identificada como problema crítico | `src/models/evaluate_generator.py`<br/>`models/t5_generator/evaluation_summary.json`<br/>`docs/EVALUACION_RESULTADOS.md` |
| **dashboard** | 0/3 |  No existe<br/> Sin UI/UX definida<br/> Sin Streamlit u otra herramienta | `src/dashboard/` (no existe)<br/>`streamlit_app.py` (no existe) |
| **dvc_s3** | 2/3 |  Scripts de configuración existen<br/> `setup_dvc_s3.ps1` y `configure_aws.ps1` vacíos<br/> Pipeline DVC configurado (3 stages completados) | `dvc.yaml`<br/>`configure_aws.ps1` (vacío)<br/>`setup_dvc_s3.ps1` (vacío) |

## Brechas y Riesgos Críticos

- **🚨 CRÍTICO: Calidad Generador Insuficiente**: T5 entrenado produce ROUGE-L 0.361, **por debajo del target mínimo 0.38** y muy lejos de producción (0.50+). Modelo actual no es viable para continuar con semi-supervisado.
- **Datos Sintéticos Limitados**: Solo 21K pares sintéticos (generados con reglas simples) vs 50K+ recomendados para transformers. Calidad de pares es pobre.
- **Dependencia GPU Externa**: El re-entrenamiento depende de Google Colab, creando bloqueo externo y dificultando iteración rápida.
- **Semi-supervised Bloqueado**: No puede implementarse hasta mejorar base supervisada (T5 actual es mal teacher model).
- **Dashboard Inexistence**: Sin interfaz para demo o validación cualitativa, dificultando iteración con stakeholders.

## Plan de Trabajo

### 🚨 URGENTE: Mejorar Generador (BLOQUEO)

#### T1: Analizar Calidad de Pares Sintéticos
**Objetivo**: Identificar por qué los pares sintéticos generan modelo pobre

**Pasos**:
1. Revisar 100 pares aleatorios de `synthetic_pairs.jsonl` manualmente
2. Calcular métricas promedio: compresión ratio, overlap léxico, Flesch delta
3. Identificar patrones problemáticos (demasiado literal, cambios mínimos)
4. Generar reporte de calidad en `docs/ANALISIS_PARES_SINTETICOS.md`
5. Comparar con pares reales PLS si disponibles

**Criterio de aceptación**: Reporte identifica 3+ problemas específicos de calidad

---

#### T2: Generar Más Pares Sintéticos de Mejor Calidad
**Objetivo**: Expandir de 21K a 40K+ pares con mejor calidad

**Pasos**:
1. Ajustar parámetros en `create_synthetic_pairs.py` (reemplazo léxico más agresivo)
2. Implementar LLM-based augmentation (usar GPT-3.5 API o local)
3. Generar 30K adicionales pares de alta calidad
4. Validar muestra aleatoria (100 pares) manualmente
5. Actualizar `synthetic_pairs.jsonl` con nuevo dataset

**Criterio de aceptación**: 40K+ pares con compresión ratio promedio 0.5-0.8 y overlap <70%

---

#### T3: Re-entrenar T5 con Mejores Datos y Parámetros
**Objetivo**: Mejorar ROUGE-L de 0.361 a ≥0.42 (target mejorado)

**Pasos**:
1. Revisar hiperparámetros en `params.yaml` (LR, epochs, warmup)
2. Aumentar epochs a 5 (vs 3 actual)
3. Implementar learning rate scheduling más suave
4. Entrenar en Google Colab con nuevo dataset
5. Evaluar y validar métricas mejoradas

**Criterio de aceptación**: ROUGE-L ≥0.42 (mejora de +0.06)

---

### Quick Wins (≤3 días) - Secundario

#### T4: Completar Scripts DVC-S3
**Objetivo**: Hacer reproducible el setup de entorno completo

**Pasos**:
1. Implementar `configure_aws.ps1` (credenciales AWS, permisos S3)
2. Implementar `setup_dvc_s3.ps1` (configurar DVC remote, inicializar cache)
3. Validar `dvc pull` descarga todos los raw files de S3
4. Documentar proceso en `docs/GUIA_DESCARGA_DATOS_S3.md`
5. Probar pipeline completo desde cero

**Criterio de aceptación**: Ejecutar `dvc pull && dvc repro` completa sin errores

---

#### T3: Dashboard Mínimo Streamlit
**Objetivo**: Interfaz básica para generar PLS desde texto

**Pasos**:
1. Crear `src/dashboard/app.py` con Streamlit
2. Implementar carga de modelo T5 (usar checkpoint existente)
3. Text area de entrada + botón de generación
4. Mostrar texto original vs. PLS generado
5. Añadir métricas básicas (longitud, Flesch Reading)

**Criterio de aceptación**: Dashboard corre con `streamlit run src/dashboard/app.py` y genera PLS válidos

---

### Bloques 1-2 Semanas

#### T7: Dashboard Básico para Demo y Análisis
**Objetivo**: Interfaz para evaluar calidad del generador mejorado

**Pasos**:
1. Crear `src/dashboard/app.py` con Streamlit
2. Implementar carga de modelo T5 (checkpoint mejorado)
3. Text area de entrada + botón de generación
4. Mostrar texto original vs. PLS generado side-by-side
5. Añadir métricas (Flesch, suffixes, nuevas palabras)
6. Validar con stakeholders/usuarios reales

**Criterio de aceptación**: Dashboard corre y permite evaluación cualitativa de 20+ ejemplos

---

#### T8: Implementar Semi-supervisado (Iteración 1) - **Solo si T1-T3 exitosos**
**Objetivo**: Reutilizar 46K textos no etiquetados con teacher model mejorado

**Pasos**:
1. **VALIDAR**: T5 mejorado tiene ROUGE-L ≥0.42 (si no, aplicar T3 nuevamente/LoRA)
2. Crear `src/models/semi_supervised.py`
3. Implementar generación pseudo-labels (usar teacher: T5 mejorado)
4. Aplicar filtros de aceptación (BERTScore ≥0.85, Flesch ≥60)
5. Seleccionar 1000 mejores muestras de datos no etiquetados
6. Fine-tune con LoRA (2 epochs) usando mix 70% real + 30% pseudo
7. Evaluar y comparar con baseline supervisado

**Criterio de aceptación**: ROUGE-L mejora a ≥0.45 (mejora +0.03 desde baseline mejorado)

---

#### T9: Pipeline End-to-End Automatizado
**Objetivo**: Un único comando ejecuta todo el flujo

**Pasos**:
1. Añadir stage `evaluate` en `dvc.yaml` (depende de T5 entrenado)
2. Crear `Makefile` targets: `make train-all`, `make evaluate-all`
3. Validar dependencias entre stages
4. Ejecutar `dvc repro` completo
5. Documentar en `README.md` con badge de status

**Criterio de aceptación**: `make train-all` ejecuta preprocess → split → classifier → pairs → T5 → evaluate

---

### Bloques 2-4 Semanas

#### T10: Iteración Semi-supervisada Completa (2 rondas)
**Objetivo**: 2 rondas de re-entrenamiento con datos no etiquetados

**Pasos**:
1. Extender `semi_supervised.py` para manejar múltiples rondas
2. Seleccionar batches de 1000 textos no etiquetados por ronda (priorizar cortos primero)
3. Aplicar estrategia de selección (`confidence`, `uncertainty`)
4. Validar early-stopping si `rouge_l_dev ≥ 0.50`
5. Comparar métricas finales vs. baseline supervisado
6. Análisis de ejemplos cualitativos (50 samples)

**Criterio de aceptación**: ROUGE-L final ≥0.48 (mejora +0.06 desde T8, +0.12 desde inicio)

---

#### T11: Dashboard Avanzado con Comparación
**Objetivo**: Herramienta completa de evaluación y generación

**Pasos**:
1. Expandir `src/dashboard/` con vistas: Compare Models, Batch Processing
2. Integrar comparación T5 baseline vs. semi-supervisado
3. Visualizaciones: Distribución de métricas, ejemplos side-by-side
4. Exportar resultados a CSV/JSON
5. Deploy a Streamlit Cloud o local server

**Criterio de aceptación**: Dashboard permite comparar 2+ modelos y exportar resultados

---

#### T12: Documentación y Reproducibilidad
**Objetivo**: Proyecto completamente documentado y reproducible

**Pasos**:
1. Escribir `docs/GUIA_ENTRENAR_T5.md` (paso a paso Colab)
2. Actualizar `README.md` con badges, ejemplos de uso
3. Crear `docs/ARQUITECTURA_PROYECTO.md` con diagramas actualizados
4. Validar que nuevo colaborador puede reproducir desde `git clone`
5. Añadir unit tests críticos (`tests/`)

**Criterio de aceptación**: Documentación permite reproducir proyecto en <2 horas

---

## Issues JSON (Tareas de GitHub/GitLab)

```json
{
  "issues": [
    {
      "title": "🚨 CRÍTICO: Analizar calidad de pares sintéticos (21K)",
      "labels": ["critical", "data", "blocking"],
      "steps": [
        "Revisar 100 pares aleatorios manualmente",
        "Calcular métricas: compresión, overlap, Flesch delta",
        "Identificar patrones problemáticos",
        "Generar reporte en docs/ANALISIS_PARES_SINTETICOS.md"
      ]
    },
    {
      "title": "🚨 CRÍTICO: Generar más pares sintéticos (21K → 40K+ con mejor calidad)",
      "labels": ["critical", "data", "blocking"],
      "steps": [
        "Ajustar parámetros create_synthetic_pairs.py",
        "Implementar LLM-based augmentation",
        "Generar 30K adicionales",
        "Validar muestra aleatoria manualmente"
      ]
    },
    {
      "title": "🚨 CRÍTICO: Re-entrenar T5 con datos mejorados (target: ROUGE-L ≥0.42)",
      "labels": ["critical", "training", "generator", "blocking"],
      "steps": [
        "Ajustar hiperparámetros (LR, epochs=5)",
        "Entrenar en Google Colab con nuevo dataset",
        "Evaluar métricas",
        "Validar mejora de +0.06 ROUGE-L"
      ]
    },
    {
      "title": "Completar scripts de configuración DVC-S3",
      "labels": ["infrastructure", "dvc"],
      "steps": [
        "Implementar configure_aws.ps1",
        "Implementar setup_dvc_s3.ps1",
        "Validar dvc pull funciona",
        "Documentar en GUIA_DESCARGA_DATOS_S3.md"
      ]
    },
    {
      "title": "Crear dashboard Streamlit básico para evaluar generador",
      "labels": ["dashboard", "ui"],
      "steps": [
        "Crear src/dashboard/app.py",
        "Implementar carga de modelo T5 mejorado",
        "Text area + botón generación",
        "Mostrar métricas (Flesch, overlap)"
      ]
    },
    {
      "title": "Implementar bucle semi-supervisado (requiere T5 ROUGE-L ≥0.42)",
      "labels": ["semi-supervised", "generator", "priority-high"],
      "steps": [
        "Crear src/models/semi_supervised.py",
        "Generar pseudo-labels con teacher model",
        "Aplicar filtros aceptación",
        "Fine-tune con LoRA en mix 70/30"
      ]
    },
    {
      "title": "Automatizar pipeline end-to-end con DVC y Makefile",
      "labels": ["pipeline", "automation", "dvc"],
      "steps": [
        "Añadir stage evaluate en dvc.yaml",
        "Crear Makefile targets",
        "Validar dependencias",
        "Documentar en README"
      ]
    },
    {
      "title": "Iteración completa semi-supervisada (2 rondas)",
      "labels": ["semi-supervised", "generator"],
      "steps": [
        "Extender para múltiples rondas",
        "Implementar selección batches",
        "Aplicar early-stopping",
        "Comparar métricas finales"
      ]
    },
    {
      "title": "Completar scripts de configuración DVC-S3 (configure_aws.ps1)",
      "labels": ["infrastructure", "dvc", "quick-win"],
      "steps": [
        "Implementar configuración credenciales AWS",
        "Configurar permisos S3 bucket",
        "Validar dvc pull funciona",
        "Documentar proceso en GUIA_DESCARGA_DATOS_S3.md"
      ]
    },
    {
      "title": "Crear dashboard Streamlit básico para generación PLS",
      "labels": ["dashboard", "quick-win", "ui"],
      "steps": [
        "Crear src/dashboard/app.py",
        "Implementar carga de modelo T5",
        "Text area + botón generación",
        "Mostrar métricas básicas (Flesch)"
      ]
    },
    {
      "title": "Fine-tune T5-small en Google Colab con 21K pares sintéticos",
      "labels": ["training", "generator", "priority-high"],
      "steps": [
        "Adaptar train_t5_generator.py para Colab",
        "Upload datos a Colab",
        "Configurar GPU runtime",
        "Entrenar 3 epochs con checkpoints",
        "Descargar modelo final"
      ]
    },
    {
      "title": "Implementar bucle semi-supervisado (primera iteración)",
      "labels": ["semi-supervised", "generator", "priority-high"],
      "steps": [
        "Crear src/models/semi_supervised.py",
        "Generar pseudo-labels con teacher model",
        "Aplicar filtros aceptación",
        ứng "Fine-tune con LoRA en mix 70/30",
        "Evaluar y comparar"
      ]
    },
    {
      "title": "Automatizar pipeline end-to-end con DVC y Makefile",
      "labels": ["pipeline", "automation", "dvc"],
      "steps": [
        "Añadir stage evaluate en dvc.yaml",
        "Crear Makefile targets",
        "Validar dependencias",
        "Documentar en README"
      ]
    },
    {
      "title": "Iteración completa semi-supervisada (2 rondas)",
      "labels": ["semi-supervised", "generator", "priority-medium"],
      "steps": [
        "Extender para múltiples rondas",
        "Implementar selección batches",
        "Aplicar early-stopping",
        "Comparar métricas finales"
      ]
    },
    {
      "title": "Expander dashboard con comparación modelos y visualizaciones",
      "labels": ["dashboard", "ui", "priority-medium"],
      "steps": [
        "Añadir vista Compare Models",
        "Implementar visualizaciones métricas",
        "Exportar resultados CSV/JSON",
        "Deploy a Streamlit Cloud"
      ]
    },
    {
      "title": "Documentar proyecto completo y validar reproducibilidad",
      "labels": ["documentation", "reproducibility", "priority-low"],
      "steps": [
        "Escribir GUIA_ENTRENAR_T5.md",
        "Actualizar README con badges",
        "Crear tests unitarios",
        "Validar colaborador nuevo puede reproducir"
      ]
    },
    {
      "title": "Optimizar generación de pares sintéticos (21K → target mayor)",
      "labels": ["data", "synthetic", "priority-low"],
      "steps": [
        "Analizar calidad pares existentes",
        "Ajustar parámetros create_synthetic_pairs.py",
        "Generar 30K+ pares",
        "Validar distribución balanceada"
      ]
    }
  ]
}
```

---

## Resumen Ejecutivo

**Estado Actual**: Pipeline DVC con 4/8 fases completadas (50%). Preprocess, split y classifier funcionando excelentemente (F1: 0.9952). Generador T5 entrenado pero con **calidad insuficiente**: ROUGE-L 0.361 (por debajo del target 0.38+). Evaluación ejecutada, semi-supervisado bloqueado.

**🚨 PROBLEMA CRÍTICO**: El modelo T5 actual (ROUGE-L 0.361) no es viable para:
- Continuar con semi-supervisado (teacher model pobre)
- Presentar resultados de calidad
- Producción

**Prioridades Inmediatas (BLOQUEO)**: 
1. Analizar calidad pares sintéticos (21K) y generar reporte
2. Generar más pares sintéticos (40K+ con mejor calidad)
3. Re-entrenar T5 con datos mejorados (target: ROUGE-L ≥0.42)

**Riesgo Principal**: Calidad de datos sintéticos pobres genera modelo insuficiente. Dependencia de GPU externa (Colab).

**Criterio de Éxito**: ROUGE-L ≥0.42 (vs 0.361 actual), luego iterar con semi-supervisado para alcanzar ≥0.48.

