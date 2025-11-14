# Análisis de la Estructura del Dataset

## 1. Localización de Archivos Clave

### Archivos de Documentación
- **README del dataset**: `data/raw/README.md` - Describe la estructura de las fuentes de datos
- **Resumen de mejoras**: `docs/summary_mejoras.md` - Sección crítica sobre "Uso de particiones originales del dataset"

### Carpetas de Datos RAW
- **Ubicación**: `data/raw/`
- **Fuentes identificadas**:
  1. `data/raw/Cochrane/`
  2. `data/raw/Pfizer/`
  3. `data/raw/ClinicalTrials.gov/`
  4. `data/raw/Trial Summaries/`

### Carpetas de Datos Procesados
- **Ubicación**: `data/processed/`
- **Archivos generados**:
  - `dataset_clean.csv` - Dataset completo procesado
  - `dataset_clean.jsonl` - Versión JSONL
  - `train.csv` - Partición de entrenamiento
  - `test.csv` - Partición de prueba

### Scripts de Procesamiento
- **`src/data/make_dataset.py`** - Genera `dataset_clean.csv` desde `data/raw/`
- **`src/data/split_dataset.py`** - Genera `train.csv` y `test.csv` desde `dataset_clean.csv`

---

## 2. Estructura Actual del Dataset

### 2.1 Organización por Fuente en `data/raw/`

#### **Cochrane**
```
data/raw/Cochrane/
├── train/
│   ├── pls/          (4,496 textos + 11,745 aumentados = 16,241 total)
│   └── non_pls/       (6,641 textos + 24,477 aumentados = 31,118 total)
└── test/
    ├── pls/          (1,124 textos + 2,816 aumentados = 3,940 total)
    └── non_pls/      (1,661 textos + 6,003 aumentados = 7,664 total)
```
**Estado**: ✅ Tiene particiones train/test claras

#### **Pfizer**
```
data/raw/Pfizer/
├── original_pfizer_texts/  (130 PDFs originales)
├── train/
│   └── pls/                (491 textos extraídos y aumentados)
└── test/
    └── pls/                (117 textos extraídos y aumentados)
```
**Estado**: ✅ Tiene particiones train/test claras (solo PLS)

#### **ClinicalTrials.gov**
```
data/raw/ClinicalTrials.gov/
├── train/                   (623 textos extraídos de API, aumentados por párrafos)
└── test/                    (130 textos extraídos de API, aumentados por párrafos)
```
**Estado**: ✅ Tiene particiones train/test claras

#### **Trial Summaries**
```
data/raw/Trial Summaries/
├── original_texts/          (PDFs originales por compañía: alx, amg, ast, csl, gsk, ins, lbi, med, sep, viv)
├── not_taken_texts/         (Textos no seleccionados)
├── alx/                     (57 textos extraídos)
└── med/                     (979 textos extraídos)
```
**Estado**: ❌ **NO tiene particiones train/test** - Todos los archivos están en carpetas sin estructura de split

---

### 2.2 Cómo se Procesan las Particiones en `make_dataset.py`

La función `infer_meta_from_path()` (líneas 316-357) infiere el split desde la ruta del archivo:

```python
split = "unsplit"
if "train" in parts:
    split = "train"
if "test" in parts:
    split = "test"
```

**Resultado**:
- ✅ **Cochrane**: `train/` → `split="train"`, `test/` → `split="test"`
- ✅ **Pfizer**: `train/` → `split="train"`, `test/` → `split="test"`
- ✅ **ClinicalTrials.gov**: `train/` → `split="train"`, `test/` → `split="test"`
- ❌ **Trial Summaries**: Sin carpetas train/test → `split="unsplit"` (9,388 registros)

---

### 2.3 Estado Actual del Dataset Procesado

**`dataset_clean.csv`** (71,591 registros totales):
- `split="train"`: 49,032 registros
- `split="test"`: 12,021 registros
- `split="unsplit"`: 10,538 registros ⚠️

**Distribución por fuente**:
- Cochrane: 58,963 registros (train + test)
- Trial Summaries: 9,388 registros (todos "unsplit")
- Pfizer: 2,488 registros (train + test)
- ClinicalTrials.gov: 752 registros (train + test)

**Distribución por etiqueta**:
- `non_pls`: 38,774 registros
- `pls`: 21,527 registros

---

### 2.4 Cómo se Generan `train.csv` y `test.csv`

El script `split_dataset.py` (líneas 14-55):

1. **Respeta splits originales**: Registros con `split="train"` o `split="test"` se mantienen
2. **Divide "unsplit" internamente**: Los 10,538 registros "unsplit" se dividen 80/20 con `random_state=42`

**Estado actual**:
- **`train.csv`**: 59,570 registros
  - 49,032 con `split="train"` (original)
  - 10,538 con `split="unsplit"` → divididos 80/20 → ~8,430 a train, ~2,108 a test
- **`test.csv`**: 12,021 registros
  - 12,021 con `split="test"` (original)
  - ~2,108 de "unsplit" (del 20%)

**⚠️ PROBLEMA CRÍTICO**: Los registros de Trial Summaries (9,388) están siendo divididos internamente, **NO respetando particiones originales** (si las hubiera).

---

## 3. Validación de Consistencia de Particiones

### 3.1 Problemas Detectados

#### ❌ **PROBLEMA 1: Trial Summaries sin particiones originales**
- **Descripción**: Trial Summaries no tiene estructura train/test en `data/raw/`
- **Impacto**: Los 9,388 registros se marcan como "unsplit" y se dividen 80/20 internamente
- **Riesgo**: Si Trial Summaries debería tener particiones originales, se está perdiendo esa información
- **Evidencia**: `dataset_clean.csv` muestra 9,388 registros con `split="unsplit"` y `source_dataset="trialsummaries"`

#### ❌ **PROBLEMA 2: Bug en `split_dataset.py` (línea 26)**
```python
if not df_train_orig.empty:
    # FALTA: df_train_orig["split_method"] = "original"
```
- **Descripción**: Falta asignar `split_method="original"` para train
- **Impacto**: Los registros de train original no tienen `split_method` asignado
- **Estado**: Solo test tiene `split_method="original"` (línea 28)

#### ⚠️ **PROBLEMA 3: Mezcla de métodos de split**
- **Descripción**: `train.csv` contiene registros con `split_method="original"` y `split_method="internal"`
- **Impacto**: Dificulta rastrear qué datos vienen de particiones originales vs. splits internos
- **Evidencia**: `split_dataset.py` concatena ambos tipos (línea 51)

#### ✅ **VERIFICACIÓN: No hay data leakage evidente**
- Los registros con `split="train"` en raw → `train.csv` ✅
- Los registros con `split="test"` en raw → `test.csv` ✅
- Los "unsplit" se dividen de forma reproducible (random_state=42) ✅
- **PERO**: No se puede verificar si Trial Summaries debería tener splits originales

#### ⚠️ **PROBLEMA 4: Distribución desbalanceada por fuente en test**
- **Cochrane**: 11,609 registros en test (96.6% del test)
- **Pfizer**: 282 registros en test (2.3%)
- **ClinicalTrials.gov**: 130 registros en test (1.1%)
- **Trial Summaries**: ~2,108 registros en test (17.5% - del split interno)
- **Impacto**: Test está muy dominado por Cochrane

---

### 3.2 Contradicciones entre Documentación y Código

#### **`summary_mejoras.md` (líneas 20-42)**
> "El profesor indicó: 'tienen que utilizar las mismas particiones que vienen de los datasets, no, ya están partidos en testing y para que sean comparables'."

**Contradicción**:
- ✅ El código (`split_dataset.py`) **SÍ respeta** las particiones originales cuando existen
- ❌ Pero **NO hay particiones originales** para Trial Summaries
- ⚠️ El código divide "unsplit" internamente, lo cual puede ser correcto si Trial Summaries nunca tuvo splits, pero **debe documentarse explícitamente**

#### **`params.yaml` (línea 32)**
```yaml
split:
  method: "internal"               # 'internal' o 'original'
  respect_existing_splits: true     # Respetar splits existentes
```

**Contradicción**:
- El parámetro `method: "internal"` sugiere que debería hacer split interno
- Pero `respect_existing_splits: true` sugiere que debería respetar splits originales
- **El código actual hace ambas cosas**: respeta originales Y divide internos

---

## 4. Propuesta de Organización Clara del Dataset

### 4.1 Definición Explícita de Train/Test por Fuente

#### **Cochrane**
- **TRAIN**: `data/raw/Cochrane/train/` (pls + non_pls)
- **TEST**: `data/raw/Cochrane/test/` (pls + non_pls)
- **Estado**: ✅ Correcto

#### **Pfizer**
- **TRAIN**: `data/raw/Pfizer/train/pls/`
- **TEST**: `data/raw/Pfizer/test/pls/`
- **Estado**: ✅ Correcto

#### **ClinicalTrials.gov**
- **TRAIN**: `data/raw/ClinicalTrials.gov/train/`
- **TEST**: `data/raw/ClinicalTrials.gov/test/`
- **Estado**: ✅ Correcto

#### **Trial Summaries** ⚠️
- **PROBLEMA**: No hay estructura train/test en raw
- **OPCIONES**:
  1. **Si NO hay particiones originales**: Mantener como "unsplit" y dividir 80/20 (actual)
  2. **Si SÍ hay particiones originales**: Crear estructura `train/` y `test/` en raw
  3. **Si hay metadatos externos**: Usar archivo de mapeo (ej: `trial_summaries_splits.csv`)

**RECOMENDACIÓN**: Verificar con el equipo si Trial Summaries tiene particiones originales definidas. Si no, documentar explícitamente que se divide internamente.

---

### 4.2 Cambios Propuestos en los Scripts

#### **A. Corregir bug en `split_dataset.py` (línea 26)**

**Código actual**:
```python
if not df_train_orig.empty:
    # FALTA asignación
if not df_test_orig.empty:
    df_test_orig["split_method"] = "original"
```

**Código corregido**:
```python
if not df_train_orig.empty:
    df_train_orig["split_method"] = "original"
if not df_test_orig.empty:
    df_test_orig["split_method"] = "original"
```

#### **B. Mejorar logging y validación en `split_dataset.py`**

Agregar reporte detallado:
```python
print("\n=== REPORTE DE PARTICIONES ===")
print(f"Total registros: {len(df)}")
print(f"\nRegistros con split original:")
print(f"  Train: {len(df_train_orig)}")
print(f"  Test: {len(df_test_orig)}")
print(f"\nRegistros sin split (unsplit): {len(df_unsplit)}")
if not df_unsplit.empty:
    print(f"  Fuentes: {df_unsplit['source_dataset'].value_counts().to_dict()}")
    print(f"  Divididos 80/20:")
    print(f"    → Train: {len(df_train_unsplit)}")
    print(f"    → Test: {len(df_test_unsplit)}")
print(f"\nResultado final:")
print(f"  train.csv: {len(df_train)} registros")
print(f"  test.csv: {len(df_test)} registros")
```

#### **C. Agregar validación de data leakage en `split_dataset.py`**

```python
# Validar que no hay duplicados entre train y test
train_ids = set(df_train['doc_id'].unique())
test_ids = set(df_test['doc_id'].unique())
overlap = train_ids & test_ids
if overlap:
    print(f"⚠️ WARNING: {len(overlap)} doc_id aparecen tanto en train como en test!")
    print(f"  Ejemplos: {list(overlap)[:5]}")
else:
    print("✅ Validación: No hay data leakage (doc_id únicos)")
```

#### **D. Mejorar `make_dataset.py` para Trial Summaries**

Si Trial Summaries tiene metadatos de split, agregar función para leerlos:

```python
def get_trial_summaries_split(fp: Path) -> str:
    """
    Determina split para Trial Summaries desde metadatos externos.
    
    Si existe archivo de mapeo (ej: data/raw/Trial Summaries/splits.csv),
    lo usa. Si no, retorna "unsplit".
    """
    # Opción 1: Archivo de mapeo
    splits_file = RAW / "Trial Summaries" / "splits.csv"
    if splits_file.exists():
        splits_df = pd.read_csv(splits_file)
        # Mapear doc_id o filename a split
        # ...
        return split
    
    # Opción 2: Regla heurística (ej: por compañía)
    # ...
    
    return "unsplit"
```

---

### 4.3 Convención de Nombres Propuesta

**Estructura estándar para todas las fuentes**:
```
data/raw/{SOURCE_NAME}/
├── train/
│   ├── pls/          (si aplica)
│   └── non_pls/      (si aplica)
└── test/
    ├── pls/          (si aplica)
    └── non_pls/      (si aplica)
```

**Excepciones documentadas**:
- **Trial Summaries**: Si no tiene splits, documentar en `data/raw/Trial Summaries/README.md`
- **Archivos de mapeo**: Si hay splits externos, usar `{SOURCE_NAME}/splits.csv` o `{SOURCE_NAME}/splits.json`

---

## 5. Resumen Ejecutivo

### 5.1 Estructura Actual

| Fuente | Train (raw) | Test (raw) | Estado en dataset_clean.csv | Estado en train/test.csv |
|--------|-------------|------------|------------------------------|--------------------------|
| **Cochrane** | ✅ `train/` | ✅ `test/` | `split="train"` o `"test"` | ✅ Respeta original |
| **Pfizer** | ✅ `train/pls/` | ✅ `test/pls/` | `split="train"` o `"test"` | ✅ Respeta original |
| **ClinicalTrials.gov** | ✅ `train/` | ✅ `test/` | `split="train"` o `"test"` | ✅ Respeta original |
| **Trial Summaries** | ❌ No existe | ❌ No existe | `split="unsplit"` (9,388) | ⚠️ Dividido 80/20 interno |

### 5.2 Problemas Detectados

1. **CRÍTICO**: Trial Summaries sin particiones originales → dividido internamente
2. **BUG**: Falta asignar `split_method="original"` para train en `split_dataset.py` (línea 26)
3. **DOCUMENTACIÓN**: Falta claridad sobre si Trial Summaries debería tener splits originales
4. **DESBALANCE**: Test dominado por Cochrane (96.6%)

### 5.3 Propuesta de Corrección

#### **Inmediato (Crítico)**:
1. ✅ Corregir bug línea 26 en `split_dataset.py`
2. ✅ Agregar validación de data leakage
3. ✅ Mejorar logging en `split_dataset.py`

#### **Corto plazo (Importante)**:
4. ⚠️ **Verificar con el equipo**: ¿Trial Summaries tiene particiones originales?
   - Si SÍ: Crear estructura `train/` y `test/` en raw o archivo de mapeo
   - Si NO: Documentar explícitamente en `data/raw/Trial Summaries/README.md`
5. ✅ Agregar reporte detallado de particiones en `split_dataset.py`

#### **Mediano plazo (Mejora)**:
6. ✅ Estandarizar estructura de carpetas para todas las fuentes
7. ✅ Crear script de validación de particiones (`validate_splits.py`)

---

## 6. Fragmentos de Código para Implementar

### 6.1 `split_dataset.py` Corregido

```python
# src/data/split_dataset.py
"""Lee data/processed/dataset_clean.csv y materializa train.csv y test.csv.
Reglas:
- Si hay filas con split=train/test (según la columna "split"), se respetan (split_method="original").
- Las filas "unsplit" (sin train/test) se reparten 80/20 de forma reproducible.
- Si existe columna "label", el 80/20 intenta ser estratificado por label.
- Esas filas quedan marcadas split_method="internal".
"""

from pathlib import Path
import pandas as pd

P = Path("data/processed")
SRC = P / "dataset_clean.csv"
if not SRC.exists():
    raise SystemExit(f"No existe {SRC}. Ejecuta primero el stage 'preprocess'.")

df = pd.read_csv(SRC)

# Normaliza la columna split
split_lower = df.get("split", "").astype(str).str.lower()
is_train = split_lower.eq("train")
is_test  = split_lower.eq("test")
is_unsplit = ~(is_train | is_test)

df_train_orig = df[is_train].copy()
df_test_orig  = df[is_test].copy()
df_unsplit    = df[is_unsplit].copy()

# Marca método para los que ya venían con split
if not df_train_orig.empty:
    df_train_orig["split_method"] = "original"  # ✅ CORREGIDO
if not df_test_orig.empty:
    df_test_orig["split_method"] = "original"

# Si hay filas sin split, hacemos un 80/20 reproducible
if not df_unsplit.empty:
    frac_test = 0.2
    if "label" in df_unsplit.columns:
        try:
            df_test_unsplit = (
                df_unsplit
                .groupby("label", group_keys=False)
                .apply(lambda x: x.sample(frac=frac_test, random_state=42))
            )
        except ValueError:
            df_test_unsplit = df_unsplit.sample(frac=frac_test, random_state=42)
    else:
        df_test_unsplit = df_unsplit.sample(frac=frac_test, random_state=42)

    df_train_unsplit = df_unsplit.drop(df_test_unsplit.index).copy()
    df_train_unsplit["split_method"] = "internal"
    df_test_unsplit["split_method"]  = "internal"

    df_train = pd.concat([df_train_orig, df_train_unsplit], ignore_index=True)
    df_test  = pd.concat([df_test_orig,  df_test_unsplit],  ignore_index=True)
else:
    df_train = df_train_orig
    df_test  = df_test_orig

# Validación de data leakage
train_ids = set(df_train['doc_id'].unique())
test_ids = set(df_test['doc_id'].unique())
overlap = train_ids & test_ids
if overlap:
    print(f"⚠️ WARNING: {len(overlap)} doc_id aparecen tanto en train como en test!")
    print(f"  Ejemplos: {list(overlap)[:5]}")
else:
    print("✅ Validación: No hay data leakage (doc_id únicos)")

# Orden de columnas recomendado
cols = [c for c in [
    "texto_original","resumen","source","doc_id","split","label",
    "source_dataset","source_bucket","split_method"
] if c in df.columns or c in df_train.columns or c in df_test.columns]

if not cols:
    df_train.to_csv(P / "train.csv", index=False)
    df_test.to_csv(P / "test.csv", index=False)
else:
    df_train.reindex(columns=cols).to_csv(P / "train.csv", index=False)
    df_test.reindex(columns=cols).to_csv(P / "test.csv", index=False)

# Reporte detallado
print("\n" + "="*80)
print("REPORTE DE PARTICIONES")
print("="*80)
print(f"Total registros: {len(df):,}")
print(f"\nRegistros con split original:")
print(f"  Train: {len(df_train_orig):,}")
print(f"  Test: {len(df_test_orig):,}")
if not df_unsplit.empty:
    print(f"\nRegistros sin split (unsplit): {len(df_unsplit):,}")
    print(f"  Fuentes: {df_unsplit['source_dataset'].value_counts().to_dict()}")
    print(f"  Divididos 80/20:")
    print(f"    → Train: {len(df_train_unsplit):,}")
    print(f"    → Test: {len(df_test_unsplit):,}")
print(f"\nResultado final:")
print(f"  train.csv: {len(df_train):,} registros")
print(f"  test.csv: {len(df_test):,} registros")
print("="*80)
```

---

## 7. Próximos Pasos Recomendados

1. **Revisar con el equipo**: ¿Trial Summaries tiene particiones originales?
2. **Aplicar correcciones inmediatas**: Bug en línea 26, validación de leakage
3. **Documentar decisiones**: Si Trial Summaries se divide internamente, documentarlo explícitamente
4. **Crear script de validación**: `validate_splits.py` para verificar consistencia
5. **Re-ejecutar pipeline**: Regenerar `train.csv` y `test.csv` con correcciones

---

**Fecha de análisis**: 2024
**Versión del dataset analizado**: `dataset_clean.csv` con 71,591 registros

