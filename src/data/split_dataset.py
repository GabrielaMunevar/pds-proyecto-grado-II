
"""
Lee data/processed/dataset_clean.csv y materializa:
- data/processed/train.csv
- data/processed/dev.csv
- data/processed/test.csv

Reglas:
- Si hay filas con split=train/test (según la columna "split"), se respetan (split_method="original").
- Las filas "unsplit" (sin train/test) se reparten 80/20 de forma reproducible.
- Si existe columna "label", el 80/20 intenta ser estratificado por label.
- Esas filas quedan marcadas split_method="internal".
- Crear dev set desde train: 10% estratificado por label (PLS vs non_PLS).
- Opcional: también estratificar por fuente.
- Test se mantiene intacto para evaluación final.
"""

from pathlib import Path
import pandas as pd

P = Path("data/processed")
SRC = P / "dataset_clean.csv"
if not SRC.exists():
    raise SystemExit(f"No existe {SRC}. Ejecuta primero el stage 'preprocess'.")

print("\n" + "="*80)
print("GENERACIÓN DE PARTICIONES TRAIN/TEST")
print("="*80)

df = pd.read_csv(SRC, low_memory=False)

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
    df_train_orig["split_method"] = "original"
if not df_test_orig.empty:
    df_test_orig["split_method"] = "original"

# Si hay filas sin split, hacemos un 80/20 reproducible
if not df_unsplit.empty:
    frac_test = 0.2
    if "label" in df_unsplit.columns:
        # Intento estratificado por label; si no se puede, cae a muestreo simple
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

# ========================================================================
# CREAR DEV SET DESDE TRAIN (10% estratificado por label)
# ========================================================================
print("\n" + "="*80)
print("CREANDO DEV SET DESDE TRAIN")
print("="*80)

# Parámetros para crear dev (pueden venir de params.yaml)
try:
    import yaml
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    split_params = params.get('split', {})
    dev_frac = split_params.get('dev_size', 0.1)
    stratify_by_source = split_params.get('stratify_by_source', False)
except:
    # Valores por defecto si no se puede cargar params.yaml
    dev_frac = 0.1  # 10% para dev
    stratify_by_source = False  # Opcional: también estratificar por fuente

if "label" in df_train.columns:
    # IMPORTANTE: Solo estratificar con registros que SÍ tienen label (PLS/non_PLS)
    # Los registros sin etiqueta (NaN) NO entran en el split estratificado
    df_train_labeled = df_train[df_train['label'].notna()].copy()
    df_train_unlabeled = df_train[df_train['label'].isna()].copy()
    
    print(f"Estratificando por label (PLS vs non_PLS)...")
    print(f"  Registros con label: {len(df_train_labeled):,}")
    if len(df_train_unlabeled) > 0:
        print(f"  Registros sin label (excluidos del split estratificado): {len(df_train_unlabeled):,}")
    
    if stratify_by_source and "source_dataset" in df_train_labeled.columns:
        # Estratificar por label Y fuente (opcional)
        print("  También estratificando por fuente...")
        try:
            dev_indices = []
            for (label_val, source_val), group in df_train_labeled.groupby(["label", "source_dataset"]):
                sampled = group.sample(frac=dev_frac, random_state=42)
                dev_indices.extend(sampled.index.tolist())
            df_dev = df_train_labeled.loc[dev_indices].copy()
        except ValueError:
            # Si falla, solo por label
            print("  Falló estratificación por fuente, usando solo label...")
            dev_indices = []
            for label_val, group in df_train_labeled.groupby("label"):
                sampled = group.sample(frac=dev_frac, random_state=42)
                dev_indices.extend(sampled.index.tolist())
            df_dev = df_train_labeled.loc[dev_indices].copy()
    else:
        # Solo estratificar por label
        try:
            dev_indices = []
            for label_val, group in df_train_labeled.groupby("label"):
                sampled = group.sample(frac=dev_frac, random_state=42)
                dev_indices.extend(sampled.index.tolist())
            df_dev = df_train_labeled.loc[dev_indices].copy()
        except ValueError:
            # Si falla (pocos datos en alguna clase), muestreo simple
            print("  Falló estratificación, usando muestreo simple...")
            df_dev = df_train_labeled.sample(frac=dev_frac, random_state=42).copy()
    
    # Train final = train original (solo labeled) - dev
    df_train_final = df_train_labeled.drop(df_dev.index).copy()
    
    # Los registros sin label se mantienen en train (no se dividen)
    if len(df_train_unlabeled) > 0:
        df_train_final = pd.concat([df_train_final, df_train_unlabeled], ignore_index=True)
        print(f"  Registros sin label mantenidos en train: {len(df_train_unlabeled):,}")
    
    # Validar proporciones
    print(f"\nDistribución en train original:")
    if "label" in df_train.columns:
        print(df_train['label'].value_counts().to_dict())
    print(f"\nDistribución en train final (90%):")
    if "label" in df_train_final.columns:
        print(df_train_final['label'].value_counts().to_dict())
    print(f"\nDistribución en dev (10%):")
    if "label" in df_dev.columns:
        print(df_dev['label'].value_counts().to_dict())
    
    # Verificar que proporciones se mantienen
    if "label" in df_train.columns and "label" in df_train_final.columns and "label" in df_dev.columns:
        train_prop = df_train['label'].value_counts(normalize=True)
        dev_prop = df_dev['label'].value_counts(normalize=True)
        print(f"\nProporciones:")
        print(f"  Train original: {train_prop.to_dict()}")
        print(f"  Dev: {dev_prop.to_dict()}")
        # Calcular diferencia máxima (solo para labels comunes)
        common_labels = set(train_prop.index) & set(dev_prop.index)
        if common_labels:
            max_diff = max([abs(train_prop.get(l, 0) - dev_prop.get(l, 0)) for l in common_labels])
            print(f"  Diferencia máxima: {max_diff:.4f}")
    
    df_train = df_train_final
else:
    # Si no hay label, muestreo simple
    print("No hay columna 'label', usando muestreo simple...")
    df_dev = df_train.sample(frac=dev_frac, random_state=42)
    df_train = df_train.drop(df_dev.index).copy()

# Marcar dev con split_method
df_dev["split"] = "dev"
df_dev["split_method"] = "internal_dev"

print(f"\nResultado:")
print(f"  Train final: {len(df_train):,} registros (90% del train original)")
print(f"  Dev: {len(df_dev):,} registros (10% del train original)")
print(f"  Test: {len(df_test):,} registros (intacto)")

# Validación de data leakage
print("\n" + "="*80)
print("VALIDACIÓN DE DATA LEAKAGE")
print("="*80)

if "doc_id" in df_train.columns:
    train_ids = set(df_train['doc_id'].unique())
    dev_ids = set(df_dev['doc_id'].unique())
    test_ids = set(df_test['doc_id'].unique())
    
    # Train vs Dev
    overlap_train_dev = train_ids & dev_ids
    if overlap_train_dev:
        print(f"WARNING: {len(overlap_train_dev)} doc_id aparecen tanto en train como en dev!")
        print(f"   Ejemplos: {list(overlap_train_dev)[:5]}")
    else:
        print(" Train y Dev: No hay overlap (doc_id únicos)")
    
    # Train vs Test
    overlap_train_test = train_ids & test_ids
    if overlap_train_test:
        print(f"WARNING: {len(overlap_train_test)} doc_id aparecen tanto en train como en test!")
        print(f"   Ejemplos: {list(overlap_train_test)[:5]}")
    else:
        print(" Train y Test: No hay overlap (doc_id únicos)")
    
    # Dev vs Test
    overlap_dev_test = dev_ids & test_ids
    if overlap_dev_test:
        print(f"WARNING: {len(overlap_dev_test)} doc_id aparecen tanto en dev como en test!")
        print(f"   Ejemplos: {list(overlap_dev_test)[:5]}")
    else:
        print(" Dev y Test: No hay overlap (doc_id únicos)")

# Orden de columnas recomendado (usa solo las que existan)
cols = [c for c in [
    "texto_original","resumen","source","doc_id","split","label",
    "source_dataset","source_bucket","split_method"
] if c in df.columns or c in df_train.columns or c in df_test.columns]

if not cols:  # por si acaso
    df_train.to_csv(P / "train.csv", index=False)
    df_dev.to_csv(P / "dev.csv", index=False)
    df_test.to_csv(P / "test.csv", index=False)
else:
    df_train.reindex(columns=cols).to_csv(P / "train.csv", index=False)
    df_dev.reindex(columns=cols).to_csv(P / "dev.csv", index=False)
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
    if "source_dataset" in df_unsplit.columns:
        print(f"  Fuentes: {df_unsplit['source_dataset'].value_counts().to_dict()}")
    print(f"  Divididos 80/20:")
    print(f"    → Train: {len(df_train_unsplit):,}")
    print(f"    → Test: {len(df_test_unsplit):,}")
print(f"\nResultado final:")
print(f"  train.csv: {len(df_train):,} registros")
print(f"  dev.csv: {len(df_dev):,} registros")
print(f"  test.csv: {len(df_test):,} registros")

# Distribución por fuente en train, dev y test
if "source_dataset" in df_train.columns:
    print(f"\nDistribución por fuente en train.csv:")
    for source, count in df_train['source_dataset'].value_counts().head(10).items():
        print(f"  {source}: {count:,}")
if "source_dataset" in df_dev.columns:
    print(f"\nDistribución por fuente en dev.csv:")
    for source, count in df_dev['source_dataset'].value_counts().head(10).items():
        print(f"  {source}: {count:,}")
if "source_dataset" in df_test.columns:
    print(f"\nDistribución por fuente en test.csv:")
    for source, count in df_test['source_dataset'].value_counts().head(10).items():
        print(f"  {source}: {count:,}")

print("="*80 + "\n")
