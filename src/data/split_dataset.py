# src/data/split_dataset.py
"""
Lee data/processed/dataset_clean.csv y materializa:
- data/processed/train.csv
- data/processed/test.csv

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

# Validación de data leakage
if "doc_id" in df_train.columns and "doc_id" in df_test.columns:
    train_ids = set(df_train['doc_id'].unique())
    test_ids = set(df_test['doc_id'].unique())
    overlap = train_ids & test_ids
    if overlap:
        print(f"\n⚠️  WARNING: {len(overlap)} doc_id aparecen tanto en train como en test!")
        print(f"   Ejemplos: {list(overlap)[:5]}")
    else:
        print("\n✅ Validación: No hay data leakage (doc_id únicos)")

# Orden de columnas recomendado (usa solo las que existan)
cols = [c for c in [
    "texto_original","resumen","source","doc_id","split","label",
    "source_dataset","source_bucket","split_method"
] if c in df.columns or c in df_train.columns or c in df_test.columns]

if not cols:  # por si acaso
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
    if "source_dataset" in df_unsplit.columns:
        print(f"  Fuentes: {df_unsplit['source_dataset'].value_counts().to_dict()}")
    print(f"  Divididos 80/20:")
    print(f"    → Train: {len(df_train_unsplit):,}")
    print(f"    → Test: {len(df_test_unsplit):,}")
print(f"\nResultado final:")
print(f"  train.csv: {len(df_train):,} registros")
print(f"  test.csv: {len(df_test):,} registros")

# Distribución por fuente en train y test
if "source_dataset" in df_train.columns:
    print(f"\nDistribución por fuente en train.csv:")
    for source, count in df_train['source_dataset'].value_counts().head(10).items():
        print(f"  {source}: {count:,}")
if "source_dataset" in df_test.columns:
    print(f"\nDistribución por fuente en test.csv:")
    for source, count in df_test['source_dataset'].value_counts().head(10).items():
        print(f"  {source}: {count:,}")

print("="*80 + "\n")
