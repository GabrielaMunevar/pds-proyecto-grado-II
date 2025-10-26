# src/data/split_dataset.py
"""Lee data/processed/dataset_clean_v1.csv si materializa: - data/processed/train.csv - data/processed/test.csv Reglas: - Si si hay filas con split=train/test (según la columna"split"), se respetan (split_method="original"). - Las filas"unsplit"(sin train/test) se reparten 80/20 de forma reproducible. - Si existe columna"label", el 80/20 intenta ser estratificado por label. - Esas filas quedan marcadas split_method="internal"."""

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

print(f"train={len(df_train)}, test={len(df_test)}, total={len(df)}")
