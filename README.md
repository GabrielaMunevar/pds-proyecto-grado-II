# Proyecto PLS Biomédico con Semi-Supervisión

Este proyecto implementa un sistema para generar Plain Language Summaries (PLS) de textos biomédicos utilizando técnicas de aprendizaje semi-supervisado.

## Estructura del Proyecto

```
├── data/
│   ├── raw/               # Datos crudos (TXT/CSV/JSON/JSONL)
│   ├── processed/         # Datos procesados y splits
│   ├── outputs/           # PLS generados (supervisados y sintéticos)
│   └── evaluation/        # Reportes de evaluación
├── src/
│   ├── data/              # Scripts de procesamiento de datos
│   ├── models/            # Scripts de entrenamiento y evaluación
│   └── loops/             # Bucle semi-supervisado
├── models/                # Modelos entrenados (pesos y checkpoints)
├── docs/                  # Documentación del proyecto
├── notebooks/             # Jupyter notebooks para exploración
├── dvc.yaml               # Pipeline DVC
└── params.yaml            # Hiperparámetros centralizados
```

## Flujo de Trabajo

1. **Ingesta y Preprocesamiento**: Normalización, limpieza y deduplicación
2. **Split de Datos**: División estratificada train/dev/test
3. **Clasificador PLS/non-PLS**: Detecta si un texto ya es PLS
4. **Generación Supervisada**: Entrenamiento con pares reales
5. **Bucle Semi-Supervisado**: Amplía pares con datos sintéticos filtrados
6. **Evaluación**: Métricas automáticas (ROUGE, BERTScore, Flesch)

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Ejecutar el pipeline completo con DVC
```bash
dvc repro
```

### Ejecutar etapas individuales
```bash
python src/data/make_dataset.py
python src/data/split_dataset.py
python src/models/train_classifier.py
python src/models/generate_pls.py --mode train
python src/loops/semi_supervised_loop.py
python src/models/evaluate.py
```

## Métricas Objetivo

- **Clasificador**: F1_macro ≥ 0.85
- **Generador**: ROUGE-L ≥ 0.35-0.40, BERTScore F1 ≥ 0.85, ΔFlesch ≥ +15

## Requisitos

- Python 3.8+
- PyTorch
- Transformers
- DVC
- Pandas, NumPy, scikit-learn
- Otras dependencias en `requirements.txt`


