# Módulo T5 - Generación de Plain Language Summaries

Este módulo contiene todas las funcionalidades relacionadas con el modelo T5 para generar resúmenes en lenguaje sencillo (PLS) a partir de textos biomédicos técnicos.

## Estructura

```
src/models/t5/
├── __init__.py              # Exportaciones principales
├── README.md                # Este archivo
├── training/                # Entrenamiento del modelo
│   ├── __init__.py
│   └── trainer.py           # Funciones de entrenamiento
├── evaluation/              # Evaluación del modelo
│   ├── __init__.py
│   └── evaluator.py         # Funciones de evaluación y generación
├── chunking/                # Chunking de documentos largos
│   ├── __init__.py
│   └── chunker.py           # Funciones de chunking
└── length_analysis/         # Análisis de longitudes
    ├── __init__.py
    └── analyzer.py          # Funciones de análisis de longitudes
```

## Uso

### Entrenamiento

```python
from models.t5 import train_t5_generator, load_synthetic_pairs

# Cargar datos
pairs = load_synthetic_pairs()

# Entrenar modelo
trainer, tokenizer, metrics = train_t5_generator(
    pairs=pairs,
    model_name='t5-base',
    output_dir='models/t5_generator',
    num_epochs=3,
    batch_size=8
)
```

### Evaluación

```python
from models.t5 import evaluate_t5_generator, load_t5_model

# Cargar modelo
model, tokenizer = load_t5_model()

# Evaluar
results = evaluate_t5_generator(
    model=model,
    tokenizer=tokenizer,
    test_pairs=test_pairs,
    output_dir=Path('models/t5_generator/evaluation')
)
```

### Generación de PLS

```python
from models.t5 import generate_pls, load_t5_model

# Cargar modelo
model, tokenizer = load_t5_model()

# Generar PLS
pls = generate_pls(
    model=model,
    tokenizer=tokenizer,
    technical_text="Texto técnico biomédico...",
    use_chunking=True
)
```

### Chunking

```python
from models.t5 import expand_pairs_with_chunking

# Expandir pares con chunking
expanded_pairs = expand_pairs_with_chunking(
    pairs=pairs,
    tokenizer=tokenizer,
    max_tokens=400
)
```

### Análisis de Longitudes

```python
from models.t5 import analyze_lengths_for_t5, analyze_lengths_before_after_chunking

# Analizar longitudes
analysis = analyze_lengths_for_t5(
    pairs=pairs,
    tokenizer=tokenizer,
    max_length_source=400,
    save_report=True,
    output_dir=Path('models/t5_generator/length_analysis')
)

# Comparar antes y después de chunking
comparison = analyze_lengths_before_after_chunking(
    pairs_before=original_pairs,
    pairs_after=expanded_pairs,
    tokenizer=tokenizer,
    output_dir=Path('models/t5_generator/length_analysis')
)
```

## Características

- **Chunking automático**: Divide documentos largos en chunks que quepan en la ventana de contexto de T5 (512 tokens)
- **Análisis de longitudes**: Analiza la distribución de longitudes de documentos y calcula pérdida por truncation
- **Evaluación completa**: Calcula métricas ROUGE, BLEU, SARI, BERTScore y legibilidad
- **Organización modular**: Código organizado por funcionalidad para facilitar mantenimiento

## Configuración

El módulo usa la configuración centralizada en `src/config.py`:
- Prompt estándar para simplificación
- Configuración de chunking
- Configuración de modelos

