# Arquitectura del Proyecto: PLS Biomédico

```mermaid
graph TB
    subgraph "FUENTES DE DATOS"
        S3[S3 Bucket<br/>65,941 archivos raw]
        Raw[TXT, PDF<br/>ClinicalTrials, Cochrane, Pfizer]
    end

    subgraph "ETAPA 1: PREPROCESAMIENTO"
        DVC1[DVC Stage: preprocess]
        Clean[Dataset Clean<br/>67,672 registros<br/>Parágrafos, filtros calidad]
        Stats[Estadísticas<br/>JSON]
    end

    subgraph "ETAPA 2: DIVISIÓN"
        DVC2[DVC Stage: split]
        Train[Train<br/>54,138 registros]
        Test[Test<br/>13,534 registros]
    end

    subgraph "ETAPA 3: CLASIFICADOR"
        DVC3[DVC Stage: train_classifier]
        CLF[Baseline Classifier<br/>TF-IDF + Logistic Regression<br/>F1: 0.9952]
        Metrics1[Metrics JSON<br/>Source Metrics]
    end

    subgraph "ETAPA 4: PARES SINTÉTICOS"
        Pairs[Create Synthetic Pairs]
        Synthetic[21,527 pares<br/>técnico → simple]
        TrainPairs[Train: 17,298]
        TestPairs[Test: 4,229]
    end

    subgraph "ETAPA 5: GENERADOR T5"
        DVC5[DVC Stage: train_t5_generator]
        T5Model[T5-Small Generator<br/>Encoder-Decoder<br/>Status: Coded & Tested]
        T5Metrics[Metrics JSON]
    end

    subgraph "EVALUACIÓN"
        Eval[Evaluar T5<br/>ROUGE, BERTScore]
        Compare[Comparar modelos]
    end

    subgraph "SEMI-SUPERVISADO"
        Loop[Loop Semi-supervisado<br/>Generar PLS sintéticos]
        ReTrain[Re-entrenar T5<br/>con datos no etiquetados]
    end

    subgraph "OUTPUT FINAL"
        Final[T5 Entrenado<br/>Modelo + Tokenizer]
        Predictions[PLS Generados]
    end

    %% Flujo de datos
    S3 --> Raw
    Raw --> DVC1
    DVC1 --> Clean
    DVC1 --> Stats
    Clean --> DVC2
    DVC2 --> Train
    DVC2 --> Test
    Train --> DVC3
    Test --> DVC3
    DVC3 --> CLF
    DVC3 --> Metrics1

    Train --> Pairs
    Test --> Pairs
    Pairs --> Synthetic
    Synthetic --> TrainPairs
    Synthetic --> TestPairs
    
    TrainPairs --> DVC5
    TestPairs --> DVC5
    DVC5 --> T5Model
    DVC5 --> T5Metrics

    T5Model --> Eval
    Eval --> Compare
    Compare --> Loop
    Loop --> ReTrain
    ReTrain --> Final
    Final --> Predictions

    %% Styling
    classDef completed fill:#90EE90,stroke:#006400,stroke-width:2px
    classDef pending fill:#FFE4B5,stroke:#FF8C00,stroke-width:2px
    classDef current fill:#87CEEB,stroke:#0000CD,stroke-width:2px

    class DVC1,DVC2,DVC3,Clean,Train,Test,CLF,Pairs,Synthetic,DVC5,T5Model completed
    class Eval,Compare,Loop,ReTrain pending
    class T5Model current
```

## Componentes Implementados

1. **Pipeline DVC** (3 stages completados)
2. **Preprocesamiento** (67K registros limpios)
3. **Clasificador Baseline** (F1: 0.9952)
4. **Generador de Pares Sintéticos** (21K pares)
5. **T5-Small** (Código verificado, listo para GPU)

## Componentes Pendientes

1. **Evaluación** (ROUGE, BERTScore)
2. **Bucle Semi-supervisado**
3. **Refinamiento**
4. **Dashboard Streamlit**

## Pipeline DVC Actual

```yaml
stages:
  preprocess          → dataset_clean.csv (67,672 registros)
  split               → train.csv, test.csv
  train_classifier    → classifier.pkl (F1: 0.9952)
  train_t5_generator  → T5 model (entrenado en Colab)
  evaluate            → Métricas finales
  semi_supervised     → Iteración con datos no etiquetados
```

## Arquitectura de Componentes

```mermaid
graph LR
    subgraph "DATA LAYER"
        Raw[Raw Data<br/>65,941 files]
        Processed[Processed<br/>67,672 records]
    end

    subgraph "MODEL LAYER"
        CLF[Classifier<br/>TF-IDF + LogReg]
        T5[T5-Small<br/>Encoder-Decoder]
    end

    subgraph "EVALUATION LAYER"
        Metrics[Metrics<br/>ROUGE, BERTScore]
    end

    Raw --> Processed
    Processed --> CLF
    Processed --> T5
    CLF --> Metrics
    T5 --> Metrics
```

## Stack Tecnológico

```
┌──────────────────────────────────────────┐
│ Python 3.11                              │
│ - Pandas, NumPy                          │
│ - scikit-learn (TF-IDF, LogReg)          │
│ - Transformers (T5)                      │
│ - PyTorch                                │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ DVC (Data Version Control)               │
│ - Pipeline: YAML                         │
│ - Cache: Local                           │
│ - Remote: S3                             │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ AWS S3                                   │
│ - Raw data: 65,941 files                 │
│ - Storage: GBs                           │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│ Google Colab (Planificado)               │
│ - GPU: T4                                │
│ - T5 Fine-tuning                         │
└──────────────────────────────────────────┘
```

## Flujo de Datos Completo

```
1. DVC pull           → Descargar raw de S3
2. make_dataset.py    → Limpiar y normalizar
3. split_dataset.py    → Dividir train/test
4. train_classifier   → Clasificar PLS/non_PLS
5. create_pairs       → Generar pares sintéticos
6. train_t5          → Entrenar generador
7. evaluate          → Evaluar con métricas
8. semi_supervised   → Iteración con no etiquetados
```

## Estado del Proyecto

```
┌─────────────────────────────────────────────────┐
│ Semana 0: Setup                                 │
│ - DVC configurado                               │
│ - S3 conectado                                  │
│ - Datos subidos                                 │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│ Semana 1: Preprocesamiento                      │
│ - EDA completo                                  │
│ - 67K registros limpios                         │
│ - Split train/test                              │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│ Semana 2: Clasificador                          │
│ - Baseline: F1 0.9952                           │
│ - Target cumplido                               │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│ Semana 3: Generador                             │
│ - Pares sintéticos generados                    │
│ - T5 código implementado                        │
│ - T5 entrenado en Colab                         │
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│ Semana 4-6: Finalización                        │
│ - Evaluación completa                           │
│ - Análisis de resultados                        │
│ - Documentación                                 │
└─────────────────────────────────────────────────┘
```
