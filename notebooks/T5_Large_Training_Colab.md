# Entrenar T5-Large en Google Colab

Este documento contiene el código completo para entrenar T5-Large en Google Colab con todas las mejoras implementadas.

## Pasos para usar en Colab

1. **Subir datos a Colab:**
   - Sube el archivo `data/processed/synthetic_pairs_improved/synthetic_pairs.jsonl` a Colab
   - O monta Google Drive y accede a los datos desde ahí

2. **Ejecutar las siguientes celdas en orden:**

---

## Celda 1: Instalación de dependencias

```python
!pip install -q transformers datasets accelerate rouge-score nltk sacrebleu textstat
```

---

## Celda 2: Imports y configuración

```python
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import nltk
nltk.download('punkt', quiet=True)
from rouge_score import rouge_scorer
from sacrebleu import BLEU

# Verificar GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Configuración
MODEL_CONFIG = {
    'model_name': 't5-large',
    'max_context': 512,
    'max_target': 256,
    'batch_size': 2,  # Ajustar según GPU (T5-large es grande)
    'gradient_accumulation_steps': 16,  # Batch efectivo = 32
    'learning_rate': 3e-5,
    'num_epochs': 4,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'fp16': True,
}

# PROMPT ESTANDARIZADO
STANDARD_PROMPT = "simplify medical text into plain language: "

def apply_prompt(text):
    return STANDARD_PROMPT + text
```

---

## Celda 3: Cargar datos (RESPETANDO SPLITS ORIGINALES)

```python
def load_synthetic_pairs(data_path='synthetic_pairs.jsonl'):
    """Carga pares sintéticos respetando splits originales."""
    print("="*80)
    print("CARGANDO PARES SINTÉTICOS")
    print("="*80)
    
    pairs = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    
    print(f"Total pares: {len(pairs)}")
    
    # CRÍTICO: Usar splits originales
    train_pairs = [p for p in pairs if p.get('split') == 'train']
    test_pairs = [p for p in pairs if p.get('split') == 'test']
    
    print(f"Train (split original): {len(train_pairs)}")
    print(f"Test (split original): {len(test_pairs)}")
    
    # Validar
    valid_train = [p for p in train_pairs 
                   if 'texto_tecnico' in p and 'texto_simple' in p 
                   and len(p['texto_tecnico'].split()) >= 20 
                   and len(p['texto_simple'].split()) >= 10]
    
    valid_test = [p for p in test_pairs 
                  if 'texto_tecnico' in p and 'texto_simple' in p 
                  and len(p['texto_tecnico'].split()) >= 20 
                  and len(p['texto_simple'].split()) >= 10]
    
    print(f"Train válidos: {len(valid_train)}")
    print(f"Test válidos: {len(valid_test)}")
    
    return valid_train, valid_test

# Cargar datos
train_pairs, test_pairs = load_synthetic_pairs('synthetic_pairs.jsonl')
```

---

## Celda 4: Análisis de longitudes

```python
def analyze_lengths(pairs, tokenizer, max_length=512):
    """Analiza distribución de longitudes."""
    lengths = []
    truncated = 0
    
    for pair in pairs:
        tokens = len(tokenizer.encode(pair['texto_tecnico']))
        lengths.append(tokens)
        if tokens > max_length:
            truncated += 1
    
    print(f"Promedio: {np.mean(lengths):.0f} tokens")
    print(f"Mediana: {np.median(lengths):.0f} tokens")
    print(f"Máximo: {np.max(lengths)} tokens")
    print(f"Exceden {max_length}: {truncated} ({100*truncated/len(pairs):.1f}%)")
    
    return lengths

# Cargar tokenizer para análisis
tokenizer = T5Tokenizer.from_pretrained('t5-large')
print("\nANÁLISIS DE LONGITUDES (ANTES DE CHUNKING):")
analyze_lengths(train_pairs + test_pairs, tokenizer)
```

---

## Celda 5: Chunking para documentos largos

```python
def split_into_chunks(text, tokenizer, max_tokens=400, overlap=50):
    """Divide texto en chunks con overlap."""
    paragraphs = text.split('\n\n')
    if not paragraphs:
        paragraphs = text.split('. ')
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        if not para.strip():
            continue
        para_tokens = len(tokenizer.encode(para))
        
        if current_tokens + para_tokens > max_tokens:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                if len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1]]
                    current_tokens = len(tokenizer.encode(current_chunk[0]))
                else:
                    current_chunk = []
                    current_tokens = 0
        current_chunk.append(para)
        current_tokens += para_tokens
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks if chunks else [text]

def expand_with_chunking(pairs, tokenizer, max_tokens=400):
    """Aplica chunking a documentos largos."""
    expanded = []
    chunked = 0
    
    for pair in pairs:
        tech_text = pair['texto_tecnico']
        tokens = len(tokenizer.encode(tech_text))
        
        if tokens > max_tokens:
            chunks = split_into_chunks(tech_text, tokenizer, max_tokens)
            chunked += 1
            for chunk in chunks:
                new_pair = pair.copy()
                new_pair['texto_tecnico'] = chunk
                expanded.append(new_pair)
        else:
            expanded.append(pair)
    
    print(f"Originales: {len(pairs)}")
    print(f"Chunked: {chunked}")
    print(f"Después: {len(expanded)}")
    
    return expanded

# Aplicar chunking
print("\nAPLICANDO CHUNKING:")
train_expanded = expand_with_chunking(train_pairs, tokenizer, max_tokens=450)
test_expanded = expand_with_chunking(test_pairs, tokenizer, max_tokens=450)
```

---

## Celda 6: Preparación de datasets

```python
def tokenize_function(examples, tokenizer, max_source=400, max_target=256):
    """Tokeniza ejemplos con prompt estandarizado."""
    inputs = [apply_prompt(text) for text in examples['texto_tecnico']]
    targets = examples['texto_simple']
    
    model_inputs = tokenizer(inputs, max_length=max_source, truncation=True, padding='max_length')
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target, truncation=True, padding='max_length')
    
    labels['input_ids'] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels['input_ids']
    ]
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Crear datasets
train_dataset = Dataset.from_list(train_expanded)
test_dataset = Dataset.from_list(test_expanded)

# Crear dev desde train (10%)
train_size = len(train_dataset)
dev_size = int(0.1 * train_size)
train_dataset = train_dataset.shuffle(seed=42)
dev_dataset = train_dataset.select(range(dev_size))
train_dataset = train_dataset.select(range(dev_size, train_size))

print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")

# Tokenizar
def tokenize_wrapper(examples):
    return tokenize_function(examples, tokenizer, max_source=450, max_target=256)

train_dataset = train_dataset.map(tokenize_wrapper, batched=True, remove_columns=train_dataset.column_names)
dev_dataset = dev_dataset.map(tokenize_wrapper, batched=True, remove_columns=dev_dataset.column_names)
test_dataset = test_dataset.map(tokenize_wrapper, batched=True, remove_columns=test_dataset.column_names)
```

---

## Celda 7: Métricas completas (ROUGE, BLEU, SARI)

```python
def compute_metrics(eval_pred, tokenizer):
    """Calcula ROUGE, BLEU, SARI."""
    predictions, labels = eval_pred
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = [], [], []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = rouge_scorer_obj.score(label, pred)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)
    
    # BLEU
    bleu = BLEU()
    bleu_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        try:
            score = bleu.sentence_score(pred, [label])
            bleu_scores.append(score.score / 100.0)
        except:
            bleu_scores.append(0.0)
    
    # SARI simplificado
    def sari(pred, ref):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        keep = len(pred_words & ref_words) / max(len(pred_words), 1)
        return keep
    
    sari_scores = [sari(p, l) for p, l in zip(decoded_preds, decoded_labels)]
    
    return {
        'rouge1': np.mean(rouge1),
        'rouge2': np.mean(rouge2),
        'rougeL': np.mean(rougeL),
        'bleu': np.mean(bleu_scores),
        'sari': np.mean(sari_scores)
    }
```

---

## Celda 8: Entrenamiento

```python
# Cargar modelo
print("Cargando T5-large...")
model = T5ForConditionalGeneration.from_pretrained('t5-large')
model = model.to(device)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./t5_large_pls',
    num_train_epochs=MODEL_CONFIG['num_epochs'],
    per_device_train_batch_size=MODEL_CONFIG['batch_size'],
    per_device_eval_batch_size=MODEL_CONFIG['batch_size'],
    gradient_accumulation_steps=MODEL_CONFIG['gradient_accumulation_steps'],
    learning_rate=MODEL_CONFIG['learning_rate'],
    warmup_ratio=MODEL_CONFIG['warmup_ratio'],
    weight_decay=MODEL_CONFIG['weight_decay'],
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    fp16=MODEL_CONFIG['fp16'],
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
    compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Entrenar
print("\n" + "="*80)
print("INICIANDO ENTRENAMIENTO")
print("="*80)
trainer.train()
```

---

## Celda 9: Evaluación en test

```python
# Evaluar en test
print("\nEVALUANDO EN TEST SET:")
test_results = trainer.evaluate(eval_dataset=test_dataset)

print("\nRESULTADOS FINALES:")
print(f"ROUGE-1: {test_results.get('eval_rouge1', 0):.4f}")
print(f"ROUGE-2: {test_results.get('eval_rouge2', 0):.4f}")
print(f"ROUGE-L: {test_results.get('eval_rougeL', 0):.4f}")
print(f"BLEU: {test_results.get('eval_bleu', 0):.4f}")
print(f"SARI: {test_results.get('eval_sari', 0):.4f}")
```

---

## Celda 10: Guardar modelo

```python
# Guardar
trainer.save_model('./t5_large_pls_final')
tokenizer.save_pretrained('./t5_large_pls_final')

# Guardar métricas
with open('./t5_large_pls_final/metrics.json', 'w') as f:
    json.dump(test_results, f, indent=2)

print("✅ Modelo guardado en ./t5_large_pls_final")
```

---

## Celda 11: Probar generación (opcional)

```python
# Probar generación
model.eval()
test_text = "Hypertension is a common cardiovascular condition characterized by elevated blood pressure."

input_text = apply_prompt(test_text)
inputs = tokenizer(input_text, return_tensors="pt", max_length=450, truncation=True).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        length_penalty=1.1,
        early_stopping=True
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Original: {test_text}")
print(f"Generado: {generated}")
```

---

## Notas importantes

1. **Ajustar batch_size**: T5-large es grande. Si tienes problemas de memoria:
   - Reduce `batch_size` a 1
   - Aumenta `gradient_accumulation_steps` para mantener batch efectivo

2. **Guardar en Drive**: Para no perder el modelo:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Cambiar output_dir a '/content/drive/MyDrive/t5_large_pls'
   ```

3. **Monitoreo**: Puedes usar Weights & Biases:
   ```python
   !pip install wandb
   # En training_args: report_to="wandb"
   ```

4. **Tiempo estimado**: Con T5-large y ~18k pares, espera 4-6 horas en GPU T4.

