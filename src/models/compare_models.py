#!/usr/bin/env python3
"""
Comparación de Modelos T5
==========================

Compara el modelo baseline con el modelo mejorado entrenado.

Uso:
    python src/models/compare_models.py
"""

import torch
import json
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import pandas as pd
import sys

# Importar configuración
sys.path.append(str(Path(__file__).parent.parent))
from config import apply_prompt, get_prompt

def load_model(model_path, tokenizer_path, device):
    """Cargar modelo y tokenizer"""
    print(f"Cargando modelo desde: {model_path}")
    
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    
    model = model.to(device)
    model.eval()
    
    print(f"Modelo cargado en {device}")
    
    return model, tokenizer

def generate_pls(model, tokenizer, input_text, device, max_length=256):
    """Generar PLS con modelo"""
    # Usar prompt estándar centralizado
    full_input = apply_prompt(input_text)
    
    inputs = tokenizer(
        full_input,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            length_penalty=1.1,
            do_sample=False
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def load_test_examples(n=10):
    """Cargar ejemplos de test"""
    test_file = Path('data/processed/synthetic_pairs/synthetic_pairs.jsonl')
    
    examples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            pair = json.loads(line)
            examples.append({
                'tecnico': pair['texto_tecnico'],
                'simple': pair['texto_simple']
            })
    
    return examples

def calculate_simple_metrics(text):
    """Calcular métricas simples"""
    words = text.split()
    sentences = text.split('.')
    
    return {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0
    }

def compare_models(baseline_model_path, improved_model_path, test_examples):
    """Comparar dos modelos"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar modelos
    baseline_model, baseline_tokenizer = load_model(
        baseline_model_path,
        baseline_model_path,
        device
    )
    
    improved_model, improved_tokenizer = load_model(
        improved_model_path,
        improved_model_path,
        device
    )
    
    results = []
    
    print("\n" + "=" * 80)
    print("GENERANDO PREDICCIONES")
    print("=" * 80)
    
    for i, example in enumerate(tqdm(test_examples, desc="Procesando")):
        tecnico = example['tecnico']
        ground_truth = example['simple']
        
        # Generar con modelo baseline
        baseline_pred = generate_pls(
            baseline_model, baseline_tokenizer, tecnico, device
        )
        
        # Generar con modelo mejorado
        improved_pred = generate_pls(
            improved_model, improved_tokenizer, tecnico, device
        )
        
        # Métricas baseline
        baseline_metrics = calculate_simple_metrics(baseline_pred)
        ground_truth_metrics = calculate_simple_metrics(ground_truth)
        
        # Comparación con ground truth
        baseline_word_diff = abs(baseline_metrics['word_count'] - ground_truth_metrics['word_count'])
        improved_metrics = calculate_simple_metrics(improved_pred)
        improved_word_diff = abs(improved_metrics['word_count'] - ground_truth_metrics['word_count'])
        
        results.append({
            'example_id': i,
            'tecnico': tecnico[:200] + "...",
            'ground_truth': ground_truth[:200] + "...",
            'baseline_pred': baseline_pred[:200] + "...",
            'improved_pred': improved_pred[:200] + "...",
            'baseline_word_count': baseline_metrics['word_count'],
            'improved_word_count': improved_metrics['word_count'],
            'ground_truth_word_count': ground_truth_metrics['word_count'],
            'baseline_word_diff': baseline_word_diff,
            'improved_word_diff': improved_word_diff,
            'improvement': baseline_word_diff - improved_word_diff
        })
    
    return results

def save_results(results, output_path):
    """Guardar resultados"""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    print(f"\nResultados guardados en: {output_path}")
    
    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE COMPARACIÓN")
    print("=" * 80)
    
    avg_baseline_diff = df['baseline_word_diff'].mean()
    avg_improved_diff = df['improved_word_diff'].mean()
    improvement = avg_baseline_diff - avg_improved_diff
    
    print(f"\nDiferencia promedio vs ground truth:")
    print(f"  Baseline: {avg_baseline_diff:.1f} palabras")
    print(f"  Mejorado: {avg_improved_diff:.1f} palabras")
    print(f"  Mejora: {improvement:+.1f} palabras")
    
    positive_improvement = (df['improvement'] > 0).sum()
    print(f"\nEjemplos mejorados: {positive_improvement}/{len(df)} ({positive_improvement/len(df)*100:.1f}%)")

def main():
    """Función principal"""
    print("=" * 80)
    print("COMPARACIÓN DE MODELOS T5")
    print("=" * 80)
    
    # Rutas
    baseline_path = Path('models/t5_generator/model')  # Modelo actual
    improved_path = Path('models/t5_generator/final_model')  # Modelo mejorado
    
    # Verificar existencia
    if not baseline_path.exists():
        print(f"Modelo baseline no encontrado: {baseline_path}")
        return
    
    if not improved_path.exists():
        print(f"Modelo mejorado no encontrado aún: {improved_path}")
        print("   Esperando a que termine el entrenamiento en Colab...")
        return
    
    # Cargar ejemplos de test
    test_examples = load_test_examples(n=20)
    
    # Comparar
    results = compare_models(baseline_path, improved_path, test_examples)
    
    # Guardar
    output_path = Path('reports/model_comparison.csv')
    output_path.parent.mkdir(exist_ok=True)
    save_results(results, output_path)
    
    print("\n" + "=" * 80)
    print("COMPARACIÓN COMPLETADA")
    print("=" * 80)

if __name__ == '__main__':
    main()

