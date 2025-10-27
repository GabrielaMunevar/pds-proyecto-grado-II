#!/usr/bin/env python3
"""
Evaluar Generador T5
Calcula métricas ROUGE y BERTScore para el modelo T5 entrenado.

Uso:
    python src/models/evaluate_generator.py
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json
import pandas as pd
from pathlib import Path
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except:
    ROUGE_AVAILABLE = False
    print("rouge_score no disponible. Instalar: pip install rouge-score")

try:
    from bert_score import score
    BERTSCORE_AVAILABLE = True
except:
    BERTSCORE_AVAILABLE = False
    print("bert_score no disponible. Instalar: pip install bert-score")

def loadModel():
    """Carga el modelo T5 entrenado."""
    print("Cargando modelo T5...")
    model = T5ForConditionalGeneration.from_pretrained('models/t5_generator/model')
    tokenizer = T5Tokenizer.from_pretrained('models/t5_generator/tokenizer')
    return model, tokenizer

def loadSyntheticPairs():
    """Carga los pares sintéticos para evaluación."""
    print("Cargando pares sintéticos...")
    
    pairs = []
    with open('data/processed/synthetic_pairs/synthetic_pairs.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            pairs.append(json.loads(line))
    
    # Filtrar solo test
    testPairs = [p for p in pairs if p['split'] == 'test']
    
    print(f"Pares de test: {len(testPairs)}")
    
    return testPairs

def generatePLS(model, tokenizer, technicalText):
    """Genera PLS a partir de texto técnico."""
    # Agregar prefijo "simplify: "
    inputText = f"simplify: {technicalText}"
    
    # Tokenizar
    inputs = tokenizer(inputText, return_tensors='pt', max_length=256, truncation=True)
    
    # Generar
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    # Decodificar
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated

def calculateROUGE(predictions, references):
    """Calcula métricas ROUGE."""
    if not ROUGE_AVAILABLE:
        return None
    
    print("Calculando ROUGE...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rougeScores = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions)):
        scores = scorer.score(ref, pred)
        rougeScores.append({
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        })
    
    # Promedios
    avgRouge1 = sum([r['rouge1'] for r in rougeScores]) / len(rougeScores)
    avgRouge2 = sum([r['rouge2'] for r in rougeScores]) / len(rougeScores)
    avgRougeL = sum([r['rougeL'] for r in rougeScores]) / len(rougeScores)
    
    return {
        'rouge1': avgRouge1,
        'rouge2': avgRouge2,
        'rougeL': avgRougeL,
        'individual_scores': rougeScores
    }

def calculateBERTScore(predictions, references):
    """Calcula BERTScore."""
    if not BERTSCORE_AVAILABLE:
        return None
    
    print("Calculando BERTScore...")
    
    # Calcular en batches
    P, R, F1 = score(predictions, references, lang='en', verbose=True)
    
    avgP = P.mean().item()
    avgR = R.mean().item()
    avgF1 = F1.mean().item()
    
    return {
        'precision': avgP,
        'recall': avgR,
        'f1': avgF1
    }

def main():
    """Función principal de evaluación."""
    print("=== EVALUACIÓN DEL GENERADOR T5 ===\n")
    
    # Cargar modelo
    model, tokenizer = loadModel()
    
    # Cargar datos de test
    testPairs = loadSyntheticPairs()
    
    # Limitar para evaluación rápida (cambiar por todas si quieres)
    print(f"\nEvaluando con {min(100, len(testPairs))} ejemplos...")
    testSample = testPairs[:100]  # Primera evaluación rápida
    
    # Generar PLS
    print("\nGenerando PLS...")
    predictions = []
    references = []
    
    for pair in tqdm(testSample):
        technical = pair['texto_tecnico']
        simple = pair['texto_simple']
        
        # Generar PLS
        generated = generatePLS(model, tokenizer, technical)
        
        predictions.append(generated)
        references.append(simple)
    
    print(f"\nGeneración completada: {len(predictions)} ejemplos")
    
    # Calcular métricas
    results = {}
    
    # ROUGE
    if ROUGE_AVAILABLE:
        rougeResults = calculateROUGE(predictions, references)
        if rougeResults:
            results['rouge'] = {
                'rouge1': float(rougeResults['rouge1']),
                'rouge2': float(rougeResults['rouge2']),
                'rougeL': float(rougeResults['rougeL'])
            }
            
            print(f"\n=== RESULTADOS ROUGE ===")
            print(f"ROUGE-1: {rougeResults['rouge1']:.4f}")
            print(f"ROUGE-2: {rougeResults['rouge2']:.4f}")
            print(f"ROUGE-L: {rougeResults['rougeL']:.4f}")
    
    # BERTScore
    if BERTSCORE_AVAILABLE:
        bertResults = calculateBERTScore(predictions, references)
        if bertResults:
            results['bertscore'] = {
                'precision': float(bertResults['precision']),
                'recall': float(bertResults['recall']),
                'f1': float(bertResults['f1'])
            }
            
            print(f"\n=== RESULTADOS BERTScore ===")
            print(f"Precision: {bertResults['precision']:.4f}")
            print(f"Recall: {bertResults['recall']:.4f}")
            print(f"F1: {bertResults['f1']:.4f}")
    
    # Guardar resultados
    outputPath = Path('models/t5_generator/evaluation_results.json')
    with open(outputPath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResultados guardados en: {outputPath}")
    
    # Guardar ejemplos
    examples = []
    for i, (pred, ref, tech) in enumerate(zip(predictions[:10], references[:10], [p['texto_tecnico'] for p in testSample[:10]])):
        examples.append({
            'idx': i,
            'technical': tech[:200],
            'generated': pred[:200],
            'reference': ref[:200]
        })
    
    with open('models/t5_generator/evaluation_examples.json', 'w') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print("Ejemplos guardados en: models/t5_generator/evaluation_examples.json")
    
    print("\n=== RESUMEN ===")
    if 'rouge' in results:
        print(f"ROUGE-L: {results['rouge']['rougeL']:.4f}")
    if 'bertscore' in results:
        print(f"BERTScore F1: {results['bertscore']['f1']:.4f}")
    
    # Target
    targetRougeL = 0.35
    if 'rouge' in results and results['rouge']['rougeL'] >= targetRougeL:
        print(f"\nTARGET CUMPLIDO: ROUGE-L {results['rouge']['rougeL']:.4f} >= {targetRougeL}")
    elif 'rouge' in results:
        print(f"\nTARGET NO CUMPLIDO: ROUGE-L {results['rouge']['rougeL']:.4f} < {targetRougeL}")

if __name__ == "__main__":
    main()
