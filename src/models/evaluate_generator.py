#!/usr/bin/env python3
"""
Evaluar Generador T5
Calcula métricas completas: ROUGE, BLEU, SARI, BERTScore (best score) y legibilidad.

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

# Importar utilidades y configuración
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.text_chunking import split_into_chunks
from utils.evaluation_metrics import calculate_all_metrics
from utils.length_analysis import analyze_lengths_for_training
from config import apply_prompt, get_prompt

warnings.filterwarnings('ignore')

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

def generatePLS(model, tokenizer, technicalText, use_chunking=True, max_tokens=400):
    """
    Genera PLS a partir de texto técnico.
    
    Si el texto es largo, lo divide en chunks, genera PLS para cada chunk,
    y luego combina los resultados.
    
    Args:
        model: Modelo T5 entrenado
        tokenizer: Tokenizer T5
        technicalText: Texto técnico a simplificar
        use_chunking: Si True, usa chunking para textos largos
        max_tokens: Máximo de tokens por chunk (default: 400)
    
    Returns:
        Texto PLS generado
    """
    # Contar tokens del texto técnico
    tech_tokens = len(tokenizer.encode(technicalText, add_special_tokens=False))
    
    # Si el texto es corto o chunking deshabilitado, procesar directamente
    if not use_chunking or tech_tokens <= max_tokens:
        # Usar prompt estándar centralizado
        inputText = apply_prompt(technicalText)
        
        # Tokenizar
        inputs = tokenizer(inputText, return_tensors='pt', max_length=512, truncation=True)
        
        # Generar
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
        
        # Decodificar
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    
    # Texto largo: dividir en chunks y procesar cada uno
    chunks = split_into_chunks(
        technicalText,
        tokenizer=tokenizer.encode,
        max_tokens=max_tokens,
        overlap=50
    )
    
    if len(chunks) == 1:
        # Solo un chunk, procesar normalmente
        inputText = apply_prompt(chunks[0])
        inputs = tokenizer(inputText, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Múltiples chunks: generar PLS para cada uno y combinar
    generated_chunks = []
    for i, chunk in enumerate(chunks):
        inputText = apply_prompt(chunk)
        inputs = tokenizer(inputText, return_tensors='pt', max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
        
        chunk_pls = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_chunks.append(chunk_pls)
    
    # Combinar chunks generados
    # Estrategia simple: unir con espacios y eliminar duplicados cercanos
    combined = ' '.join(generated_chunks)
    
    # Limpiar: eliminar repeticiones excesivas de palabras
    words = combined.split()
    cleaned_words = []
    prev_word = None
    for word in words:
        if word != prev_word or (prev_word and len(cleaned_words) > 0 and cleaned_words[-1] != word):
            cleaned_words.append(word)
        prev_word = word
    
    return ' '.join(cleaned_words)


def main():
    """Función principal de evaluación."""
    print("=== EVALUACIÓN DEL GENERADOR T5 ===\n")
    
    # Cargar modelo
    model, tokenizer = loadModel()
    
    # Cargar datos de test
    testPairs = loadSyntheticPairs()
    
    # ANÁLISIS DE LONGITUDES DEL TEST SET
    print("\n" + "="*80)
    print("ANÁLISIS DE LONGITUDES DEL TEST SET")
    print("="*80)
    
    analyze_lengths_for_training(
        pairs=testPairs,
        tokenizer=tokenizer,
        max_length_source=400,
        save_report=True,
        output_dir=Path('models/t5_generator')
    )
    
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
    
    # Preparar sources para SARI (textos técnicos originales)
    sources = [pair['texto_tecnico'] for pair in testSample]
    
    # Calcular TODAS las métricas
    print("\n" + "="*80)
    print("CALCULANDO MÉTRICAS COMPLETAS")
    print("="*80)
    
    results = calculate_all_metrics(
        predictions=predictions,
        references=references,
        sources=sources,
        include_readability=True
    )
    
    # Mostrar resultados
    print("\n" + "="*80)
    print("RESULTADOS DE EVALUACIÓN")
    print("="*80)
    
    # ROUGE
    if 'rouge' in results:
        print(f"\n=== ROUGE ===")
        print(f"ROUGE-1 F1: {results['rouge']['rouge1_f']:.4f}")
        print(f"ROUGE-2 F1: {results['rouge']['rouge2_f']:.4f}")
        print(f"ROUGE-L F1: {results['rouge']['rougeL_f']:.4f}")
        print(f"ROUGE-Lsum F1: {results['rouge']['rougeLsum_f']:.4f}")
    
    # BLEU
    if 'bleu' in results:
        print(f"\n=== BLEU ===")
        print(f"BLEU-1: {results['bleu']['bleu1']:.4f}")
        print(f"BLEU-2: {results['bleu']['bleu2']:.4f}")
        print(f"BLEU-3: {results['bleu']['bleu3']:.4f}")
        print(f"BLEU-4: {results['bleu']['bleu4']:.4f}")
    
    # SARI
    if 'sari' in results:
        print(f"\n=== SARI ===")
        print(f"SARI: {results['sari']['sari']:.4f}")
        print(f"  Keep: {results['sari']['keep']:.4f}")
        print(f"  Add: {results['sari']['add']:.4f}")
        print(f"  Delete: {results['sari']['delete']:.4f}")
    
    # BERTScore (Best Score)
    if 'bertscore' in results:
        print(f"\n=== BERTScore (Best Score) ===")
        print(f"Precision: {results['bertscore']['precision']:.4f}")
        print(f"Recall: {results['bertscore']['recall']:.4f}")
        print(f"F1: {results['bertscore']['f1']:.4f}")
    
    # Readability
    if 'readability' in results:
        print(f"\n=== MÉTRICAS DE LEGIBILIDAD ===")
        read = results['readability']
        
        if 'flesch_reading_ease' in read:
            print(f"Flesch Reading Ease: {read['flesch_reading_ease']:.2f}")
            print(f"  (Mayor = más fácil de leer, rango 0-100)")
        
        if 'flesch_kincaid' in read:
            print(f"Flesch-Kincaid Grade Level: {read['flesch_kincaid']['avg_grade_level']:.2f}")
            print(f"  (Años de educación necesarios)")
        
        if 'gunning_fog' in read:
            print(f"Gunning Fog Index: {read['gunning_fog']['avg_grade_level']:.2f}")
        
        if 'dale_chall' in read:
            print(f"Dale-Chall Score: {read['dale_chall']['avg_score']:.2f}")
        
        if 'ari' in read:
            print(f"Automated Readability Index (ARI): {read['ari']['avg_grade_level']:.2f}")
        
        if 'smog' in read:
            print(f"SMOG Index: {read['smog']['avg_grade_level']:.2f}")
    
    # Preparar resultados para guardar (solo valores principales)
    results_to_save = {}
    
    if 'rouge' in results:
        results_to_save['rouge'] = {
            'rouge1_f': results['rouge']['rouge1_f'],
            'rouge2_f': results['rouge']['rouge2_f'],
            'rougeL_f': results['rouge']['rougeL_f'],
            'rougeLsum_f': results['rouge']['rougeLsum_f']
        }
    
    if 'bleu' in results:
        results_to_save['bleu'] = {
            'bleu1': results['bleu']['bleu1'],
            'bleu2': results['bleu']['bleu2'],
            'bleu3': results['bleu']['bleu3'],
            'bleu4': results['bleu']['bleu4']
        }
    
    if 'sari' in results:
        results_to_save['sari'] = {
            'sari': results['sari']['sari'],
            'keep': results['sari']['keep'],
            'add': results['sari']['add'],
            'delete': results['sari']['delete']
        }
    
    if 'bertscore' in results:
        results_to_save['bertscore'] = {
            'precision': results['bertscore']['precision'],
            'recall': results['bertscore']['recall'],
            'f1': results['bertscore']['f1']
        }
    
    if 'readability' in results:
        results_to_save['readability'] = results['readability']
    
    # Guardar resultados (versión simplificada + completos)
    results_to_save['full_results'] = results  # Incluir resultados completos
    
    outputPath = Path('models/t5_generator/evaluation_results.json')
    with open(outputPath, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
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
    
    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)
    
    if 'rouge' in results_to_save:
        print(f"ROUGE-L: {results_to_save['rouge']['rougeL_f']:.4f}")
    if 'bleu' in results_to_save:
        print(f"BLEU-4: {results_to_save['bleu']['bleu4']:.4f}")
    if 'sari' in results_to_save:
        print(f"SARI: {results_to_save['sari']['sari']:.4f}")
    if 'bertscore' in results_to_save:
        print(f"BERTScore F1 (Best Score): {results_to_save['bertscore']['f1']:.4f}")
    
    # Target
    targetRougeL = 0.35
    if 'rouge' in results_to_save and results_to_save['rouge']['rougeL_f'] >= targetRougeL:
        print(f"\n✅ TARGET CUMPLIDO: ROUGE-L {results_to_save['rouge']['rougeL_f']:.4f} >= {targetRougeL}")
    elif 'rouge' in results_to_save:
        print(f"\n❌ TARGET NO CUMPLIDO: ROUGE-L {results_to_save['rouge']['rougeL_f']:.4f} < {targetRougeL}")

if __name__ == "__main__":
    main()
