"""
Evaluador del modelo T5.

Este módulo contiene funciones para evaluar el modelo T5 generando PLS
y calculando métricas de calidad (ROUGE, BLEU, SARI, BERTScore, legibilidad).
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Importar utilidades
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from utils.text_chunking import split_into_chunks
from utils.evaluation_metrics import calculate_all_metrics
from utils.length_analysis import analyze_lengths_for_training
from models.t5.config import apply_prompt, get_prompt, get_t5_generation_config


def load_t5_model(model_path: str = 'models/t5_generator/model', tokenizer_path: str = 'models/t5_generator/tokenizer'):
    """
    Carga el modelo T5 entrenado.
    
    Args:
        model_path: Ruta al modelo entrenado
        tokenizer_path: Ruta al tokenizer
    
    Returns:
        Tupla (model, tokenizer)
    """
    print("Cargando modelo T5...")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def generate_pls(model, tokenizer, technical_text: str, use_chunking: bool = True, max_tokens: int = 400) -> str:
    """
    Genera PLS a partir de texto técnico.
    
    Si el texto es largo, lo divide en chunks, genera PLS para cada chunk,
    y luego combina los resultados.
    
    Args:
        model: Modelo T5 entrenado
        tokenizer: Tokenizer T5
        technical_text: Texto técnico a simplificar
        use_chunking: Si True, usa chunking para textos largos
        max_tokens: Máximo de tokens por chunk (default: 400)
    
    Returns:
        Texto PLS generado
    """
    # Contar tokens del texto técnico
    tech_tokens = len(tokenizer.encode(technical_text, add_special_tokens=False))
    
    # Si el texto es corto o chunking deshabilitado, procesar directamente
    if not use_chunking or tech_tokens <= max_tokens:
        # Usar prompt estándar centralizado
        input_text = apply_prompt(technical_text)
        
        # Obtener configuración de generación
        gen_config = get_t5_generation_config()
        
        # Tokenizar
        inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
        
        # Generar
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=gen_config['max_length'],
                num_beams=gen_config['num_beams'],
                early_stopping=gen_config['early_stopping'],
                no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 3)
            )
        
        # Decodificar
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    
    # Texto largo: dividir en chunks y procesar cada uno
    chunks = split_into_chunks(
        technical_text,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap=50
    )
    
    # Obtener configuración de generación
    gen_config = get_t5_generation_config()
    
    if len(chunks) == 1:
        # Solo un chunk, procesar normalmente
        input_text = apply_prompt(chunks[0])
        inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=gen_config['max_length'],
                num_beams=gen_config['num_beams'],
                early_stopping=gen_config['early_stopping'],
                no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 3)
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Múltiples chunks: generar PLS para cada uno y combinar
    generated_chunks = []
    for i, chunk in enumerate(chunks):
        input_text = apply_prompt(chunk)
        inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=gen_config['max_length'],
                num_beams=gen_config['num_beams'],
                early_stopping=gen_config['early_stopping'],
                no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 3)
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


def evaluate_t5_generator(
    model,
    tokenizer,
    test_pairs: list,
    output_dir: Path = None,
    max_samples: int = None
) -> dict:
    """
    Evalúa el modelo T5 generando PLS y calculando métricas.
    
    Args:
        model: Modelo T5 entrenado
        tokenizer: Tokenizer T5
        test_pairs: Lista de pares de test (texto_tecnico, texto_simple)
        output_dir: Directorio donde guardar resultados
        max_samples: Número máximo de muestras a evaluar (None = todas)
    
    Returns:
        Diccionario con resultados de evaluación
    """
    if output_dir is None:
        output_dir = Path('models/t5_generator/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ANÁLISIS DE LONGITUDES DEL TEST SET
    print("\n" + "="*80)
    print("ANÁLISIS DE LONGITUDES DEL TEST SET")
    print("="*80)
    
    analyze_lengths_for_training(
        pairs=test_pairs,
        tokenizer=tokenizer,
        max_length_source=400,
        save_report=True,
        output_dir=output_dir / 'length_analysis'
    )
    
    # Limitar muestras si se especifica
    if max_samples:
        test_pairs = test_pairs[:max_samples]
        print(f"\nEvaluando con {len(test_pairs)} ejemplos...")
    
    # Generar PLS
    print("\nGenerando PLS...")
    predictions = []
    references = []
    
    for pair in tqdm(test_pairs, desc="Generando PLS"):
        technical = pair['texto_tecnico']
        simple = pair['texto_simple']
        
        # Generar PLS
        generated = generate_pls(model, tokenizer, technical)
        
        predictions.append(generated)
        references.append(simple)
    
    print(f"\nGeneración completada: {len(predictions)} ejemplos")
    
    # Preparar sources para SARI (textos técnicos originales)
    sources = [pair['texto_tecnico'] for pair in test_pairs]
    
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
    
    # BERTScore
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
        
        if 'flesch_kincaid' in read:
            print(f"Flesch-Kincaid Grade Level: {read['flesch_kincaid']['avg_grade_level']:.2f}")
    
    # Preparar resultados para guardar
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
    
    # Guardar resultados
    results_to_save['full_results'] = results
    
    output_path = output_dir / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    
    print(f"\nResultados guardados en: {output_path}")
    
    # Guardar ejemplos
    examples = []
    for i, (pred, ref, tech) in enumerate(zip(predictions[:10], references[:10], sources[:10])):
        examples.append({
            'idx': i,
            'technical': tech[:200],
            'generated': pred[:200],
            'reference': ref[:200]
        })
    
    examples_path = output_dir / 'evaluation_examples.json'
    with open(examples_path, 'w') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"Ejemplos guardados en: {examples_path}")
    
    return results_to_save

