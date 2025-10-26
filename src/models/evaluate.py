"""
Script de evaluación automática de PLS generados.

Con ground truth:
- ROUGE-L
- BERTScore (F1)
- Ratio de compresión/expansión
- Flesch Reading Ease (mejora esperada)

Sin ground truth:
- Legibilidad
- Compresión
- Chequeos heurísticos
- QA opcional para relevancia
"""

import yaml
import json
import pandas as pd
import logging
from pathlib import Path
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from textstat import flesch_reading_ease

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params():
    """Carga parámetros desde params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate_with_reference(predictions, references, originals=None):
    """
    Evalúa PLS generados contra referencias (ground truth).
    
    Args:
        predictions: Lista de PLS generados
        references: Lista de PLS de referencia
        originals: Lista de textos originales (opcional)
    
    Returns:
        metrics: Diccionario con métricas agregadas
    """
    logger.info("Evaluando con referencias...")
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)['rougeL'].fmeasure 
                   for ref, pred in zip(references, predictions)]
    
    # BERTScore
    P, R, F1 = bertscore(predictions, references, lang="en", verbose=False)
    
    # Legibilidad
    flesch_refs = [flesch_reading_ease(ref) for ref in references]
    flesch_preds = [flesch_reading_ease(pred) for pred in predictions]
    flesch_deltas = [p - r for p, r in zip(flesch_preds, flesch_refs)]
    
    # Compresión
    if originals:
        compression_ratios = [len(pred) / len(orig) if len(orig) > 0 else 0
                             for pred, orig in zip(predictions, originals)]
    else:
        compression_ratios = []
    
    metrics = {
        "rouge_l": {
            "mean": sum(rouge_scores) / len(rouge_scores),
            "scores": rouge_scores
        },
        "bertscore": {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
        },
        "flesch": {
            "refs_mean": sum(flesch_refs) / len(flesch_refs),
            "preds_mean": sum(flesch_preds) / len(flesch_preds),
            "delta_mean": sum(flesch_deltas) / len(flesch_deltas),
        }
    }
    
    if compression_ratios:
        metrics["compression"] = {
            "mean": sum(compression_ratios) / len(compression_ratios),
            "ratios": compression_ratios
        }
    
    logger.info(f"ROUGE-L: {metrics['rouge_l']['mean']:.4f}")
    logger.info(f"BERTScore F1: {metrics['bertscore']['f1']:.4f}")
    logger.info(f"Flesch Δ: {metrics['flesch']['delta_mean']:.2f}")
    
    return metrics


def evaluate_without_reference(predictions, originals):
    """
    Evalúa PLS generados sin referencias (para sintéticos).
    
    Args:
        predictions: Lista de PLS generados
        originals: Lista de textos originales
    
    Returns:
        metrics: Diccionario con métricas
    """
    logger.info("Evaluando sin referencias...")
    
    # Legibilidad
    flesch_originals = [flesch_reading_ease(orig) for orig in originals]
    flesch_preds = [flesch_reading_ease(pred) for pred in predictions]
    flesch_deltas = [p - o for p, o in zip(flesch_preds, flesch_originals)]
    
    # Compresión
    compression_ratios = [len(pred) / len(orig) if len(orig) > 0 else 0
                         for pred, orig in zip(predictions, originals)]
    
    # TODO: Implementar chequeos heurísticos adicionales
    # - Detección de frases muy largas
    # - Símbolos anómalos
    # - Coherencia básica
    
    metrics = {
        "flesch": {
            "originals_mean": sum(flesch_originals) / len(flesch_originals),
            "preds_mean": sum(flesch_preds) / len(flesch_preds),
            "delta_mean": sum(flesch_deltas) / len(flesch_deltas),
        },
        "compression": {
            "mean": sum(compression_ratios) / len(compression_ratios),
            "ratios": compression_ratios
        }
    }
    
    logger.info(f"Flesch Original: {metrics['flesch']['originals_mean']:.2f}")
    logger.info(f"Flesch PLS: {metrics['flesch']['preds_mean']:.2f}")
    logger.info(f"Flesch Δ: {metrics['flesch']['delta_mean']:.2f}")
    logger.info(f"Compresión: {metrics['compression']['mean']:.2f}")
    
    return metrics


def compare_by_source(df, metrics_col="metrics"):
    """Compara métricas por fuente de datos"""
    if "source" not in df.columns:
        logger.warning("Columna 'source' no disponible")
        return {}
    
    logger.info("Comparando por fuente...")
    
    sources = df["source"].unique()
    comparison = {}
    
    for source in sources:
        source_df = df[df["source"] == source]
        # TODO: Calcular métricas por fuente
        comparison[source] = {
            "count": len(source_df),
            # Agregar métricas específicas
        }
    
    return comparison


def main():
    """Pipeline principal de evaluación"""
    params = load_params()
    
    logger.info("Iniciando evaluación...")
    
    # Cargar datos de prueba
    test_df = pd.read_parquet("data/processed/test.parquet")
    
    # TODO: Cargar predicciones generadas
    # predictions = load_predictions("data/outputs/supervised/predictions.jsonl")
    
    # TODO: Evaluar con referencias (si existen)
    # metrics_with_ref = evaluate_with_reference(...)
    
    # TODO: Evaluar sin referencias (sintéticos)
    # metrics_without_ref = evaluate_without_reference(...)
    
    # TODO: Comparar por fuente
    # comparison = compare_by_source(test_df)
    
    # Guardar reporte
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "test_samples": len(test_df),
        # "metrics": metrics,
        # "by_source": comparison,
    }
    
    output_path = Path("data/evaluation/report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Reporte guardado en {output_path}")


if __name__ == "__main__":
    main()


