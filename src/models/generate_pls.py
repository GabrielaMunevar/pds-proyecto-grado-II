"""
Script de generación de PLS (supervisado).

Entrada: data/processed/{train,dev}.parquet (solo con has_pair=True)
Salida: models/generator_sft/*, data/outputs/supervised/*.jsonl

Modelos soportados:
- BART (base/large-cnn)
- T5 (base/large)
- LED (base)
- LongT5
- Variantes bio: BioBART, BioT5

Modos:
- train: Entrenamiento supervisado con pares reales
- generate: Generación de PLS con modelo entrenado
"""

import yaml
import pandas as pd
import logging
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params():
    """Carga parámetros desde params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def prepare_pairs(df):
    """Prepara pares (texto_original, summary) para entrenamiento"""
    # Filtrar solo filas con pares
    df_pairs = df[df.get("has_pair", False) == True].copy()
    
    logger.info(f"Pares disponibles: {len(df_pairs)}")
    
    # TODO: Implementar matching extra por ID (NCT)
    # TODO: Implementar chunking para textos largos
    
    return df_pairs


def train_generator(params, train_pairs, dev_pairs):
    """Entrena el generador de PLS"""
    model_name = params["generator"]["base_model"]
    
    logger.info(f"Cargando modelo: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Parámetros de entrenamiento
    sft_params = params["generator"]["sft"]
    peft_params = params["generator"].get("peft", {})
    
    # TODO: Implementar PEFT/LoRA si está configurado
    if peft_params.get("use_lora", False):
        logger.info("Configurando LoRA...")
        # from peft import get_peft_model, LoraConfig, TaskType
        # lora_config = LoraConfig(...)
        # model = get_peft_model(model, lora_config)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="models/generator_sft",
        learning_rate=sft_params["lr"],
        num_train_epochs=sft_params["epochs"],
        per_device_train_batch_size=sft_params["batch_size"],
        per_device_eval_batch_size=sft_params["batch_size"],
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="models/generator_sft/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=sft_params.get("fp16", False),
    )
    
    # TODO: Crear datasets PyTorch
    # TODO: Implementar prompt base
    # "Explica población, intervención, comparador y desenlaces con lenguaje cotidiano."
    
    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=dev_dataset,
    #     tokenizer=tokenizer,
    # )
    
    # logger.info("Iniciando entrenamiento...")
    # trainer.train()
    
    return model, tokenizer


def generate_pls(model, tokenizer, texts, params):
    """Genera PLS para una lista de textos"""
    decoding_params = params["generator"]["decoding"]
    
    # TODO: Implementar generación con beam search
    # beam_size = decoding_params["beam_size"]
    # length_penalty = decoding_params["length_penalty"]
    # no_repeat_ngram_size = decoding_params["no_repeat_ngram_size"]
    
    logger.info(f"Generando PLS para {len(texts)} textos...")
    
    # TODO: Implementar generación batch
    
    return []


def main():
    """Pipeline principal"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "generate"], default="train")
    args = parser.parse_args()
    
    params = load_params()
    
    if args.mode == "train":
        logger.info("Modo: Entrenamiento supervisado")
        
        # Cargar datos
        train_df = pd.read_parquet("data/processed/train.parquet")
        dev_df = pd.read_parquet("data/processed/dev.parquet")
        
        # Preparar pares
        train_pairs = prepare_pairs(train_df)
        dev_pairs = prepare_pairs(dev_df)
        
        if len(train_pairs) == 0:
            logger.error("No hay pares de entrenamiento disponibles")
            return
        
        # Entrenar
        model, tokenizer = train_generator(params, train_pairs, dev_pairs)
        
        # Guardar
        model.save_pretrained("models/generator_sft")
        tokenizer.save_pretrained("models/generator_sft")
        
        logger.info("Generador entrenado y guardado")
    
    elif args.mode == "generate":
        logger.info("Modo: Generación de PLS")
        
        # TODO: Implementar modo de generación
        # Cargar modelo entrenado
        # Generar PLS para textos nuevos
        # Guardar en data/outputs/supervised/
        
        pass


if __name__ == "__main__":
    main()


