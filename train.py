import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# Silenciamos algunos avisos de Hugging Face
os.environ["WANDB_DISABLED"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main():
    print("--- 1. CARGANDO DATOS LIMPIOS ---")
    # Cargamos los CSVs que generaste en la Fase 1
    train_df = pd.read_csv("data/train.csv").dropna()
    val_df = pd.read_csv("data/val.csv").dropna()
    train_df["text"] = train_df["text"].astype(str)
    train_df["summary"] = train_df["summary"].astype(str)
    val_df["text"] = val_df["text"].astype(str)
    val_df["summary"] = val_df["summary"].astype(str)

    # Hugging Face necesita que los datos estén en su formato "Dataset"
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    print(f"Imágenes para entrenamiento: {len(train_dataset)} | Validación: {len(val_dataset)}")

    print("\n--- 2. CARGANDO TOKENIZADOR Y MODELO (T5-Small) ---")
    MODEL_CHECKPOINT = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # T5 necesita que le digamos explícitamente qué tarea va a hacer
    prefix = "summarize: "

    def preprocess_function(examples):
        # 1. Tokenizamos los artículos (inputs)
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)

        # 2. Tokenizamos los resúmenes (labels)
        labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\n--- 3. TOKENIZANDO LOS DATASETS ---")
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)

    # El DataCollator empareja las longitudes de las frases rellenando con ceros (padding)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    print("\n--- 4. CONFIGURANDO EL ENTRENAMIENTO ---")
    rouge = evaluate.load("rouge") # type: ignore

    def compute_metrics(eval_pred):
        """Calcula la métrica ROUGE durante la evaluación"""
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Reemplazamos los -100 (padding interno) por el token correcto para poder decodificar
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v * 100, 4) for k, v in result.items()} # type: ignore

    output_dir = "./models/t5_bbc_summary"
    
    # Argumentos de entrenamiento optimizados para que no explote la RAM de tu PC
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        # evaluation_strategy="epoch", # Evalúa al final de cada época
        learning_rate=2e-5,
        per_device_train_batch_size=4, # Batch size pequeño por si no tienes GPU muy potente
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3, # 3 épocas es ideal para ver resultados iniciales
        predict_with_generate=True,
        fp16=torch.cuda.is_available(), # Si tienes GPU NVIDIA, el entrenamiento volará
        logging_steps=100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val, # type: ignore
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n--- 5. INICIANDO FINE-TUNING ---")
    trainer.train()

    print("\n--- 6. GUARDANDO MODELO FINAL ---")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n¡Misión cumplida! El modelo se ha guardado en la carpeta '{output_dir}'.")

if __name__ == '__main__':
    main()