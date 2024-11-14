from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from utils import read_file, write_file, get_device
from datasets import Dataset
from itertools import chain
import numpy as np
import argparse
import evaluate
import torch
import os

MAX_INPUT_LENGTH = 64
checkpoint = "Helsinki-NLP/opus-mt-zh-en"
metric = evaluate.load("sacrebleu")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_tensors="pt")
device = get_device()

def tokenize_function(sentences):
    model_inputs = tokenizer(sentences["zh"], text_target=sentences["en"], padding="max_length", truncation=True, max_length=MAX_INPUT_LENGTH)
    
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": model_inputs["labels"],
    }

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    bleu_score = metric.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    return {"bleu": bleu_score}

def preprocess_logits_for_metrics(logits, labels):
    logits = logits[0]
    logits = torch.argmax(logits, dim=-1)
    return logits

def train(text_path, label_paths):
    multiplier = len(label_paths)
    chinese_sentences = list(chain.from_iterable([read_file(text_path)] * multiplier))
    english_sentences = list(chain.from_iterable([read_file(label_path) for label_path in label_paths]))

    data_dict = {"zh": chinese_sentences, "en": english_sentences}
    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.map(tokenize_function, batched=True, batch_size=5)

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["k_proj", "v_proj", "q_proj"]
    )

    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

    training_arguments = TrainingArguments(
        output_dir="meta_args.pt",
        learning_rate=1e-3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model, padding=True),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()
    trainer.save_model("meta.pt") 

def generate_translation(batch, model):
    inputs = tokenizer(batch["zh"], return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH).input_ids.to(device)
    outputs = model.generate(inputs)
    return [tokenizer.decode(output, skip_special_tokens=True) + "\n" for output in outputs]


def translate(text_path, batch, output_path):
    if not os.path.exists("meta.pt"):
        raise Exception("Requires model to be fine tuned first")
    model = AutoModelForSeq2SeqLM.from_pretrained("meta.pt").to(device)
    batch_size = batch or 20
    predictions = []

    test_sentences = read_file(text_path)
    data_dict = {"zh": test_sentences}

    trans_dataset = Dataset.from_dict(data_dict)
        
    for i in range(0, len(trans_dataset), batch_size):
        if i % 100 == 0:
            print(i)
            
        batch = trans_dataset[i: i + batch_size]
        pred = generate_translation(batch, model)
        predictions.extend(pred)

    write_file(output_path, predictions)

    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-text", help="Text file path containing untranslated CHINESE text", required=True)
    parser.add_argument("-label", nargs="+", help="List of prediction file paths containing translated ENGLISH text")
    parser.add_argument("-batch", type=int, help="Batch size used during translation")
    parser.add_argument("-out", help="Output file path")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    if args.label:
        train(args.text, args.label)
    else:
        translate(args.text, args.batch, args.out)