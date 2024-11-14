from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from utils import read_file, write_file, get_device
from datasets import Dataset
import argparse
import evaluate
import torch
import os
import numpy as np 

MAX_INPUT_LENGTH = 64
MAX_TRANSLATE_LENGTH = 250
checkpoint = "facebook/nllb-200-distilled-600M"
metric = evaluate.load("sacrebleu")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, src_lang="zho_Hans", tgt_lang="eng_Latn")
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

def train(text_path, label_path):
    chinese_sentences = read_file(text_path)
    english_sentences = read_file(label_path)
    
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
        output_dir='nllb_args.pt',
        learning_rate=1e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        fp16=True,
        weight_decay=0.01,
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
    trainer.save_model("nllb.pt") 

def generate_translation(batch, model):
    inputs = tokenizer(batch["zh"], return_tensors="pt", padding=True, truncation=True, max_length=MAX_TRANSLATE_LENGTH).input_ids.to(device)
    outputs = model.generate(inputs)
    return [tokenizer.decode(output, skip_special_tokens=True) + "\n" for output in outputs]

def translate(text_path, use_ft, batch, output_path):
    if use_ft:
        if not os.path.exists("nllb.pt"):
            raise Exception("Requires model to be fine tuned first")
        chosen = "nllb.pt"
    else:
        chosen = checkpoint

    model = AutoModelForSeq2SeqLM.from_pretrained(chosen).to(device)
    test_sentences = read_file(text_path)
    data_dict = {"zh": test_sentences}

    trans_dataset = Dataset.from_dict(data_dict)

    batch_size = batch or 5
    predictions = []

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
    parser.add_argument("-label", help="Label file path containing ideal translated ENGLISH text output")
    parser.add_argument("-ft", type=bool, help="True to use fine-tuned version else use default checkpoint")
    parser.add_argument("-batch", type=int, help="Batch size used during translation")
    parser.add_argument("-out", help="Output file path")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    if args.label: 
       train(args.text, args.label)
    else: 
       translate(args.text, args.ft, args.batch, args.out)
