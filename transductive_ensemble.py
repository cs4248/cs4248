from transformers import pipeline, DataCollatorForSeq2Seq, MBartForConditionalGeneration, MBart50TokenizerFast, AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from utils import read_file
from datasets import Dataset
from itertools import chain
import numpy as np
import argparse
import evaluate
import torch

MAX_INPUT_LENGTH = 64
# checkpoint = "facebook/mbart-large-50-many-to-many-mmt"
checkpoint = "Helsinki-NLP/opus-mt-zh-en"
metric = evaluate.load("sacrebleu")
# tokenizer = MBart50TokenizerFast.from_pretrained(checkpoint, src_lang="zh_CN", tgt_lang="en_XX")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_tensors="pt")

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=MAX_INPUT_LENGTH):
        if labels:
            model_inputs = tokenizer(
                text=texts, 
                text_target=labels, 
                max_length=max_length, 
                truncation=True, 
                padding=True, 
                return_tensors="pt"
            )
            self.labels = model_inputs["labels"]
        else:
            model_inputs = tokenizer(
                text=texts, 
                max_length=max_length, 
                truncation=True, 
                padding=True, 
                return_tensors="pt"
            )
            self.labels = None

        self.texts = model_inputs["input_ids"]
        self.masks = model_inputs["attention_mask"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        mask = self.masks[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return {"input_ids": text, "attention_mask": mask, "labels": label}
        else:
            return {"input_ids": text, "attention_mask": mask}
        
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return metric.compute(predictions=decoded_preds, references=decoded_labels)

def preprocess_logits_for_metrics(logits, labels):
    logits = logits[0]
    logits = torch.argmax(logits, dim=-1)
    return logits

def train(text_path, label_paths):
    multiplier = len(label_paths)
    texts = list(chain.from_iterable([read_file(text_path)] * multiplier))
    labels = list(chain.from_iterable([read_file(label_path) for label_path in label_paths]))

    print(len(texts), len(labels))

    # train_chinese, validation_chinese, train_english, validation_english = train_test_split(
    #     texts, labels, test_size=0.2, shuffle=True, random_state=42
    # )

    # train_dataset = TranslationDataset(train_chinese, train_english, tokenizer)
    # validation_dataset = TranslationDataset(validation_chinese, validation_english, tokenizer)
    dataset = TranslationDataset(texts, labels, tokenizer)

    # model = MBartForConditionalGeneration.from_pretrained(checkpoint)
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
        output_dir="marian_args.pt",
        learning_rate=1e-3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="score",
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
    trainer.save_model("marian.pt") 

# def generate_translation(batch, model):
#     # Tokenize the batch and move inputs to CUDA
#     inputs = tokenizer(batch["zh"], return_tensors="pt", padding=True, truncation=True, max_length=64).input_ids.to("cuda")
#     outputs = model.generate(inputs, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    
#     # Decode each generated output
#     return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def generate_translation(batch, model):
    inputs = tokenizer(batch["zh"], return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH).input_ids.to("cuda")
    outputs = model.generate(inputs)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


def test(text_path, output_path):
    # model = MBartForConditionalGeneration.from_pretrained(checkpoint).to("cuda")
    model = AutoModelForSeq2SeqLM.from_pretrained("marian.pt").to("cuda")
    batch_size = 20
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

    with open(output_path, "w", encoding="utf-8") as predicted_file:
        for pred in predictions:
            predicted_file.write(pred + "\n")

    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-text", help="Text file path containing untranslated CHINESE text", required=True)
    parser.add_argument("-label", nargs="+", help="List of prediction file paths containing translated ENGLISH text")
    parser.add_argument("-out", help="Output file path")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    if args.label:
        train(args.text, args.label)
    else:
        test(args.text, args.out)