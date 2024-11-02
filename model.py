from transformers import pipeline, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from torch.utils.data import Dataset
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import evaluate
import torch
import numpy as np 

MAX_INPUT_LENGTH = 64
checkpoint = "Helsinki-NLP/opus-mt-zh-en"

def read_file(file_path) :
    with open(file_path, "r", encoding="utf-8") as file: 
        return file.read().splitlines()

chinese_sentences = read_file("train.zh-en.zh")
english_sentences = read_file("train.zh-en.en")

train_chinese, val_chinese, train_english, val_english = train_test_split(
    chinese_sentences, english_sentences, test_size=0.2, shuffle=True, random_state=42
)

metric = evaluate.load("sacrebleu")

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
    

class TranslationDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
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

tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_tensors="pt")
dataset = TranslationDataset(chinese_sentences, english_sentences, tokenizer, MAX_INPUT_LENGTH)
train_dataset = TranslationDataset(train_chinese, train_english, tokenizer, MAX_INPUT_LENGTH)
validation_dataset = TranslationDataset(val_chinese, val_english, tokenizer, MAX_INPUT_LENGTH)

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
    output_dir='model.pt',
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    fp16=True,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="score",
)
    # output_dir='model.pt',
    # learning_rate=1e-4,
    # per_device_train_batch_size=32,
    # num_train_epochs=3,
    # fp16=True,
    # optim="adafactor",
    # seed=42,

trainer = Trainer(
    model=model,
    args=training_arguments,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model, padding=True),
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# trainer.train()
# trainer.save_model() 

base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained("model.pt")
generator = pipeline('translation', 
                     model=model, 
                     tokenizer=tokenizer,
                     max_new_tokens=128, 
                     device="cuda")
base_generator = pipeline('translation', 
                     model=base_model, 
                     tokenizer=tokenizer,
                     max_new_tokens=128, 
                     device="cuda")

# with open("tatoeba.zh", "r", encoding="utf-8") as test_file:
#     test_sentences = test_file.read().splitlines()
#     # test_dataset = TranslationDataset(test_sentences, labels=None, tokenizer=tokenizer, max_length=MAX_INPUT_LENGTH)
    
# responses = []
# for sent in test_sentences: 
#     response = generator(sent)[0]["translation_text"]
#     # response = base_generator(sent)[0]["translation_text"]
#     responses.append(response + "\n")

# with open("predicted_tatoeba.en", "w", encoding="utf-8") as predicted_file:
# # with open("base_predicted.en", "w", encoding="utf-8") as predicted_file:
#     predicted_file.writelines(responses)

with open("wmttest2022.zh", "r", encoding="utf-8") as test_file:
    test_sentences = test_file.read().splitlines()
    # test_dataset = TranslationDataset(test_sentences, labels=None, tokenizer=tokenizer, max_length=MAX_INPUT_LENGTH)
    
responses = []
for sent in test_sentences: 
    # response = generator(sent)[0]["translation_text"]
    response = base_generator(sent)[0]["translation_text"]
    responses.append(response + "\n")

# with open("predicted_wmt.en", "w", encoding="utf-8") as predicted_file:
with open("base_predicted_wmt.en", "w", encoding="utf-8") as predicted_file:
    predicted_file.writelines(responses)
