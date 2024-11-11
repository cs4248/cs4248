import torch
from transformers import AutoTokenizer, M2M100ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import argparse

FINE_TUNE_DATASET_SIZE = 100
NUM_TRAIN_EPOCHS = 3
train_text_path = 'train.zh-en.zh'
train_label_path = 'train.zh-en.en'


model = M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("alirezamsh/small100")
tokenizer.tgt_lang = "en"

with open(train_text_path, "r", encoding="utf-8") as train_file, \
    open(train_label_path, "r", encoding="utf-8") as test_file:
    train_sentences = train_file.read().splitlines()
    test_sentences = test_file.read().splitlines()

data_dict = {"zh": train_sentences, "en":test_sentences}
# Convert the dictionary to a Hugging Face Dataset
dataset = Dataset.from_dict(data_dict)

def tokenize_function(sentences):
    tokenized_zh = tokenizer(sentences["zh"], padding="max_length", truncation=True)
    tokenized_en = tokenizer(sentences["en"], padding="max_length", truncation=True)
    
    # Combine the tokenized fields into a dictionary
    return {
        "input_ids": tokenized_zh["input_ids"],
        "attention_mask": tokenized_zh["attention_mask"],
        "labels": tokenized_en["input_ids"],
    }

tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=5)
tokenized_dataset = tokenized_dataset.shuffle(seed=42).select(range(FINE_TUNE_DATASET_SIZE))

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Train and evaluate
trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
