from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
import numpy as np 
import argparse
import evaluate
from moe_dataset import MoEDataset

checkpoint = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_pt="pt")
accuracy = evaluate.load("accuracy")

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file: 
        return file.read().splitlines()
    

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def train(text_path, label_path):      
  texts = read_file(text_path)
  labels = read_file(label_path)

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  peft_config = LoraConfig(
      r=16,
      lora_alpha=32,
      lora_dropout=0.05,
      bias="none",
      task_type=TaskType.SEQ_CLS,
      target_modules=["k_lin", "v_lin", "q_lin"]
  )

  model = get_peft_model(model, peft_config)
  print(model.print_trainable_parameters())

  train_texts, validation_texts, train_labels, validation_labels = train_test_split(
      texts, labels, test_size=0.2, shuffle=True, random_state=42
  )

  train_dataset = MoEDataset(raw_texts=train_texts, raw_labels=train_labels, tokenizer=tokenizer)
  validation_dataset = MoEDataset(raw_texts=validation_texts, raw_labels=validation_labels, tokenizer=tokenizer)
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  training_arguments = TrainingArguments(
    output_dir='classification.pt',
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    fp16=True,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
  )

  trainer = Trainer(
      model=model,
      args=training_arguments,
      processing_class=tokenizer,
      train_dataset=train_dataset,
      eval_dataset=validation_dataset,
      data_collator=data_collator,
      compute_metrics=compute_metrics
  )

  trainer.train()
  trainer.save_model()

def test(text_path, output_path):
    test_sentences = read_file(text_path)
    classifier = AutoModelForSequenceClassification.from_pretrained("classification.pt", num_labels=2)
    classification = pipeline("text-classification", model=classifier, tokenizer=tokenizer, device="cuda")

    nllb = pipeline("translation",
                     model="facebook/nllb-200-distilled-600M",
                     src_lang="zho_Hans",
                     tgt_lang="eng_Latn",
                     max_length=500,
                     device="cuda")
    
    marian = pipeline('translation', 
                     model="marian.pt",
                     max_new_tokens=128, 
                     device="cuda")
    
    responses = []
    for sent in test_sentences: 
      label = classification(sent)[0]["label"]
      if label == "LABEL_0":
        response = nllb(sent)[0]["translation_text"]
      else:
        response = marian(sent)[0]["translation_text"]
      responses.append(response + "\n")
    
    with open(output_path, "w", encoding="utf-8") as predicted_file:
      predicted_file.writelines(responses)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-text", help="Text file path containing untranslated CHINESE text", required=True)
    parser.add_argument("-label", help="Label file path containing model to use")
    parser.add_argument("-out", help="Output file path")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    if args.label: 
       train(args.text, args.label)
    else: 
        test(args.text, args.out)

