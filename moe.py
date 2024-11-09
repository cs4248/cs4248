from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
import numpy as np 
import argparse
import evaluate
from moe_dataset import MoEDataset
from torch.optim import Adam

checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_pt="pt")
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file: 
        return file.read().splitlines()
    

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    accuracy_score = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    precision_score = precision.compute(predictions=predictions, references=labels, average="micro")["precision"]
    recall_score = recall.compute(predictions=predictions, references=labels, average="micro")["recall"]
    f1_score = f1.compute(predictions=predictions, references=labels, average="micro")["f1"]
    return {"precision": precision_score, "recall": recall_score, "f1": f1_score, "accuracy": accuracy_score}

def train(text_path, label_path):      
  texts = read_file(text_path)
  labels = read_file(label_path)

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5)

  train_texts, validation_texts, train_labels, validation_labels = train_test_split(
      texts, labels, test_size=0.01, shuffle=True, random_state=42
  )

  train_dataset = MoEDataset(raw_texts=train_texts, raw_labels=train_labels, tokenizer=tokenizer)
  validation_dataset = MoEDataset(raw_texts=validation_texts, raw_labels=validation_labels, tokenizer=tokenizer)
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  training_arguments = TrainingArguments(
    output_dir='classification_args.pt',
    learning_rate=3e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    fp16=True,
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
      compute_metrics=compute_metrics,
      optimizers=(Adam(model.parameters(), lr=training_arguments.learning_rate, betas=(0.9, 0.999), eps=1e-08), None)
  )

  trainer.train()
  train_metrics = trainer.evaluate(eval_dataset=train_dataset)
  print("Training Metrics:", train_metrics)
  trainer.save_model("classification.pt")

def test(text_path, output_path):
    test_sentences = read_file(text_path)
    classifier = AutoModelForSequenceClassification.from_pretrained("classification.pt", num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained("classification.pt")
    classification = pipeline("text-classification", model=classifier, tokenizer=tokenizer, device="cuda")

    # nllb = pipeline("translation",
    #                  model="facebook/nllb-200-distilled-600M",
    #                  src_lang="zho_Hans",
    #                  tgt_lang="eng_Latn",
    #                  max_length=500,
    #                  device="cuda")
    
    # marian = pipeline('translation', 
    #                  model="marian.pt",
    #                  max_new_tokens=500, 
    #                  device="cuda")
    
    # t5 = pipeline('translation', 
    #                  model="utrobinmv/t5_translate_en_ru_zh_small_1024",
    #                  max_length=500, 
    #                  device="cuda")
    
    # mbart = pipeline('translation', 
    #                  model="facebook/mbart-large-50-many-to-many-mmt",
    #                  src_lang="zho_Hans",
    #                  tgt_lang="eng_Latn",
    #                  max_length=500, 
    #                  device="cuda")
    
    # small100 = pipeline('translation', 
    #                  model="alirezamsh/small100",
    #                  max_length=500, 
    #                  device="cuda")

    nllb = read_file("./nllb_predictions/pred_wmttest2022.en")
    marian = read_file("./marianmt_predictions/predicted_wmt.en")
    mbart = read_file("./mbart_predictions/wmttest2022.AnnA_pred.en")
    t5 = read_file("./t5_predictions/pred_wmttest2022.en")
    small100 = read_file("./small100_predictions/pred_wmttest2022.en")

    # nllb = read_file("./nllb_predictions/pred_tatoeba.en")
    # marian = read_file("./marianmt_predictions/predicted_tatoeba.en")
    # mbart = read_file("./mbart_predictions/tatoeba_pred.en")
    # t5 = read_file("./t5_predictions/pred_tatoeba.en")
    # small100 = read_file("./small100_predictions/pred_tatoeba.en")
    
    models = {
       "LABEL_0": marian,
       "LABEL_1": mbart,
       "LABEL_2": t5,
       "LABEL_3": small100,
       "LABEL_4": nllb
    }
        
    responses = []
    for i, sent in enumerate(test_sentences): 
      label = classification(sent)[0]["label"]
      # response = models[label](sent)[0]["translation_text"]
      response = models[label][i]
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

