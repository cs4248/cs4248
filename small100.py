import torch
from transformers import AutoTokenizer, M2M100ForConditionalGeneration, pipeline, AutoModelForSeq2SeqLM
from datasets import Dataset
import argparse

# USE THE FIRST MODEL_DIR IF YOU CREATED one using small100_finetune.py
# model_dir = "./fine_tuned_model"
model_dir = "alirezamsh/small100"

model = M2M100ForConditionalGeneration.from_pretrained(model_dir).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.tgt_lang = "en"

def translate_file(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as test_file:
        test_sentences = test_file.read().splitlines()

    data_dict = {"zh": test_sentences}
    # Convert the dictionary to a Hugging Face Dataset
    dataset = Dataset.from_dict(data_dict)

    def translate_batch(batch):
        inputs = tokenizer(batch["zh"], return_tensors="pt", padding=True, truncation=True).to("cuda")
        outputs = model.generate(**inputs)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {"translation": decoded_outputs}

    # Apply the translation in batches of size something
    translated_dataset = dataset.map(translate_batch, batched=True, batch_size=5)
    translated_outputs = translated_dataset["translation"]

    with open(output_path, "w", encoding="utf-8") as predicted_file:
        predicted_file.writelines(output_line + '\n' for output_line in translated_outputs)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', help='path to the input file', required=True)
    parser.add_argument('--out_path', help='path to the output file', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    translate_file(args.in_path, args.out_path)
