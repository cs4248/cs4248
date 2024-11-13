import argparse
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils import get_device, read_file, write_file

device = get_device()
MAX_LENGTH = 500

'''
class EnglishSentenceDataset(Dataset):
    def __init__(self, sentences):
        tokens = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        self.input = tokens["input_ids"]
        self.masks = tokens["attention_mask"]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.masks[idx]
'''

# Function to perform translation from English to Chinese
def translate_to_chinese(model, tokenizer, sentences, batch_size):
    # dataset = EnglishSentenceDataset(sentences)
    dataset = Dataset.from_dict({"en": sentences})
    dataloader = DataLoader(dataset, batch_size=batch_size)
    tokenize = get_tokenize_function(tokenizer)

    decoded_outputs = []
    for batch in dataloader:
        inputs = tokenize(batch).to(device)
        # tokenized, mask = data
        # inputs = {"input_ids": tokenized.to(device), "attention_mask": mask.to(device)}
        outputs = model.generate(inputs)
        decoded_outputs += tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [output + "\n" for output in decoded_outputs]

def translate_file_to_chinese(input_path, output_path, batch_size):
    model = get_model()
    tokenizer = get_tokenizer()

    english_sentences = read_file(input_path)
    chinese_outputs = translate_to_chinese(
        model,
        tokenizer,
        english_sentences,
        batch_size
    )
    write_file(output_path, chinese_outputs)

def get_model():
    return AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh").to(device)

def get_tokenizer():
    return AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh", return_tensors="pt")

def get_tokenize_function(tokenizer):
    def tokenize(texts):
        model_inputs = tokenizer(
            texts["en"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        return model_inputs.input_ids
    return tokenize

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", help="Path to the input file", required=True)
    parser.add_argument("-out", help="Path to the output file", required=True)
    parser.add_argument("-batch", help="Batch size for prediction", type=int, default=32)
    return parser.parse_args()

# Translate each file in the list
if __name__ == "__main__":
    args = get_arguments()
    translate_file_to_chinese(args.input, args.out, args.batch)

