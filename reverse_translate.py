import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader

from utils import read_file

# Load the Helsinki-NLP model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-zh"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Function to perform translation from English to Chinese
def translate_to_chinese(sentences):
    dataset = EnglishSentenceDataset(sentences)
    dataloader = DataLoader(dataset, batch_size=32)

    decoded_outputs = []
    for data in dataloader:
        tokenized, mask = data
        inputs = {"input_ids": tokenized.to(device), "attention_mask": mask.to(device)}
        outputs = model.generate(**inputs)
        decoded_outputs += tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs

def translate_file_to_chinese(file_path, output_path):
    # Read the English sentences from the file
    english_sentences = read_file(file_path)

    # Translate the English sentences to Chinese
    chinese_outputs = translate_to_chinese(english_sentences)

    # Write the translated Chinese sentences to the output file
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.writelines(sentence + '\n' for sentence in chinese_outputs)

# List of input file paths and corresponding output file paths
input_files = [
    "marianmt_predictions/pred_tatoeba.en",
    "marianmt_predictions/pred_wmttest2022.en",
    "nllb_predictions/pred_tatoeba.en",
    "nllb_predictions/pred_wmttest2022.en",
    "mbart_predictions/pred_tatoeba.en",
    "mbart_predictions/pred_wmttest2022.en"
]  
output_files = [
    "test_data/marianmt/pred_tatoeba.zh",
    "test_data/marianmt/pred_wmttest2022.zh",
    "test_data/nllb/pred_tatoeba.zh",
    "test_data/nllb/pred_wmttest2022.zh",
    "test_data/mbart/pred_tatoeba.zh",
    "test_data/mbart/pred_wmttest2022.zh"
]  # Add your output file paths here

# Translate each file in the list
for in_file, out_file in zip(input_files, output_files):
    translate_file_to_chinese(in_file, out_file)
