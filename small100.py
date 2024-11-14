from transformers import AutoTokenizer, M2M100ForConditionalGeneration
from datasets import Dataset
from utils import read_file, write_file, get_device
import argparse

MAX_INPUT_LENGTH = 250
checkpoint = "alirezamsh/small100"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.tgt_lang = "en"
device = get_device()
model = M2M100ForConditionalGeneration.from_pretrained(checkpoint).to(device)

def generate_translation(batch, model):
    inputs = tokenizer(batch["zh"], return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH).input_ids.to(device)
    outputs = model.generate(inputs)
    return [tokenizer.decode(output, skip_special_tokens=True) + "\n" for output in outputs]

def translate(text_path, batch, output_path):
    test_sentences = read_file(text_path)
    data_dict = {"zh": test_sentences}

    trans_dataset = Dataset.from_dict(data_dict)
    batch_size = batch or 20
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
    parser.add_argument("-batch", type=int, help="Batch size used during translation")
    parser.add_argument("-out", help="Output file path", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    translate(args.text, args.batch, args.out)
