import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
from utils import read_file, write_file, get_device
from datasets import Dataset

MAX_INPUT_LENGTH=250
checkpoint = 'utrobinmv/t5_translate_en_ru_zh_small_1024'
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
device = get_device()
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

def generate_translation(batch, model):
    inputs = tokenizer.encode(batch["zh"], return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH).to(device)
    outputs = model.generate(inputs)
    return [tokenizer.decode(output, skip_special_tokens=True) + "\n" for output in outputs]

def translate(text_path, batch, output_path):
    test_sentences = read_file(text_path)
    # Prepare the input text with the appropriate prefix
    test_sentences = [f'translate to en: {sentence}' for sentence in test_sentences]
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
    parser = argparse.ArgumentParser(description="Translate Chinese text to English using utrobinmv/t5_translate_en_ru_zh_small_1024 model.")
    parser.add_argument("-text", help="Text file path containing untranslated CHINESE text", required=True)
    parser.add_argument("-batch", type=int, help="Batch size used during translation")
    parser.add_argument("-out", help="Output file path", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    translate(args.text, args.batch, args.out)
