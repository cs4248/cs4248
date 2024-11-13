import argparse
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from utils import get_device, read_file, write_file

MAX_LENGTH = 500
device = get_device()

def test(file_path, output_path, batch_size, model, tokenizer):
    dataset = Dataset.from_dict({"zh": read_file(file_path)})
    dataloader = DataLoader(dataset, batch_size=batch_size)
    tokenize = get_tokenize_function(tokenizer)

    responses = list()
    for batch in dataloader:
        outputs = model.generate(tokenize(batch).to(device))
        responses += [tokenizer.decode(output, skip_special_tokens=True) + "\n" for output in outputs]

    write_file(output_path, responses)

def get_model(model_path):
    if model_path:
        return AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    return pipeline(
        "translation",
        model="facebook/mbart-large-50-many-to-many-mmt",
        src_lang="zh_CN",
        tgt_lang="en_XX",
        max_length=MAX_LENGTH,
        device=device
    )

def get_tokenizer(model_path):
    if model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, return_tensors="pt")
    else:
        tokenizer =AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", return_tensors="pt")
    tokenizer.src_lang = "zh_CN"
    tokenizer.tgt_lang = "en_XX"
    return tokenizer

def get_tokenize_function(tokenizer):
    def tokenize(texts):
        model_inputs = tokenizer(
            texts["zh"],
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
    parser.add_argument("-model", help="Path to model")
    parser.add_argument("-batch", help="Batch size for prediction", default=32)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    test(
        args.input,
        args.out,
        args.batch,
        get_model(args.model),
        get_tokenizer(args.model)
    )

