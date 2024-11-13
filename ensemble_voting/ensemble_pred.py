import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
import argparse
from ensemble_train_utils import EnsembleModel, TrainingDataset

models = [
    AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to("cuda"),
    AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("cuda")
]

tokenizers = [
    AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en"),
    AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
]

tokenizers[1].src_lang = "zh_CN"
tokenizers[1].tgt_lang = "en_XX"

def predict_sentence_from_model(dataset, model, untranslated_text):
    model_input = dataset.create_model_input(untranslated_text)
    best_idx = torch.argmax(model(model_input))
    model_chosen = dataset.models[best_idx]
    model_tokenizer_chosen = dataset.model_tokenizers[best_idx]
    
    inputs = model_tokenizer_chosen(untranslated_text, return_tensors="pt").to("cuda")

    outputs = model_chosen.generate(**inputs)
    decoded_outputs = model_tokenizer_chosen.decode(outputs[0], skip_special_tokens=True)
    return decoded_outputs

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path to the model', required=True)
    parser.add_argument('--test_text_path', help='path to the input file', required=True)
    parser.add_argument('--out_path', help='path to the input file', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    model_filename = args.model_path
    test_text_path = args.test_text_path
    out_path = args.out_path

    checkpoint = torch.load(model_filename)
    model_state_dict = checkpoint['model_state_dict']

    trained_model = EnsembleModel().to('cuda')
    trained_model.load_state_dict(model_state_dict)

    dataset = TrainingDataset(None, None, models, tokenizers)

    with open(test_text_path,'r') as train_moe_labels_file, open(out_path, 'w') as filtered_train_moe_labels_file:
        for i, best_idx in enumerate(train_moe_labels_file):
            pred = predict_sentence_from_model(dataset, trained_model, best_idx)
            filtered_train_moe_labels_file.write(pred + '\n')
            print(i)