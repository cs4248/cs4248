import torch
from ensemble_test import EnsembleModel, TrainingDataset
from transformers import AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration, AutoTokenizer

models = [
    AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to("cuda"),
    M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100").to("cuda")
]

tokenizers = [
    AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en"),
    AutoTokenizer.from_pretrained("alirezamsh/small100"),
]

def predict_sentence_from_model(dataset, model, untranslated_text):
    model_input = dataset.create_model_input(untranslated_text)
    best_idx = torch.argmax(model(model_input))
    model_chosen = dataset.models[best_idx]
    model_tokenizer_chosen = dataset.model_tokenizers[best_idx]
    
    inputs = model_tokenizer_chosen(untranslated_text, return_tensors="pt").to("cuda")
    outputs = model_chosen.generate(**inputs)
    decoded_outputs = model_tokenizer_chosen.decode(outputs[0], skip_special_tokens=True)
    return decoded_outputs

checkpoint = torch.load('model.pt')
model_state_dict = checkpoint['model_state_dict']

trained_model = EnsembleModel().to('cuda')
trained_model.load_state_dict(model_state_dict)


dataset = TrainingDataset("filtered_train_moe_text.txt", "filtered_train_moe_labels.txt", models, tokenizers)

with open('wmttest2022.zh','r') as train_moe_labels_file, open('pred.txt', 'w') as filtered_train_moe_labels_file:
    for i, best_idx in enumerate(train_moe_labels_file):
        pred = predict_sentence_from_model(dataset, EnsembleModel(), best_idx)
        filtered_train_moe_labels_file.write(pred + '\n')
        print(i)