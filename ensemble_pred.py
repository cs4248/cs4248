import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast

from ensemble_train_utils import EnsembleModel, TrainingDataset

models = [
    AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to("cuda"),
    MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to("cuda")
]

tokenizers = [
    AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en"),
    MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
]

def predict_sentence_from_model(dataset, model, untranslated_text):
    model_input = dataset.create_model_input(untranslated_text)
    best_idx = torch.argmax(model(model_input))
    model_chosen = dataset.models[best_idx]
    model_tokenizer_chosen = dataset.model_tokenizers[best_idx]
    
    inputs = model_tokenizer_chosen(untranslated_text, return_tensors="pt").to("cuda")

    # mariamt
    if model == models[0]:
        outputs = model_chosen.generate(**inputs)
    # mbart
    else:
        outputs = model_chosen.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    decoded_outputs = model_tokenizer_chosen.decode(outputs[0], skip_special_tokens=True)
    return decoded_outputs

checkpoint = torch.load('model.pt')
model_state_dict = checkpoint['model_state_dict']

trained_model = EnsembleModel().to('cuda')
trained_model.load_state_dict(model_state_dict)


dataset = TrainingDataset("filtered_train_moe_text.txt", "filtered_train_moe_labels.txt", models, tokenizers)

with open('tatoeba.zh','r') as train_moe_labels_file, open('pred.txt', 'w') as filtered_train_moe_labels_file:
    for i, best_idx in enumerate(train_moe_labels_file):
        pred = predict_sentence_from_model(dataset, EnsembleModel(), best_idx)
        filtered_train_moe_labels_file.write(pred + '\n')
        print(i)
