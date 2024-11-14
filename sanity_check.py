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

tokenizers[1].src_lang = "zh_CN"
tokenizers[1].tgt_lang = "en_XX"

def translate_with_model(model, tokenizer, text, num_beams=5):

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, num_beams=num_beams, early_stopping=True)

    decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decoded_outputs)
    return outputs

example = "这些成果的主要研究者都是学生，研究覆盖了环境、机械、能源、医疗、生命科学、人文教育等各大领域，同学们从一个好奇的点子开始，创造出了许多具有应用价值的高端发明，其中一些项目已在国内国际获奖。"

t1, t2 = [translate_with_model(model, tokenizer, example) for model, tokenizer in zip(models, tokenizers)]
