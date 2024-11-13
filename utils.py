from evaluate import load
import torch

sacrebleu = load("sacrebleu")
comet = load("comet")

'''
Evaluation utils
'''
def compute_single_bleu_score(pred, lab):
    return sacrebleu.compute(predictions=[pred], references=[[lab]])["score"]

def compute_bleu_score(pred, lab):
    return sacrebleu.compute(predictions=pred, references=lab)["score"]

def compute_comet_score(src, pred, lab):
    return comet.compute(predictions=pred, references=lab, sources=src)["mean_score"]

'''
File utils
'''
def write_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(content)
        file.close()

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file: 
        return file.read().splitlines()

'''
Other utils
'''
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

