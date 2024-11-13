import evaluate
import torch

sacrebleu = evaluate.load("sacrebleu")

def compute_single_bleu_score(pred, lab):
    return sacrebleu.compute(predictions=[pred], references=[[lab]])["score"]

def compute_bleu_score(pred, lab):
    return sacrebleu.compute(predictions=pred, references=lab)["score"]


def write_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(content)
        file.close()

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file: 
        return file.read().splitlines()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

