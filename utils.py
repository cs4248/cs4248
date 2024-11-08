import evaluate

sacrebleu = evaluate.load("sacrebleu")

def compute_single_bleu_score(pred, lab):
    return sacrebleu.compute(predictions=[pred], references=[[lab]])["score"]

def compute_bleu_score(pred, lab):
    return sacrebleu.compute(predictions=pred, references=lab)["score"]

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file: 
        return file.read().splitlines()
