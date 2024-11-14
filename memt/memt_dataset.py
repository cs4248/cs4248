import argparse
from torch.utils.data import Dataset

from utils import compute_single_bleu_score, read_file

class MEMTDataset(Dataset):
    '''
    text_path: str
        Path to the text file used to generate MEMT label file
        File should contain untranslated CHINESE text
    lab_path: str
        Path to the MEMT label file
        File should contain numbers between 1 and N, where N is
        the number of datasets combined to form the MEMT dataset
    '''
    def __init__(self, text_path, lab_path, tokenizer):
        self.__init__(read_file(text_path), read_file(lab_path), tokenizer)

    '''
    text_paths: str
        Path to the text file used to generate prediction files
        File should contain untranslated CHINESE text
    pred_paths: list of str
        List of paths to the prediction files used to generate MEMT dataset
        Files should contain translated ENGLISH text
    lab_path: str
        List of paths to the label files used to generate MEMT dataset
        File should contain translated ENGLISH text
    '''
    def __init__(self, text_path, pred_paths, lab_path, tokenizer):
        raw_texts, raw_labels = get_raw_moe_dataset(text_path, lab_path, pred_paths)
        self.__init__(raw_texts, raw_labels, tokenizer)

    def __init__(self, raw_texts, raw_labels, tokenizer, max_length=128):
        model_inputs = tokenizer(
            text=raw_texts,
            max_length=max_length, 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        )
        self.texts = model_inputs["input_ids"]
        self.masks = model_inputs["attention_mask"]
        self.labels = [int(label) for label in raw_labels]

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        mask = self.masks[idx]
        lab = None
        if self.labels:
            lab = self.labels[idx]
        return {"input_ids": text, "attention_mask": mask, "labels": lab}

'''
lab_paths: list of str
    List of paths to the label files used to generate MoE dataset
    Files should contain translated ENGLISH text
'''
def get_raw_memt_dataset(text_paths, lab_path, pred_paths):
    raw_texts = read_file(text_paths)
    raw_preds = [read_file(pred_path) for pred_path in pred_paths]
    raw_labels = read_file(lab_path)
    bleu_scores = [[compute_single_bleu_score(pair[0], pair[1]) for pair in zip(preds, raw_labels)] for preds in raw_preds]
    moe_labels = [scores.index(max(scores)) for scores in zip(*bleu_scores)]
    return raw_texts, moe_labels

def write_label_file(file_path, labels):
    with open(file_path, "w", encoding="utf-8") as file:
        for label in labels:
            file.write(str(label) + "\n")

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-text", help="Text file path containing untranslated CHINESE text", required=True)
    parser.add_argument("-lab", help="Label file path containing ideal translated ENGLISH text output", required=True)
    parser.add_argument("-pred", nargs="+", help="List of prediction file paths containing translated ENGLISH text", required=True)
    parser.add_argument("-out", help="Output file path", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    _, labels = get_raw_memt_dataset(args.text, args.lab, args.pred)
    write_label_file(args.out, labels)
