import argparse

from utils import compute_bleu_score, compute_single_bleu_score, read_file

def compute_best_bleu_score(lab_path, pred_paths):
    labels = read_file(lab_path)
    predictions = [read_file(pred_path) for pred_path in pred_paths]
    
    best_predictions = []
    for i in range(len(labels)):
        best_score, best_pred = 0, ""
        for pred in predictions:
            score = compute_single_bleu_score(pred[i], labels[i])
            if score > best_score:
                best_score = score
                best_pred = pred[i]
        best_predictions.append(best_pred)

    return compute_bleu_score(best_predictions, labels)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lab", help="Label file path containing ideal translated ENGLISH text output", required=True)
    parser.add_argument("-pred", nargs="+", help="List of prediction file paths containing translated ENGLISH text", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    print(compute_best_bleu_score(args.lab, args.pred))
