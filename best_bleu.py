'''
This script computes the BLEU score using the best prediction of the n models.
The best prediction is the one that has the highest BLEU score.

Results from this script are only used to determine the theoretical best BLEU
score that can be achieved by a selection based model that uses the predictions
of the n models.
'''

import argparse
from itertools import combinations
from math import inf

from utils import compute_bleu_score, compute_single_bleu_score, read_file

def get_pred_score_pairings(labels, predictions):
    return [[(pred[i], compute_single_bleu_score(pred[i], labels[i]))
            for i in range(len(labels))]
                for pred in predictions]

def get_best_predictions(pred_score_pair):
    best_predictions = []
    for i in range(len(pred_score_pair[0])):
        best_score, best_pred = -inf, ""
        for j in range(len(pred_score_pair)):
            if pred_score_pair[j][i][1] > best_score:
                best_score = pred_score_pair[j][i][1]
                best_pred = pred_score_pair[j][i][0]
        best_predictions.append(best_pred)
    return best_predictions

def compute_best_bleu_score(lab_path, pred_paths):
    labels = read_file(lab_path)
    predictions = [read_file(pred_path) for pred_path in pred_paths]
    pred_score_pairs = get_pred_score_pairings(labels, predictions)
    best_predictions = get_best_predictions(pred_score_pairs)

    return compute_bleu_score(best_predictions, labels)

def compute_all_permutations_bleu_score(lab_path, pred_paths):
    labels = read_file(lab_path)
    predictions = [read_file(pred_path) for pred_path in pred_paths]
    pred_score_pairs = get_pred_score_pairings(labels, predictions)

    combination_set = [i for i in range(len(pred_paths))]

    perms = list()
    for i in range(len(pred_paths)):
        perms += list(combinations(combination_set, i+1))
    
    result = dict()
    for perm in perms:
        perm_str = "".join([str(i+1) for i in perm])
        best_predictions = get_best_predictions([pred_score_pairs[i] for i in perm])
        result[perm_str] = compute_bleu_score(best_predictions, labels)

    return result

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-perm", help="Flag for whether to generate all permutations", action="store_true")
    parser.add_argument("-label", help="Label file path containing ideal translated ENGLISH text output", required=True)
    parser.add_argument("-pred", nargs="+", help="List of prediction file paths containing translated ENGLISH text", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    if args.perm:
        print(compute_all_permutations_bleu_score(args.label, args.pred))
    else:
        print(compute_best_bleu_score(args.label, args.pred))
