import argparse

from utils import compute_bleu_score, read_file

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pred", help="path to the prediction file", required=True)
    parser.add_argument("-lab", help="path to the label file", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    predictions = read_file(args.pred)
    labels = read_file(args.lab)
    result = compute_bleu_score(predictions, labels)
    print(result)
