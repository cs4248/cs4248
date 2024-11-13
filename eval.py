import argparse

from utils import compute_bleu_score, compute_comet_score, read_file

def check_valid_metric(metric):
    allowed = {"comet", "bleu"}
    if metric.lower() not in allowed:
        raise argparse.ArgumentTypeError(f"Invalid choice: {metric}. Choose from 'BLEU' or 'COMET'")
    return metric.lower()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-metric", help="COMET or BLEU, case insensitive", type=check_valid_metric, required=True)
    parser.add_argument("-pred", help="Path to the translated file", required=True)
    parser.add_argument("-lab", help="Path to the label file", required=True)
    parser.add_argument("-src", help="Path to the untranslated file")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    predictions = read_file(args.pred)
    labels = read_file(args.lab)
    if args.metric == "comet":
        if not args.src:
            raise Exception("COMET requires -src")
        source = read_file(args.src)
        result = compute_comet_score(source, predictions, labels)
    elif args.metric == "bleu":
        result = compute_bleu_score(predictions, labels)
    print(result)

