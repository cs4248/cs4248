import argparse
import evaluate

metric = evaluate.load("sacrebleu")

def compute_metrics(predictions, labels):
    return metric.compute(predictions=predictions, references=labels)

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().splitlines()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', help='path to the prediction file', required=True)
    parser.add_argument('--lab', help='path to the label file', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    predictions = read_file(args.pred)
    labels = read_file(args.lab)
    result = compute_metrics(predictions, labels)
    print(result)
