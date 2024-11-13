import argparse

from utils import read_file

def predict(pred_paths, cos_sim_paths, out_path):
    preds = [read_file(pred_path) for pred_path in pred_paths]
    cos_sims = [read_file(cos_sim_path) for cos_sim_path in cos_sim_paths]
    ensemble_predictions = []
    for i in range(len(preds[0])):
        best_score, best_pred = -1, ""
        for j in range(len(preds)):
            if best_score < float(cos_sims[j][i]):
                best_score = float(cos_sims[j][i])
                best_pred = preds[j][i]
        ensemble_predictions.append(best_pred)

    with open(out_path, "w", encoding="utf-8") as file:
        file.writelines(pred + "\n" for pred in ensemble_predictions)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pred", nargs="+", help="Path to the English prediction files", required=True)
    parser.add_argument("-sim", nargs="+", help="Path to the similarity score files", required=True)
    parser.add_argument("-out", help="Path to the output file", required=True)
    args = parser.parse_args()
    if len(args.pred) != len(args.sim):
        raise Exception("No. of pred files must match no. of sim files")
    return args

if __name__ == "__main__":
    args = get_arguments()
    predict(args.pred, args.sim, args.out)

