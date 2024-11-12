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

pred_file_paths = [
    [
        "marianmt_predictions/pred_tatoeba.en",
        "nllb_predictions/pred_tatoeba.en",
        "mbart_predictions/pred_tatoeba.en",
    ],
    [
        "marianmt_predictions/pred_wmttest2022.en",
        "nllb_predictions/pred_wmttest2022.en",
        "mbart_predictions/pred_wmttest2022.en",
    ]
]
cos_sim_file_paths = [
    [
        "cos_sim/marianmt/tatoeba_cos",
        "cos_sim/nllb/tatoeba_cos",
        "cos_sim/mbart/tatoeba_cos"
    ],
    [
        "cos_sim/marianmt/wmttest2022_cos",
        "cos_sim/nllb/wmttest2022_cos",
        "cos_sim/mbart/wmttest2022_cos"
    ]
]
out_paths = [
    "cos_sim/pred_tatoeba.en",
    "cos_sim/pred_wmttest2022.en"
]

for pred_file_path, cos_sim_file_path, out_path in zip(pred_file_paths, cos_sim_file_paths, out_paths):
    predict(pred_file_path, cos_sim_file_path, out_path)

