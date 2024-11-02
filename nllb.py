import argparse
from transformers import pipeline

generator = pipeline("translation",
                     model="facebook/nllb-200-distilled-600M",
                     src_lang="zho_Hans",
                     tgt_lang="eng_Latn",
                     max_length=500,
                     device="cuda")

def translate_file(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as test_file:
        test_sentences = test_file.read().splitlines()

    responses = []
    for sent in test_sentences: 
        response = generator(sent)[0]["translation_text"]
        responses.append(response + "\n")

    with open(output_path, "w", encoding="utf-8") as predicted_file:
        predicted_file.writelines(responses)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', help='path to the input file', required=True)
    parser.add_argument('--out_path', help='path to the output file', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    translate_file(args.in_path, args.out_path)
