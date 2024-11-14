import argparse
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel

from utils import get_device, read_file

device = get_device()

# Load the tokenizer and model
model_name = "uer/sbert-base-chinese-nli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# Function to compute sentence embeddings
def get_sentence_embedding(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=500
    ).to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Average pooling
    return embeddings.cpu().numpy()

# Function to calculate cosine similarity between two sets of sentences
def calculate_similarity_scores(reference_sentences, hypothesis_sentences):
    scores = []
    for ref, hyp in zip(reference_sentences, hypothesis_sentences):
        ref_embedding = get_sentence_embedding(ref)
        hyp_embedding = get_sentence_embedding(hyp)
        score = cosine_similarity(ref_embedding, hyp_embedding)[0][0]
        scores.append(score)
    return scores

# Function to write similarity scores to a file
def write_similarity_scores(scores, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        for score in scores:
            file.write(f"{score:.4f}\n")

# Main function to process multiple files
def process_files(input_path, predicted_path, output_path):
    reference_sentences = read_file(input_path)
    hypothesis_sentences = read_file(predicted_path)

    # Calculate similarity scores for each sentence pair
    similarity_scores = calculate_similarity_scores(reference_sentences, hypothesis_sentences)

    # Write the similarity scores to the specified output file
    write_similarity_scores(similarity_scores, output_path)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-text", help="Path to the original untranslated file", required=True)
    parser.add_argument("-pred", help="Path to the reverse translated file", required=True)
    parser.add_argument("-out", help="Path to the output file", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    process_files(args.text, args.pred, args.out)

