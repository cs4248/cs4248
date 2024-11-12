import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from utils import read_file

# Load the tokenizer and model
model_name = "uer/sbert-base-chinese-nli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")

# Function to compute sentence embeddings
def get_sentence_embedding(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to("cuda")
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
def process_files(input_paths, predicted_paths, similarity_score_paths):
    for input_path, predicted_path, score_path in zip(input_paths, predicted_paths, similarity_score_paths):
        # Load the sentences from the files
        reference_sentences = read_file(input_path)
        hypothesis_sentences = read_file(predicted_path)

        # Calculate similarity scores for each sentence pair
        similarity_scores = calculate_similarity_scores(reference_sentences, hypothesis_sentences)

        # Write the similarity scores to the specified output file
        write_similarity_scores(similarity_scores, score_path)

# List of input file paths, predicted file paths, and similarity score output paths
input_file_paths = [
    "datasets/tatoeba.zh",
    "datasets/tatoeba.zh",
    "datasets/tatoeba.zh",
    "datasets/wmttest2022.zh",
    "datasets/wmttest2022.zh",
    "datasets/wmttest2022.zh"
]
predicted_file_paths = [
    "test_data/marianmt/pred_tatoeba.zh",
    "test_data/nllb/pred_tatoeba.zh",
    "test_data/mbart/pred_tatoeba.zh",
    "test_data/marianmt/pred_wmttest2022.zh",
    "test_data/nllb/pred_wmttest2022.zh",
    "test_data/mbart/pred_wmttest2022.zh"
]
similarity_score_file_paths = [
    "test_data/marianmt/tatoeba_cos",
    "test_data/nllb/tatoeba_cos",
    "test_data/mbart/tatoeba_cos",
    "test_data/marianmt/wmttest2022_cos",
    "test_data/nllb/wmttest2022_cos",
    "test_data/mbart/wmttest2022_cos"
]

# Process each set of files
process_files(input_file_paths, predicted_file_paths, similarity_score_file_paths)
