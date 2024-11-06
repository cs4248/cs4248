from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import Dataset
import torch

# print(torch.cuda.is_available())  # Should return True if CUDA is set up correctly
# print(torch.version.cuda)         # Displays the CUDA version PyTorch is using
# print(torch.cuda.current_device())# Shows the current device index
# print(torch.cuda.get_device_name(0))  # Returns the GPU model name

# Load mBART model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to("cuda")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "zh_CN"

# Generate translations for a batch
def generate_translation(batch):
    # Tokenize the batch and move inputs to CUDA
    inputs = tokenizer(batch["zh"], return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
    outputs = model.generate(inputs, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    
    # Decode each generated output
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Paths to the English and Chinese files
en_path = "train.zh-en.en"
zh_path = "train.zh-en.zh"

# Read the English and Chinese sentences
with open(en_path, "r", encoding="utf-8") as en_file:
    en_sentences = [line.strip() for line in en_file]

with open(zh_path, "r", encoding="utf-8") as zh_file:
    zh_sentences = [line.strip() for line in zh_file]

# Ensure both files have the same number of lines
assert len(en_sentences) == len(zh_sentences), "Mismatched sentence counts in .en and .zh files"

# Create a dictionary with English and Chinese parallel texts
data_dict = {"en": en_sentences, "zh": zh_sentences}

# Convert the dictionary to a Hugging Face Dataset
trans_dataset = Dataset.from_dict(data_dict)

# Process the dataset in batches
batch_size = 1  # Adjust based on available memory
predictions = []
references = []

for i in range(0, len(trans_dataset), batch_size):
    if i % 1000 == 0:
        print(i)
        
    batch = trans_dataset[i: i + batch_size]
    pred = generate_translation(batch)
    predictions.extend(pred)
    references.extend([ref for ref in batch["en"]])  # Wrap in a list for sacrebleu
    
    # print(predictions)


with open("pred_train.zh-en.en", "w", encoding="utf-8") as f:
    for pred in predictions:
        f.write(pred + "\n")
        