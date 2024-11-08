import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Initialize the model and tokenizer
model_name = 'utrobinmv/t5_translate_en_ru_zh_small_1024'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

device = 'cuda'
model.to(device)

def translate_file(input_path, output_path):
    # Read the input file
    with open(input_path, "r", encoding="utf-8") as file:
        sentences = file.read().splitlines()

    translations = []
    for sentence in sentences:
        # Prepare the input text with the appropriate prefix
        input_text = f'translate to en: {sentence}'
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        # Generate the translation
        output_ids = model.generate(input_ids)
        # Decode the output
        translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        translations.append(translation + "\n")

    # Write the translations to the output file
    with open(output_path, "w", encoding="utf-8") as file:
        file.writelines(translations)

def get_arguments():
    parser = argparse.ArgumentParser(description="Translate Chinese text to English using utrobinmv/t5_translate_en_ru_zh_small_1024 model.")
    parser.add_argument('--in_path', required=True, help='Path to the input file containing Chinese text.')
    parser.add_argument('--out_path', required=True, help='Path to the output file for the English translations.')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    translate_file(args.in_path, args.out_path)
