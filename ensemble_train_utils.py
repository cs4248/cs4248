import torch
from utils import read_file
from torch.utils.data import Dataset as Ds

class EnsembleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = torch.nn.Linear(193113, 1024).to('cuda')
        self.l2 = torch.nn.LeakyReLU(0.1)
        self.l3 = torch.nn.Dropout(0.2)
        self.l4 = torch.nn.Linear(1024, 128).to('cuda')
        self.l5 = torch.nn.LeakyReLU(0.1)
        self.l6 = torch.nn.Dropout(0.2)
        self.l7 = torch.nn.Linear(128, 2).to('cuda')

    def forward(self, concatted_outputs):
        # print(len(concatted_outputs))
        x = self.l1(concatted_outputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return x

class TrainingDataset(Ds):
    def __init__(self, text_path, lab_path, models, model_tokenizers):
        '''
        dataset_reduce_scale = reduce the sample size of the dataset. 
        E.g dataset_reduce_scale=5 on sample size 100, basically reduce sample size from 100 to 20.
        '''
        self.untranslated_texts = read_file(text_path)
        self.best_model_idx_labels = read_file(lab_path)

        self.model_tokenizers = model_tokenizers
        self.models = models

        start_token_ids = [model.config.decoder_start_token_id for model in self.models]
        self.decoder_input_ids_list = [torch.tensor([[start_token_id]]).to("cuda") for start_token_id in start_token_ids]

    def __len__(self):
        return len(self.untranslated_texts)
    
    def __getitem__(self, idx):
        untranslated_text = self.untranslated_texts[idx]
        concatted_outputs = self.create_model_input(untranslated_text)
    
        best_model_idx = torch.tensor(int(self.best_model_idx_labels[idx]))
        
        return concatted_outputs, best_model_idx
    
    def create_model_input(self, untranslated_text):
        with torch.no_grad():
            tokenized_texts = [tokenizer(untranslated_text, return_tensors="pt").to("cuda") for tokenizer in self.model_tokenizers]
            output_logits = [model(**tokenized_text, decoder_input_ids=decoder_input_ids).logits for model, tokenized_text, decoder_input_ids in zip(self.models, tokenized_texts, self.decoder_input_ids_list)]
            concatted_outputs = torch.cat(output_logits, dim=-1)
            concatted_outputs = concatted_outputs.squeeze()
        return concatted_outputs