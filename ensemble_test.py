from transformers import AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration, AutoTokenizer
from datasets import Dataset
import torch
from utils import read_file
from torch.utils.data import Dataset as Ds, Subset
import datetime
from torch.utils.data import DataLoader

# since this file only test 2 models together, and train_moe_labels.txt contains numbers from
# all 5 models, we gotta force it all into 2 numbers only. (Or else cuda will crash xpp)

with open('train.zh-en.zh', 'r') as train_moe_text_file, \
    open('train_moe_labels.txt','r') as train_moe_labels_file, \
    open('filtered_train_moe_text.txt', 'w') as filtered_train_moe_text_file, \
    open('filtered_train_moe_labels.txt', 'w') as filtered_train_moe_labels_file:
    
    for text_line, best_idx in zip(train_moe_text_file, train_moe_labels_file):
        best_idx = int(best_idx)
        if not (best_idx == 0 or best_idx == 3):
            continue

        filtered_train_moe_text_file.write(text_line)
        # small_100
        if best_idx == 3:
            filtered_train_moe_labels_file.write(str(1) + '\n')
        # mariant
        else:
            filtered_train_moe_labels_file.write(str(0) + '\n')

models = [
    AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to("cuda"),
    M2M100ForConditionalGeneration.from_pretrained("alirezamsh/small100").to("cuda")
]

tokenizers = [
    AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en"),
    AutoTokenizer.from_pretrained("alirezamsh/small100"),
]

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
            output_logits = [model(**tokenized_text, decoder_input_ids=decoder_input_ids).logits for model, tokenized_text, decoder_input_ids in zip(models, tokenized_texts, self.decoder_input_ids_list)]
            concatted_outputs = torch.cat(output_logits, dim=-1)
            concatted_outputs = concatted_outputs.squeeze()
        return concatted_outputs

def train(model, dataset, batch_size, learning_rate, num_epoch, model_path=None):
    """
    Complete the training procedure below by specifying the loss function
    and optimizers with the specified learning rate and specified number of epoch.

    """
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start = datetime.datetime.now()
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            untranslated_text = data[0].to('cuda')
            best_model_idx = data[1].to('cuda')

            # zero the parameter gradients
            model.zero_grad()

            # do forward propagation
            probs = model(untranslated_text)

            # calculate the loss
            loss = criterion(probs, best_model_idx)


            # do backward propagation
            loss.backward()

            # do the parameter optimization
            optimizer.step()

            # calculate running loss value for non padding
            running_loss += loss.item()

            # print loss value every 100 iterations and reset running loss
            if step % 100 == 99:
                print('[%d, %5d] loss: %.10f' %
                    (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0

    end = datetime.datetime.now()
    
    # define the checkpoint and save it to the model path
    # tip: the checkpoint can contain more than just the model
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, model_path)

    print('Model saved in ', model_path)
    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))

import numpy as np

# Init training data
subset_size = 1000
dataset = TrainingDataset("filtered_train_moe_text.txt", "filtered_train_moe_labels.txt", models, tokenizers)
indices = list(range(subset_size))  # Define a list of indices
subset = Subset(dataset, indices)

train(EnsembleModel().to('cuda'), subset, 2, 0.001, 3, 'model.pt')

