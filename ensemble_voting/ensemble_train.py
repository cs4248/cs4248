import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import Dataset
import torch
from ensemble_train_utils import EnsembleModel, TrainingDataset
from torch.utils.data import Subset
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
        if not (best_idx == 0 or best_idx == 4):
            continue

        filtered_train_moe_text_file.write(text_line)
        # nllb
        if best_idx == 4:
            filtered_train_moe_labels_file.write(str(1) + '\n')
        # mariant
        else:
            filtered_train_moe_labels_file.write(str(0) + '\n')

models = [
    AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to("cuda"),
    AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("cuda")
]

tokenizers = [
    AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en"),
    AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
]

tokenizers[1].src_lang = "zh_CN"
tokenizers[1].tgt_lang = "en_XX"

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

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path to save the model at', required=True)
    return parser.parse_args()
    
if __name__ == '__main__':
    # Init training data
    subset_size = 6000
    dataset = TrainingDataset("filtered_train_moe_text.txt", "filtered_train_moe_labels.txt", models, tokenizers)
    indices = list(range(subset_size))  # Define a list of indices
    subset = Subset(dataset, indices)

    args = get_arguments()
    model_path = args.model_path

    train(EnsembleModel().to('cuda'), subset, 2, 0.001, 3, model_path)

