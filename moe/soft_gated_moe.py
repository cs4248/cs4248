import argparse
import datetime
import numpy as np 
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from utils import get_device, read_file

device = get_device()

# Classifier transformer
base_model_checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(base_model_checkpoint, return_pt="pt")
base_model = AutoModel.from_pretrained(base_model_checkpoint).to(device)
base_model.eval()

# Common tokenizer for the expert model outputs
output_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

# Hyper parameters
MAX_LENGTH = 500
NUM_LABELS = 4
NUM_EPOCHS = 10
LR = 1e-3

for param in base_model.parameters():
    param.requires_grad = False

# Classifier head for classifier transformer
class SimpleMoEClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(768, 256)
        self.fc2 = torch.nn.Linear(256, NUM_LABELS)
        self.lrelu = torch.nn.LeakyReLU(0.1)
        self.dropout = torch.nn.Dropout(0.2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, X):
        X = self.fc1(X)
        X = self.lrelu(X)
        X = self.dropout(X)
        X = self.fc2(X)
        X = self.softmax(X)
        return X

    def predict(self, X):
        with torch.no_grad():
            Y = self.forward(X)
            return torch.argmax(Y, dim=1)

class SimpleMoEDataset(Dataset):
    def __init__(self, text_path, pred_paths=None, label_path=None):
        self.X1 = preprocess_X(read_file(text_path))
        self.Y = None
        self.X2 = None

        if label_path:
            tokens = output_tokenizer(
                text=read_file(label_path),
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            self.Y = tokens["input_ids"]

        if pred_paths:
            preds = [read_file(pred_path) for pred_path in pred_paths]
            self.X2 = []
            for pred in preds:
                tokens = output_tokenizer(
                    text=pred,
                    max_length=MAX_LENGTH,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                shape = tokens["input_ids"].shape
                self.X2.append(tokens["input_ids"])
            self.X2 = torch.cat(self.X2).view((shape[0], -1, shape[1]))

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        if self.Y == None:
            if self.X2 == None:
                return self.X1[idx]
            return self.X1[idx], self.X2[idx]
        if self.X2 == None:
            return self.X1[idx], self.Y[idx]
        return self.X1[idx], self.X2[idx], self.Y[idx]

class TokenizedDataset(Dataset):
    def __init__(self, texts):
        tokens = tokenizer(
            text=texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        self.texts = tokens["input_ids"]
        self.mask = tokens["attention_mask"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"input_ids": self.texts[idx], "attention_mask": self.mask[idx]}

def preprocess_X(texts):
    dataset = TokenizedDataset(texts)
    loader = DataLoader(dataset, batch_size=128)

    X_list = []
    with torch.no_grad():
        for data in loader:
            input_ids = data["input_ids"].to(device)
            masks = data["attention_mask"].to(device)
            X = {
                "input_ids": input_ids,
                "attention_mask": masks,
            }
            X_list.append(base_model(**X).pooler_output)
    result = torch.cat(X_list, dim=0)
    return result

def train(text_path, label_path, pred_paths, model_path, batch_size):
    train_dataset = SimpleMoEDataset(text_path, pred_paths, label_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    model = SimpleMoEClassifier()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start = datetime.datetime.now()
    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        for step, data in enumerate(train_loader, 0):
            X1, X2, Y = data
            X1 = X1.to(device)
            X2 = X2.to(device)
            Y = Y.to(device)

            Y_pred = torch.sum(model(X1).unsqueeze(-1) * X2, dim=1)

            loss = criterion(Y_pred, Y.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 100 == 99:
                print("[%d, %5d] loss: %.10f" % (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0
    end = datetime.datetime.now()

    checkpoint = {
      "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, model_path)
    print("Training finished in {} minutes.".format((end - start).seconds / 60.0))

def test(text_path, output_path, pred_paths, model_path, batch_size):
    models = [read_file(pred_path) for pred_path in pred_paths]

    test_dataset = SimpleMoEDataset(text_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    checkpoint = torch.load(model_path, weights_only=True)
    model = SimpleMoEClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    responses = []
    for i, data in enumerate(test_loader):
        X1 = data
        Y_pred = model.predict(X1.to(device))
        for j, y in enumerate(Y_pred):
            responses.append(models[y][i * batch_size + j] + "\n")

    with open(output_path, "w", encoding="utf-8") as predicted_file:
        predicted_file.writelines(responses)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-text", help="Text file path containing untranslated Chinese text", required=True)
    parser.add_argument("-pred", nargs="+", help="Prediction file path containing translated English text", required=True)
    parser.add_argument("-label", help="Label file path containing translated English text")
    parser.add_argument("-out", help="Output file path")
    parser.add_argument("-model", help="Model path", required=True)
    parser.add_argument("-batch", help="Batch size for predictions", type=int, default=32)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    if args.label: 
        train(args.text, args.label, args.pred, args.model, args.batch)
    else: 
        test(args.text, args.out, args.pred, args.model, args.batch)

