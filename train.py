# Script to actually train the model
# import kagglehub
import torch
from torch.utils.data import random_split, DataLoader

from dataset import EnglishToSpanish
from tokenizer import Tokenizer
from model import Transformer

# Hyperparameters
batch_size = 64
epochs = 100

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO: Figure out how to do this, but download to data folder instead
    # path = kagglehub.dataset_download("lonnieqin/englishspanish-translation-dataset")
    # print("Path to dataset files:", path)

    data = EnglishToSpanish("./data/data.csv")
    special = {
        "bos": "<BOS>",
        "eos": "<EOS>",
        "pad": "<PAD>"
    }

    print("Training tokenizer...")
    tokenizer = Tokenizer(data.df, special_tokens=special)
    print("Tokenizer training complete.")
    # print(data.df[0:10])

    train_data, test_data = random_split(data, [0.7, 0.3])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = Transformer(tokenizer, device)
    model.to(device)
    model.train(train_loader=train_loader, epochs=100)


if __name__ == "__main__":
    main()