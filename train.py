# Script to actually train the model
# import kagglehub
from torch.utils.data import random_split, DataLoader
import pandas as pd

from dataset import EnglishToSpanish
from tokenizer import Tokenizer
from model import Transformer

def main():
    # TODO: Figure out how to do this, but download to data folder instead
    # path = kagglehub.dataset_download("lonnieqin/englishspanish-translation-dataset")
    # print("Path to dataset files:", path)

    data = EnglishToSpanish("./data/data.csv")
    special = {
        "bos": "[BOS]",
        "eos": "[EOS]",
        "pad": "[PAD]"
    }

    print("Training tokenizer...")
    tokenizer = Tokenizer(data.df, special_tokens=special)
    print("Tokenizer training complete.")
    print(f"Vocab: {tokenizer.vocab}")

    print("Processing data...")
    data.preprocess(tokenizer, max_len=10, special_tokens=special)
    print("Data processing complete.")

    # print(data.df[0:10])

    train_data, test_data = random_split(data, [0.7, 0.3])
    
    train_loader = DataLoader(train_data)
    test_loader = DataLoader(test_data)

    model = Transformer(tokenizer)


if __name__ == "__main__":
    main()