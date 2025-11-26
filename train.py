# Script to actually train the model
# import kagglehub
import os

import torch
from torch.utils.data import random_split, DataLoader
import pandas as pd

from dataset import EnglishToSpanish
from tokenizer import Tokenizer
from model import Transformer

# Hyperparameters
batch_size = 64
num_batches = 50000
vocab_size = 8000

model_name = "50k_smaller_model"
checkpoint_save_dir = f"./checkpoints/{model_name}/"
tokenizer_save_dir = f"./tokenizer.json"

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

    tokenizer = Tokenizer(data.df, special_tokens=special)
    if os.path.exists(tokenizer_save_dir):
        try:
            tokenizer.load(tokenizer_save_dir)
        except:
            raise Exception(f"Could not load tokenizer at path {tokenizer_save_dir}")
    else:
        print("Training tokenizer...")
        tokenizer.train()
        tokenizer.save(tokenizer_save_dir)
        print("Tokenizer training complete.")

    train_data, test_data = random_split(data, [0.7, 0.3])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = Transformer(tokenizer, device, scheduler_num_batches=num_batches)
    model.to(device)

    # Sum all parameters
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Model is also saved to disk in this function
    model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        num_batches=num_batches,
        checkpoint_dir=checkpoint_save_dir,
        save_intermediate_checkpoints=False,
        calculate_bleu=True, # Only set to False when debugging (BLEU takes a while to calculate)
        print_examples=True
    )

if __name__ == "__main__":
    main()