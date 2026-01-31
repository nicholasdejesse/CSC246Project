# train.py
import os
import torch
from torch.utils.data import random_split, DataLoader
import pandas as pd

from dataset import EnglishToSpanish
from tokenizer import Tokenizer
from model import Transformer

# Hyperparameters
batch_size = 32
num_batches = 100000
vocab_size = 40003

model_name = "50k_new_model"
checkpoint_save_dir = f"./checkpoints/{model_name}/"
tokenizer_save_dir = f"./tokenizer.json"

def collate_keep_strings(batch):
    """Transforms list of tuples into two lists: src and tgt"""
    in_lang, out_lang = zip(*batch)  # unzip the list of tuples
    return list(in_lang), list(out_lang)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = EnglishToSpanish("./data1.csv")

    special = {
        "bos": "<BOS>",
        "eos": "<EOS>",
        "pad": "<PAD>"
    }

    # ---------------- TOKENIZER LOGIC ----------------
    if os.path.isfile(tokenizer_save_dir):
        print("Loading tokenizer...")
        tokenizer = Tokenizer(
            data="",
            special_tokens=special,
            vocab_size=vocab_size
        )
        tokenizer.load(tokenizer_save_dir)
        print("Tokenizer loaded successfully.")
    else:
        print("Training tokenizer...")
        tokenizer = Tokenizer(
            data.df,
            special_tokens=special,
            vocab_size=vocab_size
        )
        tokenizer.train()
        tokenizer.save(tokenizer_save_dir)
        print("Tokenizer training complete.")
    # -------------------------------------------------

    train_data, test_data = random_split(data, [int(len(data)*0.7), len(data) - int(len(data)*0.7)])

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_keep_strings
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_keep_strings
    )

    model = Transformer(tokenizer, device, scheduler_num_batches=num_batches)
    model.to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    model.train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        num_batches=num_batches,
        checkpoint_dir=checkpoint_save_dir,
        save_intermediate_checkpoints=False,
        calculate_bleu=True,
        print_examples=True
    )

if __name__ == "__main__":
    main()
