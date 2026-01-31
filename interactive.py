# Nicholas DeJesse (ndejesse@u.rochester.edu)
# Diego Rodriguez (drodri41@u.rochester.edu)

import torch

from model import Transformer
from dataset import EnglishToSpanish
from tokenizer import Tokenizer

model_load_path = "./checkpoints/50k_new_model/100000/checkpoint.pt"
tokenizer_path = "./tokenizer.json"

vocab_size = 40003

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = EnglishToSpanish("./data1.csv")
    special = {
        "bos": "<BOS>",
        "eos": "<EOS>",
        "pad": "<PAD>"
    }

    tokenizer = Tokenizer(data.df, special_tokens=special, vocab_size=vocab_size)
    try:
        tokenizer.load(tokenizer_path)
    except:
        raise Exception(f"Could not load tokenizer at path {tokenizer_path}. Are you sure you trained it?")

    model = Transformer(tokenizer, device)
    model.load_state_dict(torch.load(model_load_path)["model"])
    model.to(device)

    model.eval()
    # REPL
    while True:
        encoder_in = input("Enter English words: ")
        out = model.generate(encoder_in)
        print(f"Translated: {tokenizer.decode(out.squeeze().tolist())}")

if __name__ == "__main__":
    main()