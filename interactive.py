import torch

from model import Transformer
from dataset import EnglishToSpanish
from tokenizer import Tokenizer

model_load_path = "./checkpoints/50k_smaller_model/50000/checkpoint.pt"
tokenizer_path = "./tokenizer.json"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = EnglishToSpanish("./data.csv")
    special = {
        "bos": "<BOS>",
        "eos": "<EOS>",
        "pad": "<PAD>"
    }

    tokenizer = Tokenizer(data.df, special_tokens=special, vocab_size=8003)
    try:
        tokenizer.load(tokenizer_path, vocab_size=8003)
    except:
        raise Exception(f"Could not load tokenizer at path {tokenizer_path}. Are you sure you trained it?")

    model = Transformer(tokenizer, device)
    model.load_state_dict(torch.load(model_load_path)["model"])
    model.to(device)

    model.eval()
    while True:
        encoder_in = input("Enter English words: ")
        out = model.generate(encoder_in)
        print(f"Translated: {tokenizer.decode(out.squeeze().tolist())}")

if __name__ == "__main__":
    main()