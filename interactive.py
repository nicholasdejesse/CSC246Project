import torch

from model import Transformer
from dataset import EnglishToSpanish
from tokenizer import Tokenizer

model_load_path = "./models/model.pt"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: Save tokenizer so we don't have to retrain it here
    data = EnglishToSpanish("./data/data.csv")
    special = {
        "bos": "<BOS>",
        "eos": "<EOS>",
        "pad": "<PAD>"
    }

    print("Training tokenizer...")
    tokenizer = Tokenizer(data.df, special_tokens=special)
    print("Tokenizer training complete.")

    model = Transformer(tokenizer, device)
    model.load_state_dict(torch.load(model_load_path))
    model.to(device)

    model.eval()
    while True:
        encoder_in = input("Enter English words:")
        out = model.generate(encoder_in)
        print(out)

if __name__ == "__main__":
    main()