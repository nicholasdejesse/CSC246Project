# Code for tokenizer
import numpy as np
import pandas as pd

# Currently character level, needs to be "chunks" of characters at some point
class Tokenizer:
    def __init__(self, data, special_tokens):
        """
        Args:
            data (string): The training data used to determine the tokens.
        """
        if type(data) is pd.DataFrame:
            txt = "".join(data.to_numpy().flatten())
        else:
            txt = data
        self.vocab = sorted(list(set(txt)))
        for t in special_tokens.values():
            self.vocab.append(t)
        # For now, just maps individual characters to ints
        self.n = len(self.vocab)
        self.tok_to_int = {t:i for i, t in enumerate(self.vocab)}
        self.int_to_tok = {i:t for i, t in enumerate(self.vocab)}

    def encode(self, txt):
        """Takes in a string and encodes it to a list of tokens."""
        ls = []
        i = 0
        while i < len(txt):
            if txt[i] != "[":
                ls.append(self.tok_to_int[txt[i]])
                i += 1
            else:
                ls.append(self.tok_to_int[txt[i:i+5]])
                i += 5
        return ls
        # return [self.tok_to_int[c] for c in txt]

    def decode(self, ls):
        """Takes in a list of tokens and decodes it to a string."""
        return "".join([self.int_to_tok[i] for i in ls])

# Quick test to make sure everything works
if __name__ == "__main__":
    train = "Wowwy, CSC 246 is an amazing course!"
    special = {
        "bos": "[BOS]",
        "eos": "[EOS]",
        "pad": "[PAD]"
    }

    tkn = Tokenizer(train, special_tokens=special)

    test = "[BOS]Wowwy, CSC 246 is an amazing course![EOS][PAD][PAD]"
    print(f"Encoded: {tkn.encode(test)}")
    print(f"Decoded: {tkn.decode(tkn.encode(test))}")