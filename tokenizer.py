# Nicholas DeJesse (ndejesse@u.rochester.edu)
# Diego Rodriguez (drodri41@u.rochester.edu)

# Code for tokenizer

import json
import numpy as np
import pandas as pd

class Tokenizer:
    def __init__(self, data, special_tokens, vocab_size):
        """
        Args:
            data (string): The training data used to determine the tokens.
        """
        if type(data) is pd.DataFrame:
            self.text = "".join(data.to_numpy().flatten())
        else:
            self.text = data
        self.tokens = self.text.encode("utf-8")
        self.special_tokens = special_tokens
        self.tokens = list(map(int, self.tokens))
        self.sp_tok_to_id = {}
        self.merges = {}
        self.sp_ids = set(self.sp_tok_to_id.values())
        self.vocab_size = vocab_size
        self.vocab = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for i, tok in enumerate(self.special_tokens.values()):
            self.sp_tok_to_id[tok] = vocab_size - len(self.special_tokens) + i
        self.sp_ids = set(self.sp_tok_to_id.values())
        for tok, idx in self.sp_tok_to_id.items():
            self.vocab[idx] = tok.encode("utf-8")

    def train(self):
        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        ids = list(self.tokens)

        #Identifying tokens for merging
        for i in range(num_merges):
            if i % 100 == 0:
                print(f"Merge {i}/{num_merges}")
            stats = get_stats(ids, self.sp_ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = bpe_merge(ids, self.sp_ids, pair, idx)
            self.merges[pair] = idx
        # For decoding
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]


    def decode(self,ids):
        #Unmerging tokens
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self,text):
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for sp_tok, sp_id in self.sp_tok_to_id.items():
                if text.startswith(sp_tok, i):
                    tokens.append(sp_id)
                    i += len(sp_tok)
                    matched = True
                    break
            if not matched:
                char_bytes = text[i].encode("utf-8")
                tokens.extend(list(char_bytes))
                i += 1
        while len(tokens) >= 2:
            stats = get_stats(tokens, self.sp_ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = bpe_merge(tokens, self.sp_ids, pair, idx)
        return tokens

    def save(self, path):
        data = {
            "vocab": {str(k): v.decode("utf-8", errors="replace") for k, v in self.vocab.items()},
            "merges": {f"{a},{b}": idx for (a, b), idx in self.merges.items()},
            "special_tokens": self.special_tokens,
            "sp_tok_to_id": self.sp_tok_to_id,
            "vocab_size": len(self.vocab)
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path, vocab_size):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.__init__(data="", special_tokens=data["special_tokens"], vocab_size=vocab_size)

        # Restore core structures
        self.vocab = {int(k): v.encode("utf-8") for k, v in data["vocab"].items()}
        self.vocab_size = vocab_size
        self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
        self.sp_tok_to_id = {k: int(v) for k, v in data["sp_tok_to_id"].items()}
        self.sp_ids = set(self.sp_tok_to_id.values())


def get_stats(ids, sp_ids):
    counts = {}

    for a, b in zip(ids, ids[1:]):
        if a in sp_ids or b in sp_ids:
            continue
        pair = (a, b)
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def bpe_merge(ids, sp_ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        #
        if ids[i] not in sp_ids and ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1] and ids[
            i + 1] not in sp_ids:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# Quick test to make sure everything works
if __name__ == "__main__":
    train = "Wowwy, CSC 246 is an amazing course!"
    special = {
        "[BOS]": "[BOS]",
        "[EOS]": "[EOS]",
        "[PAD]": "[PAD]"

    }

    tkn = Tokenizer(train, special_tokens=special,vocab_size=276)
    #tkn.train()

    test = "[BOS]Wowy, español CSC 246 is an amazing course![EOS][PAD][PAD]"
    print(f"Encoded: {tkn.encode(test)}")
    print(f"Decoded: {tkn.decode(tkn.encode(test))}")