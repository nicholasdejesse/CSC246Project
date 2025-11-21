# Code for tokenizer
import numpy as np
import pandas as pd

# Currently character level, needs to be "chunks" of characters at some point
class Tokenizer:
    def __init__(self, data, special_tokens, vocab_size):
        """
        Args:
            data (string): The training data used to determine the tokens.
        """
        if type(data) is pd.DataFrame:
            text = "".join(data.to_numpy().flatten())
        else:
            text = data
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))
        num_merges = vocab_size - 256
        ids = list(tokens)

        self.sp_tok_to_id = {}

        for i, tok in enumerate(special_tokens.values()):
            self.sp_tok_to_id[tok] = vocab_size + i

        self.sp_ids = set(self.sp_tok_to_id.values())
        self.merges = {}
        for i in range(num_merges):
            stats = get_stats(ids, self.sp_ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = bpe_merge(ids, self.sp_ids, pair, idx)
            self.merges[pair] = idx
        # Decoding
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        for tok, idx in self.sp_tok_to_id.items():
            self.vocab[idx] = tok.encode("utf-8")


    def decode(self,ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    # Encoding
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
                tokens.append(text.encode("utf-8")[i])
                i += 1
        while len(tokens) >= 2:
            stats = get_stats(tokens, self.sp_ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = bpe_merge(tokens, self.sp_ids, pair, idx)
        return tokens

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
        "bos": "[BOS]",
        "eos": "[EOS]",
        "pad": "[PAD]"

    }

    tkn = Tokenizer(train, special_tokens=special, vocab_size=276)

    test = "[BOS]Wowwy, CSC 246 is an amazing course![EOS][PAD][PAD]"
    print(f"Encoded: {tkn.encode(test)}")
    print(f"Decoded: {tkn.decode(tkn.encode(test))}")