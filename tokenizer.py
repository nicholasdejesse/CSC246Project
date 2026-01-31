# tokenizer.py
# Nicholas DeJesse (ndejesse@u.rochester.edu)
# Diego Rodriguez (drodri41@u.rochester.edu)

import json
import pandas as pd
from collections import Counter

class Tokenizer:
    def __init__(self, data, special_tokens, vocab_size):
        if isinstance(data, pd.DataFrame):
            series = data.select_dtypes(include=["object", "string"]).stack().dropna()
            text = "".join(series.astype(str))
        else:
            text = data

        # ---- USE ONLY ONE-THIRD OF THE DATA (BYTE-SAFE) ----
        full_bytes = text.encode("utf-8")
        third = len(full_bytes) // 3
        self.tokens = list(full_bytes[:third])
        # ---------------------------------------------------

        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        self.merges = {}

        # Base byte vocab
        self.vocab = {i: bytes([i]) for i in range(256)}

        # Special tokens
        self.sp_tok_to_id = {}
        for i, tok in enumerate(self.special_tokens.values()):
            idx = vocab_size - len(self.special_tokens) + i
            self.sp_tok_to_id[tok] = idx
            self.vocab[idx] = tok.encode("utf-8")

        self.sp_ids = set(self.sp_tok_to_id.values())

    def train(self, min_freq=2):
        num_merges = self.vocab_size - 256 - len(self.special_tokens)
        ids = list(self.tokens)

        stats = init_stats(ids, self.sp_ids)

        for i in range(num_merges):
            if not stats:
                break

            pair, freq = stats.most_common(1)[0]
            if freq < min_freq:
                break

            if i % 100 == 0:
                print(f"Merge {i}/{num_merges} | freq={freq}")

            idx = 256 + i
            ids = bpe_merge_with_stats(ids, stats, self.sp_ids, pair, idx)
            self.merges[pair] = idx

        # Build vocab for decoding
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        if not isinstance(text, str):
            text = "" if text is None else str(text)

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
                tokens.extend(text[i].encode("utf-8"))
                i += 1

        while len(tokens) >= 2:
            stats = init_stats(tokens, self.sp_ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = bpe_merge(tokens, self.sp_ids, pair, self.merges[pair])

        return tokens

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        data = {
            "vocab": {str(k): v.decode("utf-8", errors="replace") for k, v in self.vocab.items()},
            "merges": {f"{a},{b}": idx for (a, b), idx in self.merges.items()},
            "special_tokens": self.special_tokens,
            "sp_tok_to_id": self.sp_tok_to_id,
            "vocab_size": self.vocab_size,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.special_tokens = data["special_tokens"]
        self.sp_tok_to_id = {k: int(v) for k, v in data["sp_tok_to_id"].items()}
        self.sp_ids = set(self.sp_tok_to_id.values())

        self.vocab = {int(k): v.encode("utf-8") for k, v in data["vocab"].items()}
        self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
        self.vocab_size = data["vocab_size"]


def init_stats(ids, sp_ids):
    stats = Counter()
    for a, b in zip(ids, ids[1:]):
        if a not in sp_ids and b not in sp_ids:
            stats[(a, b)] += 1
    return stats


def bpe_merge_with_stats(ids, stats, sp_ids, pair, idx):
    newids = []
    i = 0

    while i < len(ids):
        if (
            ids[i] == pair[0]
            and i + 1 < len(ids)
            and ids[i + 1] == pair[1]
            and ids[i] not in sp_ids
            and ids[i + 1] not in sp_ids
        ):
            prev = newids[-1] if newids else None
            next_ = ids[i + 2] if i + 2 < len(ids) else None

            stats[(pair[0], pair[1])] -= 1
            if stats[(pair[0], pair[1])] <= 0:
                del stats[(pair[0], pair[1])]

            if prev is not None:
                stats[(prev, pair[0])] -= 1
                if stats[(prev, pair[0])] <= 0:
                    del stats[(prev, pair[0])]
                stats[(prev, idx)] += 1

            if next_ is not None:
                stats[(pair[1], next_)] -= 1
                if stats[(pair[1], next_)] <= 0:
                    del stats[(pair[1], next_)]
                stats[(idx, next_)] += 1

            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1

    return newids


def bpe_merge(ids, sp_ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if (
            ids[i] == pair[0]
            and i + 1 < len(ids)
            and ids[i + 1] == pair[1]
            and ids[i] not in sp_ids
            and ids[i + 1] not in sp_ids
        ):
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


if __name__ == "__main__":
    train = "Wowwy, CSC 246 is an amazing course!"
    special = {"[BOS]": "[BOS]", "[EOS]": "[EOS]", "[PAD]": "[PAD]"}

    tkn = Tokenizer(train, special_tokens=special, vocab_size=276)

    test = "[BOS]Wowy, español is amazing![EOS][PAD][PAD]"
    encoded = tkn.encode(test)
    print("Encoded:", encoded)
    print("Decoded:", tkn.decode(encoded))
