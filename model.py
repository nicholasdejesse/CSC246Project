# Code for the actual model

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters (TODO: move these somewhere else (argparse)?)
embd_dim = 32
head_dim = 16
ff_hidden_size = 2048

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_q = nn.Linear(embd_dim, head_dim)
        self.w_k = nn.Linear(embd_dim, head_dim)
        self.w_v = nn.Linear(embd_dim, head_dim)

    def forward(self, x):
        # x is (B, T, C)
        _, T, _ = x.shape
        query, key, value = self.w_q(x), self.w_k(x), self.w_v(x) # Each (B, T, H)
        res = query @ key.transpose(1, 2) # (B, T, T)
        res /= head_dim ** 0.5 # (B, T, T)

        mask = torch.tril(torch.ones((T, T)))
        res = res.masked_fill(mask == 0, float("-inf"))
        res = F.softmax(res, dim=-1)
        res = res @ value
        return res

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.heads = [Head() for _ in range(num_heads)]
        self.lin = nn.Linear(embd_dim, embd_dim)

    def forward(self, x):
        res = [h(x) for h in self.heads]
        res = torch.cat(res, dim=-1)
        res = self.lin(res)
        return res


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff_stack = nn.Sequential(
            nn.Linear(embd_dim, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, embd_dim)
        )

    def forward(self, x):
        res = self.ff_stack(x)
        return res

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class Transformer(nn.Module):
    def __init__(self, n_embd, n_blocks):
        super().__init__()
        self.embd = nn.Embedding(num_embeddings=n_embd, embedding_dim=embd_dim)
        self.blocks = [DecoderBlock() for _ in range(n_blocks)]

    def forward(self, x):
        x = self.embd(x) + self.pos_enc(x)
        for b in self.blocks:
            x = b(x)
        return x

    def pos_enc(self, x):
        return torch.arange(len(x))