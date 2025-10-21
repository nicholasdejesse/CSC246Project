# Code for the actual model

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters (TODO: move these somewhere else (argparse)?)
embd_dim = 32

class Head(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # self.ff_stack = nn.Sequential(
            
        # )

    def forward(self, x):
        pass

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class Transformer(nn.Module):
    def __init__(self, n_embd, n_blocks):
        super().__init__()
        self.embd = nn.Embedding(num_embeddings=n_embd, embedding_dim=embd_dim)
        self.blocks = [TransformerBlock() for _ in range(n_blocks)]

    def forward(self, x):
        x = self.embd(x) + self.pos_enc(x)
        for b in self.blocks:
            x = b(x)
        return x

    def pos_enc(self, x):
        pass