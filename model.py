# Code for the actual model

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters (TODO: move these somewhere else (argparse)?)
embd_dim = 384
ff_hidden_size = 2048
mha_heads = 6
num_blocks = 6

class Head(nn.Module):
    def __init__(self, is_masked, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.w_q = nn.Linear(embd_dim, head_dim)
        self.w_k = nn.Linear(embd_dim, head_dim)
        self.w_v = nn.Linear(embd_dim, head_dim)
        self.is_masked = is_masked

    def forward(self, x, enc = None):
        # x is (B, T, C)
        _, T, _ = x.shape
        if self.is_masked:
            query, key, value = self.w_q(x), self.w_k(x), self.w_v(x) # Each (B, T, H)
        else:
            query, key, value = self.w_q(x), self.w_k(enc), self.w_v(enc) # Each (B, T, H)
        res = query @ key.transpose(1, 2) # (B, T, T)
        res /= self.head_dim ** 0.5 # (B, T, T)

        mask = torch.tril(torch.ones((T, T)))
        if self.is_masked:
            res = res.masked_fill(mask == 0, float("-inf"))
        res = F.softmax(res, dim=-1)
        res = res @ value
        return res

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, is_masked):
        super().__init__()
        head_dim = embd_dim // num_heads
        self.heads = [Head(is_masked=is_masked, head_dim=head_dim) for _ in range(num_heads)]
        self.lin = nn.Linear(embd_dim, embd_dim)

    def forward(self, x, enc = None):
        res = [h(x, enc) for h in self.heads]
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

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=mha_heads, is_masked=False)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

    def forward(self, x):
        x += self.mha(x)
        x = self.ln1(x)
        x += self.ff(x)
        x = self.ln2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.masked_mha = MultiHeadAttention(num_heads=mha_heads, is_masked=True)
        self.mha = MultiHeadAttention(num_heads=mha_heads, is_masked=False)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.ln3 = nn.LayerNorm()

    def forward(self, x, enc):
        x += self.masked_mha(x)
        x = self.ln1(x)
        x += self.mha(x, enc)
        x = self.ln2(x)
        x += self.ff(x)
        x = self.ln3(x)
        return x

class Transformer(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.embd = nn.Embedding(num_embeddings=embd_dim, embedding_dim=embd_dim)
        self.encoder_blocks = nn.Sequential(*(DecoderBlock() for _ in range(num_blocks))) # Pass each decoder block as an arg
        self.decoder_blocks = nn.Sequential(*(DecoderBlock() for _ in range(num_blocks))) # Pass each decoder block as an arg
        self.lin = nn.Linear(embd_dim, self.tokenizer.n)

    def forward(self, x):
        x = self.embd(x) + self.pos_enc(x)
        enc_out = self.encoder_blocks(x)
        x = self.decoder_blocks(x, enc_out)
        x = self.lin(x)
        return x

    # TODO: Do the sin cos thing described in paper
    def pos_enc(self, x):
        return torch.arange(len(x))
    
    def generate(self, context, num_tokens):
        """
        Context: string
        """
        tok_con = self.tokenizer.encode(tok_con)
        for _ in num_tokens:
            pass
        # Need to implement this
        return context
