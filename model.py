# Code for the actual model

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# Hyperparameters (TODO: move these somewhere else (argparse)?)
embd_dim = 384
ff_hidden_size = 2048
mha_heads = 6
num_blocks = 6
context_size = 32
max_enc_len = 1024
max_dec_len = 1024
lr = 0.0001

class Head(nn.Module):
    def __init__(self, is_masked, head_dim, device):
        super().__init__()
        self.device = device
        self.head_dim = head_dim
        self.w_q = nn.Linear(embd_dim, head_dim).to(device)
        self.w_k = nn.Linear(embd_dim, head_dim).to(device)
        self.w_v = nn.Linear(embd_dim, head_dim).to(device)
        self.is_masked = is_masked

    def forward(self, x, enc = None):
        # x is (B, T, C)
        _, T, _ = x.shape
        if enc is None:
            query, key, value = self.w_q(x), self.w_k(x), self.w_v(x) # Each (B, T, H)
        else:
            query, key, value = self.w_q(x), self.w_k(enc), self.w_v(enc) # Each (B, T, H)
        res = query @ key.transpose(1, 2) # (B, T, T)
        res /= self.head_dim ** 0.5 # (B, T, T)

        mask = torch.tril(torch.ones((T, T))).to(self.device)
        if self.is_masked:
            res = res.masked_fill(mask == 0, float("-inf"))
        res = F.softmax(res, dim=-1)
        res = res @ value
        return res

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, is_masked, device):
        super().__init__()
        self.device = device
        head_dim = embd_dim // num_heads
        self.heads = [Head(is_masked=is_masked, head_dim=head_dim, device=device) for _ in range(num_heads)]
        self.lin = nn.Linear(embd_dim, embd_dim)

    def forward(self, x, enc = None):
        res = [h(x, enc) for h in self.heads]
        res = torch.cat(res, dim=-1).to(self.device)
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
    def __init__(self, device):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=mha_heads, is_masked=False, device=device)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.ln2 = nn.LayerNorm(embd_dim)

    def forward(self, x):
        x = x + self.mha(x)
        x = self.ln1(x)
        x = x + self.ff(x)
        x = self.ln2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.masked_mha = MultiHeadAttention(num_heads=mha_heads, is_masked=True, device=device)
        self.mha = MultiHeadAttention(num_heads=mha_heads, is_masked=False, device=device)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.ln2 = nn.LayerNorm(embd_dim)
        self.ln3 = nn.LayerNorm(embd_dim)

    def forward(self, x, enc):
        x = x + self.masked_mha(x)
        x = self.ln1(x)
        x = x + self.mha(x, enc)
        x = self.ln2(x)
        x = x + self.ff(x)
        x = self.ln3(x)
        return x

class Transformer(nn.Module):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device

        self.tok_embd_enc = nn.Embedding(num_embeddings=tokenizer.n, embedding_dim=embd_dim)
        self.tok_embd_dec = nn.Embedding(num_embeddings=tokenizer.n, embedding_dim=embd_dim)
        self.pos_embd_enc = nn.Embedding(num_embeddings=max_enc_len, embedding_dim=embd_dim)
        self.pos_embd_dec = nn.Embedding(num_embeddings=max_dec_len, embedding_dim=embd_dim)

        self.encoder_blocks = nn.Sequential(*(EncoderBlock(device) for _ in range(num_blocks))) # Pass each decoder block as an arg
        self.decoder_blocks = nn.ModuleList([DecoderBlock(device) for _ in range(num_blocks)]) # Pass each decoder block as an arg
        self.lin = nn.Linear(embd_dim, self.tokenizer.n)

        self.optim = optim.AdamW(self.parameters(), lr=lr)

    def forward(self, enc_in, dec_in):
        _, enc_t = enc_in.shape
        _, dec_t = dec_in.shape
        enc_in = self.tok_embd_enc(enc_in) + self.pos_embd_enc(self.pos_enc(enc_t))
        dec_in = self.tok_embd_dec(dec_in) + self.pos_embd_dec(self.pos_enc(dec_t))
        enc_out = self.encoder_blocks(enc_in)
        res = self.decoder_blocks[0](dec_in, enc_out)
        for d in self.decoder_blocks[1:]:
            res = d(res, enc_out)
        res = self.lin(res)
        return res

    def train(self, train_loader, epochs):
        for e in range(epochs):
            if e % 100 == 0:
                print(f"---Epoch {e}---")
            in_lang, out_lang = next(iter(train_loader))
            # Each (B, T)
            enc_in = process_batch(in_lang, self.tokenizer, special_tokens=self.tokenizer.special_tokens)
            out_lang_processed = process_batch(out_lang, self.tokenizer, special_tokens=self.tokenizer.special_tokens)

            dec_in = out_lang_processed[:, :-1]
            target = out_lang_processed[:, 1:]

            enc_in, dec_in, target = enc_in.to(self.device), dec_in.to(self.device), target.to(self.device)

            out = self(enc_in, dec_in)
            B, T, C = out.shape
            out = out.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(out, target)

            loss.backward()
            self.optim.zero_grad()
            self.optim.step()

    # TODO: Do the sin cos thing described in paper
    def pos_enc(self, length):
        return torch.arange(length).to(self.device)
    
    def generate(self, context, num_tokens):
        """
        Context: string
        """
        tok_con = self.tokenizer.encode(context)
        for _ in num_tokens:
            pass
        # Need to implement this
        return context

def process_batch(batch, tokenizer, special_tokens):
    bos = tokenizer.encode(special_tokens["bos"])
    eos = tokenizer.encode(special_tokens["eos"])
    tokenized = [torch.tensor(bos + tokenizer.encode(b) + eos, dtype=torch.long) for b in batch]
    batch = pad_sequence(tokenized, batch_first=True, padding_value=tokenizer.tok_to_int[special_tokens["pad"]])
    return batch