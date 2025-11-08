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
num_encoder_blocks = 6
num_decoder_blocks = 6
max_encoder_len = 512
max_decoder_len = 512
eval_iters = 8
lr = 0.0003

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
        self.pos_embd_enc = nn.Embedding(num_embeddings=max_encoder_len, embedding_dim=embd_dim)
        self.pos_embd_dec = nn.Embedding(num_embeddings=max_decoder_len, embedding_dim=embd_dim)

        # Pass each encoder block as an arg
        self.encoder_blocks = nn.Sequential(*(EncoderBlock(device) for _ in range(num_encoder_blocks)))
        # Kept as list because sequential can't take in multiple inputs (and we need to pass in encoder logits)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(device) for _ in range(num_decoder_blocks)])
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
        return res # (B, T, C)

    def train_model(self, train_loader, test_loader, epochs):
        self.train()
        for e in range(epochs):
            if e % 10 == 0:
                train_loss, test_loss = self.estimate_loss(train_loader, test_loader)
                print(f"---Epoch {e}, Train loss: {train_loss:.3f}, Test loss: {test_loss:.3f}---")
            
            enc_in, dec_in, target = get_batch(train_loader, self.tokenizer, self.device)

            out = self(enc_in, dec_in)
            B, T, C = out.shape
            out = out.view(B*T, C)
            target = target.view(-1)
            loss = F.cross_entropy(out, target)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    # TODO: Do the sin cos thing described in paper
    def pos_enc(self, length):
        return torch.arange(length).to(self.device)
    
    def generate(self, encoder_input):
        encoder_input = torch.tensor(self.tokenizer.encode(encoder_input), dtype=torch.long, device=self.device).unsqueeze(0)
        context = self.tokenizer.special_tokens["bos"]
        for _ in range(max_decoder_len):
            dec_in = torch.tensor(self.tokenizer.encode(context), dtype=torch.long, device=self.device).unsqueeze(0)
            with torch.no_grad():
                out = self(encoder_input, dec_in)
            out = out[:, -1, :] # Extract data for last token (1, C)
            out = torch.softmax(out, -1)
            tok = torch.multinomial(out, num_samples=1)
            decoded = self.tokenizer.decode(tok.flatten().tolist())
            context += decoded
            if decoded == self.tokenizer.special_tokens["eos"]:
                break   
        return context
    
    def estimate_loss(self, train_loader, test_loader):
        self.eval()
        losses = []
        for loader in [train_loader, test_loader]:
            total_loss = 0
            for k in range(eval_iters):
                enc_in, dec_in, target = get_batch(loader, self.tokenizer, self.device)
                with torch.no_grad():
                    out = self(enc_in, dec_in)
                B, T, C = out.shape
                out = out.view(B*T, C)
                target = target.view(-1)
                total_loss += F.cross_entropy(out, target)
            total_loss /= k
            losses.append(total_loss)
        self.train()
        return losses

def get_batch(loader, tokenizer, device):
    in_lang, out_lang = next(iter(loader))
    # Each (B, T)
    enc_in = process_batch(in_lang, tokenizer, special_tokens=tokenizer.special_tokens)
    out_lang_processed = process_batch(out_lang, tokenizer, special_tokens=tokenizer.special_tokens)

    dec_in = out_lang_processed[:, :-1]
    target = out_lang_processed[:, 1:]

    enc_in, dec_in, target = enc_in.to(device), dec_in.to(device), target.to(device)

    return enc_in, dec_in, target

def process_batch(batch, tokenizer, special_tokens):
    bos = tokenizer.encode(special_tokens["bos"])
    eos = tokenizer.encode(special_tokens["eos"])
    tokenized = [torch.tensor(bos + tokenizer.encode(b) + eos, dtype=torch.long) for b in batch]
    batch = pad_sequence(tokenized, batch_first=True, padding_value=tokenizer.tok_to_int[special_tokens["pad"]])
    return batch