# Code for the actual model

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import os

from bleu import calculate_bleu

# Hyperparameters (TODO: move these somewhere else (argparse)?)
d_model = 512
ff_hidden_size = 2048
mha_heads = 8
num_encoder_blocks = 6
num_decoder_blocks = 6
max_encoder_len = 512
max_decoder_len = 512
eval_iters = 8
dropout = 0.3
pos_embd_mode = "sin"

lr = 3e-4
# Increase lr at start then start decreasing it after warmup_steps
warmup_steps = 4000

class Head(nn.Module):
    def __init__(self, is_masked, head_dim, device):
        super().__init__()
        self.device = device
        self.head_dim = head_dim
        self.w_q = nn.Linear(d_model, head_dim).to(device)
        self.w_k = nn.Linear(d_model, head_dim).to(device)
        self.w_v = nn.Linear(d_model, head_dim).to(device)
        self.is_masked = is_masked
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc = None):
        # x is (B, T, C)
        _, T, _ = x.shape
        if enc is None:
            query, key, value = self.w_q(x), self.w_k(x), self.w_v(x) # Each (B, T, H)
        else:
            query, key, value = self.w_q(x), self.w_k(enc), self.w_v(enc) # Each (B, T, H)
        res = query @ key.transpose(1, 2) # (B, T, T)
        res /= self.head_dim ** 0.5 # (B, T, T)

        if self.is_masked:
            mask = torch.tril(torch.ones((T, T))).to(self.device)
            res = res.masked_fill(mask == 0, float("-inf"))
        res = F.softmax(res, dim=-1)
        res = res @ value
        res = self.dropout(res)
        return res

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, is_masked, device):
        super().__init__()
        self.device = device
        head_dim = d_model // num_heads
        self.heads = [Head(is_masked=is_masked, head_dim=head_dim, device=device) for _ in range(num_heads)]
        self.lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc = None):
        res = [h(x, enc) for h in self.heads]
        res = torch.cat(res, dim=-1)
        res = self.lin(res)
        res = self.dropout(res)
        return res

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff_stack = nn.Sequential(
            nn.Linear(d_model, ff_hidden_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_size, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        res = self.ff_stack(x)
        return res

class EncoderBlock(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=mha_heads, is_masked=False, device=device)
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

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
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

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

        self.tok_embd_enc = nn.Embedding(num_embeddings=tokenizer.n, embedding_dim=d_model)
        self.tok_embd_dec = nn.Embedding(num_embeddings=tokenizer.n, embedding_dim=d_model)
        self.pos_embd_enc = nn.Embedding(num_embeddings=max_encoder_len, embedding_dim=d_model)
        self.pos_embd_dec = nn.Embedding(num_embeddings=max_decoder_len, embedding_dim=d_model)

        # Pass each encoder block as an arg
        self.encoder_blocks = nn.Sequential(*(EncoderBlock(device) for _ in range(num_encoder_blocks)))
        # Kept as list because sequential can't take in multiple inputs (and we need to pass in encoder logits)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(device) for _ in range(num_decoder_blocks)])
        self.lin = nn.Linear(d_model, self.tokenizer.n)
        self.dropout = nn.Dropout(dropout)

        lmda = lambda epoch : (d_model ** -0.5) * min((epoch + 1) ** -0.5, (epoch + 1) * warmup_steps ** -1.5)
        self.optim = optim.AdamW(self.parameters(), lr=1)
        self.lr_schedule = scheduler.LambdaLR(self.optim, lmda)

        self.loss_info = {"Batch": [], "Train loss": [], "Test loss": []}


    def forward(self, enc_in, dec_in):
        _, enc_t = enc_in.shape
        _, dec_t = dec_in.shape
        enc_in = self.tok_embd_enc(enc_in) + self.pos_embd_enc(self.pos_enc(enc_t))
        dec_in = self.tok_embd_dec(dec_in) + self.pos_embd_dec(self.pos_enc(dec_t))
        enc_in = self.dropout(enc_in)
        dec_in = self.dropout(dec_in)
        enc_out = self.encoder_blocks(enc_in)
        for d in self.decoder_blocks:
            dec_in = d(dec_in, enc_out)
        dec_in = self.lin(dec_in)
        return dec_in # (B, T, C)

    def train_model(self, train_loader, test_loader, num_batches, checkpoint_dir):
        self.train()
        for b in range(num_batches):
            if b % 1000 == 0:
                chk_dir = f"{checkpoint_dir}{b}"
                os.makedirs(chk_dir, exist_ok=True)
                full_path = f"{chk_dir}/checkpoint.pt"
                torch.save(self.state_dict(), full_path)
                print(f"Checkpoint saved as {full_path}\n")
            if b % 100 == 0:
                train_loss, test_loss = self.estimate_loss(train_loader, test_loader)
                train_bleu, test_bleu = self.estimate_bleu_score(train_loader, test_loader)
                print(f"---Epoch {b}, Train loss: {train_loss:.3f}, Test loss: {test_loss:.3f}, Train BLEU: {train_bleu:.5f}, Test BLEU: {test_bleu:.5f}---")
                self.loss_info["Batch"].append(b)
                self.loss_info["Train loss"].append(train_loss)
                self.loss_info["Test loss"].append(test_loss)
                self.loss_info["Train BLEU"].append(train_bleu)
                self.loss_info["Test BLEU"].append(test_bleu)
            
            enc_in, dec_in, target = get_batch(train_loader, self.tokenizer, self.device)

            out = self(enc_in, dec_in)
            B, T, C = out.shape
            out = out.view(B*T, C)
            target = target.view(-1)
            # Ignore padding tokens in loss
            loss = F.cross_entropy(out, target, ignore_index=self.tokenizer.encode(self.tokenizer.special_tokens["pad"])[0])

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.lr_schedule.step()

        pd.DataFrame(self.loss_info).to_csv(checkpoint_dir, index=False)

    # TODO: Do the sin cos thing described in paper
    def pos_enc(self, length):
        if pos_embd_mode == "sin":
            # Sinusoidal positional embedding shown in AIAYN
            pos = torch.arange(length, dtype=torch.float32).unsqueeze(1) # Row, has to be float32 instead of int64 because we are dividing
            i = (torch.arange(d_model // 2)).unsqueeze(1).T # Column
            i = 10000 ** -(2 * i / d_model)
            # Number of columns is half of what we need because it will be spread out when creating final result
            ang = pos @ i # Inner product to get inputs to trig functions
            res = torch.zeros((length, d_model))
            res[:,0::2] = torch.sin(ang)
            res[:,1::2] = torch.cos(ang)
            return res
        elif pos_embd_mode == "add":
            return torch.arange(length).to(self.device)
        else:
            raise NotImplementedError(pos_embd_mode)
    
    def generate(self, encoder_input):
        encoder_input = torch.tensor(self.tokenizer.encode(encoder_input), dtype=torch.long, device=self.device).unsqueeze(0)
        context = torch.tensor(self.tokenizer.encode(self.tokenizer.special_tokens["bos"]), dtype=torch.long, device=self.device).unsqueeze(0)
        for _ in range(max_decoder_len):
            with torch.no_grad():
                out = self(encoder_input, context)
            out = out[:, -1, :] # Extract data for last token (B, C)
            out = F.softmax(out, 1) # (B, C)
            # tok = out.argmax(1).unsqueeze(0) # Greedy
            tok = torch.multinomial(out, num_samples=1)
            context = torch.cat((context, tok), dim=1)
            if tok.squeeze().item() == self.tokenizer.encode(self.tokenizer.special_tokens["eos"])[0]:
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
                total_loss += F.cross_entropy(out, target, ignore_index=self.tokenizer.encode(self.tokenizer.special_tokens["pad"])[0])
            total_loss /= eval_iters
            losses.append(total_loss)
        self.train()
        return losses

    def estimate_bleu_score(self, train_loader, test_loader):
        self.eval()
        bleus = []
        for loader in [train_loader, test_loader]:
            total_bleu = 0
            for k in range(eval_iters):
                in_lang, out_lang = next(iter(loader))
                translated = self.generate(in_lang)[1:-1] # Remove <BOS> and <EOS>
                total_bleu += calculate_bleu(translated, self.tokenizer.encode(out_lang))
            total_bleu /= eval_iters
            bleus.append(total_bleu)
        self.train()
        return bleus

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

def main():
    length = 3
    d_model = 4


    print(res)
    print(res.shape)
if __name__ == "__main__":
    main()