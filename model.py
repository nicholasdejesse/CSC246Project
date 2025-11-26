# Code for the actual model

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import os
import json

from bleu import calculate_bleu

# Hyperparameters (TODO: move these somewhere else (argparse)?)
d_model = 256
ff_hidden_size = 1024
mha_heads = 4
num_encoder_blocks = 3
num_decoder_blocks = 3
max_encoder_len = 512
max_decoder_len = 512
eval_iters = 1 # Can be low because each iteration evaluates an entire batch
dropout = 0.3
pos_embd_mode = "sin"

save_freq = 1000
log_freq = 1000

# Increase lr at start then start decreasing it after warmup_steps
lr_warmup_steps = 2000
lr_max = 3e-4
lr_min = 1e-6 # Need to look into a better value for this, kind of arbitrary right now

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

    def forward(self, x, enc = None, enc_pad = None, dec_pad = None):
        if enc is None:
            query, key, value = self.w_q(x), self.w_k(x), self.w_v(x) # Each (B, T_dec, H)
        else:
            query, key, value = self.w_q(x), self.w_k(enc), self.w_v(enc) # (B, T_dec, H), (B, T_enc, H), (B, T_enc, H)
            _, T_enc, _ = key.shape
            _, T_dec, _ = query.shape
        res = query @ key.transpose(1, 2) # (B, T, T)
        res /= self.head_dim ** 0.5 # (B, T, T)

        B, T, _ = x.shape
        # Padding mask (i.e. ignore pad tokens)
        if enc_pad is not None:
            mask = enc_pad
            if enc is None:
                mask = mask.unsqueeze(1).expand(B, T, T)
            else:
                mask = mask.unsqueeze(1).expand(B, T_dec, T_enc)
        else:
            mask = dec_pad.unsqueeze(1).expand(B, T, T)

        # Causal mask (i.e. prevent tokens from attending to future ones)
        if self.is_masked:
            # x is (B, T_dec, C)
            mask = torch.logical_and(mask, torch.tril(torch.ones((T, T), dtype=torch.bool)).to(self.device))

        res = res.masked_fill(mask == 0, float("-inf"))

        res = F.softmax(res, dim=-1)
        res = self.dropout(res)
        res = res @ value
        return res

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, is_masked, device):
        super().__init__()
        self.device = device
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList([Head(is_masked=is_masked, head_dim=head_dim, device=device) for _ in range(num_heads)])
        self.lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc = None, enc_pad = None, dec_pad = None):
        res = [h(x=x, enc=enc, enc_pad=enc_pad, dec_pad=dec_pad) for h in self.heads]
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

    def forward(self, x, enc_pad):
        x = x + self.mha(x, enc=None, enc_pad=enc_pad, dec_pad=None)
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

    def forward(self, x, enc = None, enc_pad = None, dec_pad = None):
        x = x + self.masked_mha(x, enc=None, dec_pad=dec_pad)
        x = self.ln1(x)
        x = x + self.mha(x, enc, enc_pad=enc_pad)
        x = self.ln2(x)
        x = x + self.ff(x)
        x = self.ln3(x)
        return x

class Transformer(nn.Module):
    def __init__(self, tokenizer, device, scheduler_num_batches = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device

        self.tok_embd_enc = nn.Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=d_model)
        self.tok_embd_dec = nn.Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=d_model)

        # Pass each encoder block as an arg
        self.encoder_blocks = nn.ModuleList([EncoderBlock(device) for _ in range(num_encoder_blocks)])
        # Kept as list because sequential can't take in multiple inputs (and we need to pass in encoder logits)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(device) for _ in range(num_decoder_blocks)])
        self.lin = nn.Linear(d_model, self.tokenizer.vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.optim = optim.AdamW(self.parameters(), lr=lr_max)

        # This is None when we are running inference
        if scheduler_num_batches is not None:
            # Cosine lr scheduler with warmup
            warmup_scheduler = scheduler.LinearLR(self.optim, start_factor=lr_min, end_factor=1, total_iters=lr_warmup_steps)
            cosine_scheduler = scheduler.CosineAnnealingLR(self.optim, T_max=scheduler_num_batches - lr_warmup_steps, eta_min=lr_min)
            self.lr_schedule = scheduler.SequentialLR(self.optim, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[lr_warmup_steps])

        self.loss_info = {"batch": [], "train_loss": [], "test_loss": [], "train_bleu": [], "test_bleu": []}


    def forward(self, enc_in, dec_in, enc_pad, dec_pad):
        _, enc_t = enc_in.shape
        _, dec_t = dec_in.shape
        # (B, T, C) + (T, C)
        enc_in = self.tok_embd_enc(enc_in) + self.pos_enc(enc_t)
        dec_in = self.tok_embd_dec(dec_in) + self.pos_enc(dec_t)

        enc_in = self.dropout(enc_in)
        dec_in = self.dropout(dec_in)
        
        for e in self.encoder_blocks:
            enc_in = e(enc_in, enc_pad)

        for d in self.decoder_blocks:
            dec_in = d(dec_in, enc_in, enc_pad, dec_pad)
        dec_in = self.lin(dec_in)
        return dec_in # (B, T, C)

    def train_model(self, train_loader, test_loader, num_batches, checkpoint_dir, save_intermediate_checkpoints, calculate_bleu, print_examples):
        self.train()
        self.dump_model_info(checkpoint_dir, num_batches)
        for b in range(num_batches):
            if b % save_freq == 0:
                if save_intermediate_checkpoints:
                    self.save_checkpoint(checkpoint_dir, b+1)
            if b % log_freq == 0:
                train_loss, test_loss = self.estimate_loss(train_loader, test_loader)
                if calculate_bleu:
                    train_bleu, test_bleu = self.estimate_bleu_score(train_loader, test_loader, print_examples)
                else:
                    train_bleu, test_bleu = 0, 0
                print(f"---Batch {b}, Train loss: {train_loss:.3f}, Test loss: {test_loss:.3f}, Train BLEU: {train_bleu:.5f}, Test BLEU: {test_bleu:.5f}---")
                self.loss_info["batch"].append(b)
                self.loss_info["train_loss"].append(train_loss.item())
                self.loss_info["test_loss"].append(test_loss.item())
                self.loss_info["train_bleu"].append(train_bleu)
                self.loss_info["test_bleu"].append(test_bleu)
            
            enc_in, dec_in, target, enc_pad, dec_pad = get_batch(train_loader, self.tokenizer, self.device)

            out = self(enc_in, dec_in, enc_pad, dec_pad)
            B, T, C = out.shape
            out = out.view(B*T, C)
            target = target.view(-1)
            # Ignore padding tokens in loss
            loss = F.cross_entropy(out, target, ignore_index=self.tokenizer.encode(self.tokenizer.special_tokens["pad"])[0], label_smoothing=0.1)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1) # Clip gradient to 1 for stability
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.lr_schedule.step()

        self.save_checkpoint(checkpoint_dir, b+1)
        pd.DataFrame(self.loss_info).to_csv(f"{checkpoint_dir}log.txt", index=False)

    def pos_enc(self, length):
        if pos_embd_mode == "sin":
            # Sinusoidal positional embedding shown in AIAYN
            pos = torch.arange(length, dtype=torch.float32, device=self.device).unsqueeze(1) # Row
            i = (torch.arange(d_model // 2, device=self.device)).unsqueeze(1).T # Column
            i = 10000 ** -(2 * i / d_model)
            # Number of columns is half of what we need because it will be spread out when creating final result
            ang = pos @ i # Inner product to get inputs to trig functions
            res = torch.zeros((length, d_model), device=self.device)
            res[:,0::2] = torch.sin(ang)
            res[:,1::2] = torch.cos(ang)
            return res
        elif pos_embd_mode == "index":
            return torch.arange(length).to(self.device)
        else:
            raise NotImplementedError(pos_embd_mode)
    
    def save_checkpoint(self, checkpoint_dir, iter):
        chk_dir = f"{checkpoint_dir}{iter}"
        if os.path.isdir(chk_dir):
            return
        os.makedirs(chk_dir, exist_ok=True)

        checkpoint = {
            "model": self.state_dict(),
            "optimizer": self.optim.state_dict(),
            "scheduler": self.lr_schedule.state_dict(),
            "iter": iter,
        }
        torch.save(checkpoint, f"{chk_dir}/checkpoint.pt")
        print(f"Checkpoint saved to {chk_dir}")

    def generate(self, encoder_input):
        """
        Only works for batch size = 1 for now.
        """
        encoder_input = torch.tensor(self.tokenizer.encode(encoder_input), dtype=torch.long, device=self.device).unsqueeze(0)
        context = torch.tensor(self.tokenizer.encode(self.tokenizer.special_tokens["bos"]), dtype=torch.long, device=self.device).unsqueeze(0)
        pad = self.tokenizer.encode(self.tokenizer.special_tokens["pad"])[0]
        for _ in range(max_decoder_len):
            with torch.no_grad():
                out = self(encoder_input, context, (encoder_input != pad).to(self.device), (context != pad).to(self.device))
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
                enc_in, dec_in, target, enc_pad, dec_pad = get_batch(loader, self.tokenizer, self.device)
                with torch.no_grad():
                    out = self(enc_in, dec_in, enc_pad, dec_pad)
                B, T, C = out.shape
                out = out.view(B*T, C)
                target = target.view(-1)
                total_loss += F.cross_entropy(out, target, ignore_index=self.tokenizer.encode(self.tokenizer.special_tokens["pad"])[0], label_smoothing=0.1)
            total_loss /= eval_iters
            losses.append(total_loss)
        self.train()
        return losses

    def estimate_bleu_score(self, train_loader, test_loader, print_examples):
        self.eval()
        bleus = []
        printed_example = False
        for loader in [train_loader, test_loader]:
            total_bleu = 0
            for k in range(eval_iters):
                in_lang, out_lang = next(iter(loader))
                for in_traj, out_traj in zip(in_lang, out_lang):
                    # print(f"In: {in_traj}")
                    # print(f"Out: {out_traj}")
                    translated = self.generate(in_traj).squeeze()[1:-1] # Remove <BOS> and <EOS>
                    # print(f"Translated: {translated.tolist()}")
                    bleu = calculate_bleu(translated.tolist(), self.tokenizer.encode(out_traj))
                    # print(f"BLEU score: {bleu}")
                    total_bleu += bleu
                    if print_examples and not printed_example:
                        print(f"EXAMPLE ({'train' if loader is train_loader else 'test'})")
                        print(f"Encoder input:   {in_traj}")
                        print(f"Expected output: {out_traj}")
                        print(f"Actual output:   {self.tokenizer.decode(translated.tolist())}")
                        printed_example = True
            total_bleu /= eval_iters
            bleus.append(total_bleu)
            printed_example = False
        self.train()
        return bleus
    
    def dump_model_info(self, path, num_batches):
        config = {
            "num_batches":num_batches,
            "total_parameters":sum(p.numel() for p in self.parameters()),
            "d_model": d_model,
            "ff_hidden_size": ff_hidden_size,
            "mha_heads": mha_heads,
            "num_encoder_blocks": num_encoder_blocks,
            "num_decoder_blocks": num_decoder_blocks,
            "max_encoder_len": max_encoder_len,
            "max_decoder_len": max_decoder_len,
            "eval_iters": eval_iters,
            "dropout": dropout,
            "pos_embd_mode": pos_embd_mode,
            "lr_warmup_steps": lr_warmup_steps,
        }
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/model_info.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


def get_batch(loader, tokenizer, device):
    in_lang, out_lang = next(iter(loader))
    # Each (B, T)
    enc_in, enc_pad = process_batch(in_lang, tokenizer, special_tokens=tokenizer.special_tokens, device=device)
    out_lang_processed, dec_pad = process_batch(out_lang, tokenizer, special_tokens=tokenizer.special_tokens, device=device)

    dec_pad = dec_pad[:, :-1]
    dec_in = out_lang_processed[:, :-1]
    target = out_lang_processed[:, 1:]

    enc_in, dec_in, target = enc_in.to(device), dec_in.to(device), target.to(device)

    return enc_in, dec_in, target, enc_pad, dec_pad

def process_batch(batch, tokenizer, special_tokens, device):
    bos = tokenizer.encode(special_tokens["bos"])
    eos = tokenizer.encode(special_tokens["eos"])
    pad = tokenizer.encode(special_tokens["pad"])[0]
    tokenized = [torch.tensor(bos + tokenizer.encode(b) + eos, dtype=torch.long) for b in batch]
    batch = pad_sequence(tokenized, batch_first=True, padding_value=pad)
    pad_mask = (batch != pad).to(device)
    return batch, pad_mask