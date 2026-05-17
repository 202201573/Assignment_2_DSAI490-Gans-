from __future__ import annotations
import os, sys, random, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.dataset   import DateDatasetFlat
from utils.tokenizer import (DateTokenizer, validate_date, score_predictions,
                              VOCAB_SIZE, PAD_ID, TOKEN2ID, ID2TOKEN, DIGIT_TOKENS)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

T_STEPS    = 50          # diffusion timesteps
EMBED_DIM  = 64
HIDDEN_DIM = 256
NHEAD      = 4
NUM_LAYERS = 3
NUM_DIGITS = 8
DIGIT_VOCAB = 10
BATCH_SIZE = 512
LR         = 3e-4
EPOCHS     = 60
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tok = DateTokenizer()

BETAS = torch.linspace(0.02, 0.5, T_STEPS)
ALPHAS = 1.0 - BETAS
ALPHA_BAR = torch.cumprod(ALPHAS, dim=0)   # shape (T,)



def q_sample(x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

    alpha_bar_t = ALPHA_BAR.to(x0.device)[t]  # (B,)
    keep_mask = torch.bernoulli(alpha_bar_t.unsqueeze(1).expand_as(x0.float())).bool()
    random_digits = torch.randint(0, DIGIT_VOCAB, x0.shape, device=x0.device)
    return torch.where(keep_mask, x0, random_digits)



class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args  = t[:, None].float() * freqs[None]
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


class ConditionEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        self.proj  = nn.Sequential(nn.Linear(4 * EMBED_DIM, HIDDEN_DIM), nn.SiLU())

    def forward(self, cond_ids):
        return self.proj(self.embed(cond_ids).view(cond_ids.size(0), -1))


class DenoisingNetwork(nn.Module):


    def __init__(self):
        super().__init__()
        self.digit_emb  = nn.Embedding(DIGIT_VOCAB, EMBED_DIM)
        self.pos_emb    = nn.Embedding(NUM_DIGITS, EMBED_DIM)
        self.time_emb   = TimestepEmbedding(HIDDEN_DIM)
        self.cond_emb   = ConditionEmbedder()

        self.input_proj = nn.Linear(EMBED_DIM, HIDDEN_DIM)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, nhead=NHEAD, dim_feedforward=HIDDEN_DIM * 2,
            dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=NUM_LAYERS)

        self.out_proj = nn.Linear(HIDDEN_DIM, DIGIT_VOCAB)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, cond_ids: torch.Tensor) -> torch.Tensor:

        B, L = xt.shape

        pos = torch.arange(L, device=xt.device).unsqueeze(0).expand(B, -1)
        x   = self.digit_emb(xt) + self.pos_emb(pos)   # (B, 8, EMBED_DIM)
        x   = self.input_proj(x)                         # (B, 8, HIDDEN_DIM)

        t_emb = self.time_emb(t).unsqueeze(1)           # (B, 1, HIDDEN_DIM)
        c_emb = self.cond_emb(cond_ids).unsqueeze(1)    # (B, 1, HIDDEN_DIM)
        x = x + t_emb + c_emb

        x = self.transformer(x)                          # (B, 8, HIDDEN_DIM)
        return self.out_proj(x)                          # (B, 8, DIGIT_VOCAB)



def train(data_path: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    dataset = DateDatasetFlat(data_path)
    n_train = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train],
                                    generator=torch.Generator().manual_seed(SEED))
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)

    model = DenoisingNetwork().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for cond_ids, digit_ids in loader:
            cond_ids  = cond_ids.to(DEVICE)
            digit_ids = digit_ids.to(DEVICE)
            x0 = digit_ids - TOKEN2ID["0"]   # (B, 8) values 0-9

            B = x0.size(0)
            t = torch.randint(0, T_STEPS, (B,), device=DEVICE)
            xt = q_sample(x0, t)

            logits = model(xt, t, cond_ids)   # (B, 8, 10)
            loss = F.cross_entropy(
                logits.view(B * NUM_DIGITS, DIGIT_VOCAB),
                x0.view(B * NUM_DIGITS)
            )
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        model.eval()
        val_conds, val_preds = [], []
        with torch.no_grad():
            for cond_ids, _ in DataLoader(val_ds, batch_size=512):
                cond_ids = cond_ids.to(DEVICE)
                preds = _ddpm_sample(model, cond_ids)
                for i, ci in enumerate(cond_ids.cpu().numpy()):
                    cond_str = " ".join(ID2TOKEN[x] for x in ci)
                    val_conds.append(cond_str); val_preds.append(preds[i])

        acc = score_predictions(val_conds, val_preds)
        print(f"[DIF] Epoch {epoch:3d}/{EPOCHS}  loss={np.mean(losses):.4f}")

    torch.save(model.state_dict(), os.path.join(save_dir, "diffusion_weights.pt"))
    print(f"[DIF] Saved to {save_dir}/diffusion_weights.pt")


def _ddpm_sample(model: DenoisingNetwork, cond_ids: torch.Tensor) -> list[str]:

    B = cond_ids.size(0)
    xt = torch.randint(0, DIGIT_VOCAB, (B, NUM_DIGITS), device=cond_ids.device)

    for step in reversed(range(T_STEPS)):
        t = torch.full((B,), step, dtype=torch.long, device=cond_ids.device)
        logits = model(xt, t, cond_ids)      # (B, 8, 10)
        x0_pred = logits.argmax(-1)          # (B, 8)

        if step > 0:
            t_prev = t - 1
            xt = q_sample(x0_pred, t_prev)
        else:
            xt = x0_pred

    results = []
    for row in xt.cpu().numpy():
        d8   = [str(int(x)) for x in row]
        dd   = int("".join(d8[0:2]))
        mm   = int("".join(d8[2:4]))
        yyyy = int("".join(d8[4:8]))
        results.append(f"{dd}-{mm}-{yyyy}")
    return results


def predict(conditions: list[str], weights_path: str) -> list[str]:
    model = DenoisingNetwork().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    results = []
    with torch.no_grad():
        for cond_str in conditions:
            cond_ids = torch.tensor(
                _tok.encode_conditions(cond_str), dtype=torch.long
            ).unsqueeze(0).to(DEVICE)

            best = None
            for _ in range(20):
                preds = _ddpm_sample(model, cond_ids)
                date_str = preds[0]
                if validate_date(date_str, cond_str):
                    best = date_str; break
                if best is None:
                    best = date_str
            results.append(best)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--data",    default="../data/data.txt")
    p.add_argument("--weights", default="diffusion_weights.pt")
    p.add_argument("--epochs",  type=int, default=EPOCHS)
    args = p.parse_args()
    if args.train:
        EPOCHS = args.epochs
        train(args.data, save_dir=os.path.dirname(args.weights) or ".")
