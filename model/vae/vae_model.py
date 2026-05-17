from __future__ import annotations
import os, sys, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.dataset   import DateDatasetFlat
from utils.tokenizer import (DateTokenizer, validate_date, score_predictions,
                              VOCAB_SIZE, PAD_ID, TOKEN2ID, DIGIT_TOKENS,
                              ID2TOKEN)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

EMBED_DIM  = 32
LATENT_DIM = 64
HIDDEN_DIM = 256
NUM_DIGITS = 8
DIGIT_VOCAB = 10
BATCH_SIZE = 512
LR         = 1e-3
EPOCHS     = 50
BETA       = 1.0          # KL weight
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tok = DateTokenizer()


class ConditionEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        self.proj  = nn.Sequential(nn.Linear(4 * EMBED_DIM, HIDDEN_DIM), nn.ReLU())

    def forward(self, cond_ids):
        return self.proj(self.embed(cond_ids).view(cond_ids.size(0), -1))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_emb  = ConditionEmbedder()
        self.digit_emb = nn.Embedding(DIGIT_VOCAB, EMBED_DIM)
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM + NUM_DIGITS * EMBED_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        self.mu_head      = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.logvar_head  = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, cond_ids, digit_ids):
        c = self.cond_emb(cond_ids)
        d = self.digit_emb(digit_ids).view(digit_ids.size(0), -1)
        h = self.net(torch.cat([c, d], dim=1))
        return self.mu_head(h), self.logvar_head(h)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_emb = ConditionEmbedder()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM + LATENT_DIM, HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.LayerNorm(HIDDEN_DIM * 2),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, NUM_DIGITS * DIGIT_VOCAB),
        )

    def forward(self, cond_ids, z):
        c = self.cond_emb(cond_ids)
        x = torch.cat([c, z], dim=1)
        return self.net(x).view(-1, NUM_DIGITS, DIGIT_VOCAB)


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, cond_ids, digit_ids):
        mu, logvar = self.encoder(cond_ids, digit_ids)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(cond_ids, z)
        return logits, mu, logvar

    def generate(self, cond_ids):
        z = torch.randn(cond_ids.size(0), LATENT_DIM, device=cond_ids.device)
        return self.decoder(cond_ids, z)


def vae_loss(logits, targets, mu, logvar, beta=BETA):
    B = logits.size(0)
    recon = F.cross_entropy(logits.view(B * NUM_DIGITS, DIGIT_VOCAB),
                            targets.view(B * NUM_DIGITS))
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon + beta * kl, recon, kl


def train(data_path: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    dataset = DateDatasetFlat(data_path)
    n_train = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train],
                                    generator=torch.Generator().manual_seed(SEED))
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)

    model = CVAE().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for cond_ids, digit_ids in loader:
            cond_ids  = cond_ids.to(DEVICE)
            digit_ids = digit_ids.to(DEVICE)
            real_d    = digit_ids - TOKEN2ID["0"]   # shift to 0-9

            logits, mu, logvar = model(cond_ids, real_d)
            loss, recon, kl = vae_loss(logits, real_d, mu, logvar)

            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()

        model.eval()
        val_conds, val_preds = [], []
        with torch.no_grad():
            for cond_ids, _ in DataLoader(val_ds, batch_size=1024):
                cond_ids = cond_ids.to(DEVICE)
                logits   = model.generate(cond_ids)
                pred_d   = logits.argmax(-1).cpu().numpy()
                for i, ci in enumerate(cond_ids.cpu().numpy()):
                    cond_str = " ".join(ID2TOKEN[x] for x in ci)
                    date_str = _digits_to_date(pred_d[i])
                    val_conds.append(cond_str); val_preds.append(date_str)

        acc = score_predictions(val_conds, val_preds)
        print(f"[VAE] Epoch {epoch:3d}/{EPOCHS}  loss={np.mean(losses):.4f}")

    torch.save(model.state_dict(), os.path.join(save_dir, "vae_weights.pt"))
    print(f"[VAE] Saved to {save_dir}/vae_weights.pt")


def predict(conditions: list[str], weights_path: str) -> list[str]:
    model = CVAE().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    results = []
    with torch.no_grad():
        for cond_str in conditions:
            cond_ids = torch.tensor(
                _tok.encode_conditions(cond_str), dtype=torch.long
            ).unsqueeze(0).to(DEVICE)

            best = None
            for _ in range(50):
                logits   = model.generate(cond_ids)
                pred_d   = logits.argmax(-1).cpu().numpy()[0]
                date_str = _digits_to_date(pred_d)
                if validate_date(date_str, cond_str):
                    best = date_str; break
                if best is None:
                    best = date_str
            results.append(best)
    return results


def _digits_to_date(digit_arr) -> str:
    d8 = [str(int(x)) for x in digit_arr]
    dd   = int("".join(d8[0:2]))
    mm   = int("".join(d8[2:4]))
    yyyy = int("".join(d8[4:8]))
    return f"{dd}-{mm}-{yyyy}"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--data",    default="../data/data.txt")
    p.add_argument("--weights", default="vae_weights.pt")
    p.add_argument("--epochs",  type=int, default=EPOCHS)
    args = p.parse_args()
    if args.train:
        EPOCHS = args.epochs
        train(args.data, save_dir=os.path.dirname(args.weights) or ".")
