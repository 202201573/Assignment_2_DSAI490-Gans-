from __future__ import annotations
import os, sys, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.dataset   import DateDatasetFlat
from utils.tokenizer import (DateTokenizer, validate_date, score_predictions,
                              VOCAB_SIZE, PAD_ID, TOKEN2ID, ID2TOKEN)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

EMBED_DIM   = 64
LATENT_DIM  = 128
HIDDEN_DIM  = 256
NUM_DIGITS  = 8
DIGIT_VOCAB = 10
BATCH_SIZE  = 512
LR_G        = 2e-4
LR_D        = 1e-4
EPOCHS      = 60
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tok = DateTokenizer()


class ConditionEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)
        self.proj  = nn.Sequential(
            nn.Linear(4 * EMBED_DIM, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
        )

    def forward(self, cond_ids: torch.Tensor) -> torch.Tensor:
        e = self.embed(cond_ids).view(cond_ids.size(0), -1)
        return self.proj(e)



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_emb = ConditionEmbedder()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + HIDDEN_DIM, HIDDEN_DIM * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM * 2, NUM_DIGITS * DIGIT_VOCAB),
        )

    def forward(self, cond_ids: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        c = self.cond_emb(cond_ids)
        x = torch.cat([c, z], dim=1)
        logits = self.net(x)
        return logits.view(-1, NUM_DIGITS, DIGIT_VOCAB)



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_emb = ConditionEmbedder()
        self.digit_proj = nn.Linear(DIGIT_VOCAB, EMBED_DIM)
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM + NUM_DIGITS * EMBED_DIM, HIDDEN_DIM * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, cond_ids: torch.Tensor, digit_probs: torch.Tensor) -> torch.Tensor:

        c = self.cond_emb(cond_ids)
        d = self.digit_proj(digit_probs).view(digit_probs.size(0), -1) # (B, 8*EMBED)
        return self.net(torch.cat([c, d], dim=1))



def train(data_path: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    dataset = DateDatasetFlat(data_path)
    n_train = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train],
                                    generator=torch.Generator().manual_seed(SEED))
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    netG = Generator().to(DEVICE)
    netD = Discriminator().to(DEVICE)

    optG = torch.optim.Adam(netG.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=LR_D, betas=(0.5, 0.999))

    for epoch in range(1, EPOCHS + 1):
        netG.train(); netD.train()
        errD_list, errG_list = [], []

        for cond_ids, digit_ids in loader:
            B = cond_ids.size(0)
            cond_ids = cond_ids.to(DEVICE)
            real_d = (digit_ids - TOKEN2ID["0"]).to(DEVICE) # (B, 8) values 0-9
            
            netD.zero_grad()
            
            real_probs = F.one_hot(real_d, num_classes=DIGIT_VOCAB).float()
            output_real = netD(cond_ids, real_probs).view(-1)
            lossD_real = F.binary_cross_entropy_with_logits(output_real, torch.ones_like(output_real))
            lossD_real.backward()

            z = torch.randn(B, LATENT_DIM, device=DEVICE)
            fake_logits = netG(cond_ids, z)
            fake_probs = F.gumbel_softmax(fake_logits, tau=1.0, hard=True)
            output_fake = netD(cond_ids, fake_probs.detach()).view(-1)
            lossD_fake = F.binary_cross_entropy_with_logits(output_fake, torch.zeros_like(output_fake))
            lossD_fake.backward()
            
            optD.step()
            errD_list.append(lossD_real.item() + lossD_fake.item())

            netG.zero_grad()
            output = netD(cond_ids, fake_probs).view(-1)
            lossG = F.binary_cross_entropy_with_logits(output, torch.ones_like(output))
            lossG.backward()
            optG.step()
            errG_list.append(lossG.item())

        netG.eval()
        val_conds, val_preds = [], []
        with torch.no_grad():
            for v_cond_ids, _ in DataLoader(val_ds, batch_size=1024):
                v_cond_ids = v_cond_ids.to(DEVICE)
                z = torch.randn(v_cond_ids.size(0), LATENT_DIM, device=DEVICE)
                logits = netG(v_cond_ids, z)
                pred_d = logits.argmax(-1).cpu().numpy()
                for i, ci in enumerate(v_cond_ids.cpu().numpy()):
                    cond_str = " ".join(ID2TOKEN[x] for x in ci)
                    val_conds.append(cond_str)
                    val_preds.append(_digits_to_date(pred_d[i]))

        acc = score_predictions(val_conds, val_preds)
        print(f"[GAN] Epoch {epoch:3d}/{EPOCHS}  Loss_D={np.mean(errD_list):.4f}  "
              f"Loss_G={np.mean(errG_list):.4f}")

    torch.save(netG.state_dict(), os.path.join(save_dir, "gan_weights.pt"))
    print(f"[GAN] Saved weights to {save_dir}/gan_weights.pt")


def _digits_to_date(digit_arr: np.ndarray) -> str:
    d8 = [str(int(x)) for x in digit_arr]
    dd   = int("".join(d8[0:2]))
    mm   = int("".join(d8[2:4]))
    yyyy = int("".join(d8[4:8]))
    return f"{dd}-{mm}-{yyyy}"


def predict(conditions: list[str], weights_path: str) -> list[str]:
    model = Generator().to(DEVICE)
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
                z = torch.randn(1, LATENT_DIM, device=DEVICE)
                logits = model(cond_ids, z)
                pred_d = logits.argmax(-1).cpu().numpy()[0]
                date_str = _digits_to_date(pred_d)
                if validate_date(date_str, cond_str):
                    best = date_str; break
                if best is None:
                    best = date_str
            results.append(best)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--data",    default="../../data/data.txt")
    p.add_argument("--weights", default="gan_weights.pt")
    p.add_argument("--epochs",  type=int, default=EPOCHS)
    args = p.parse_args()

    if args.train:
        EPOCHS = args.epochs
        train(args.data, save_dir=os.path.dirname(args.weights) or ".")
