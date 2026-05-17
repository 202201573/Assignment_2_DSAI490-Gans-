from __future__ import annotations
import os, sys, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.dataset   import DateDataset
from utils.tokenizer import (DateTokenizer, validate_date, score_predictions,
                              VOCAB_SIZE, PAD_ID, SOS_ID, EOS_ID,
                              ID2TOKEN, DIGIT_TOKENS)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3
BATCH_SIZE  = 512
LR          = 1e-3
EPOCHS      = 60
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tok = DateTokenizer()



class ConditionalLSTM(nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD_ID)

        self.encoder = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.enc2dec_h = nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM)
        self.enc2dec_c = nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM)

        self.decoder = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.out_proj = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.drop = nn.Dropout(DROPOUT)


    def encode(self, cond_ids: torch.Tensor):

        emb = self.drop(self.embed(cond_ids))          # (B, 4, EMBED_DIM)
        _, (hn, cn) = self.encoder(emb)                # hn: (2*L, B, H)

   
        hn = hn.view(NUM_LAYERS, 2, hn.size(1), HIDDEN_DIM)
        cn = cn.view(NUM_LAYERS, 2, cn.size(1), HIDDEN_DIM)
        hn = torch.cat([hn[:, 0, :, :], hn[:, 1, :, :]], dim=-1)  # (L, B, 2H)
        cn = torch.cat([cn[:, 0, :, :], cn[:, 1, :, :]], dim=-1)

        h0 = torch.tanh(self.enc2dec_h(hn))           # (L, B, H)
        c0 = torch.tanh(self.enc2dec_c(cn))
        return h0, c0


    def forward(self, cond_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:

        h0, c0 = self.encode(cond_ids)
        tgt_emb = self.drop(self.embed(tgt_ids))       # (B, T, EMBED)
        out, _ = self.decoder(tgt_emb, (h0, c0))      # (B, T, H)
        return self.out_proj(self.drop(out))           # (B, T, V)



def train(data_path: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    dataset = DateDataset(data_path)
    n_train = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(
        dataset, [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(SEED))

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True)

    model = ConditionalLSTM().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, steps_per_epoch=len(loader), epochs=EPOCHS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for cond_ids, date_ids in loader:
            cond_ids = cond_ids.to(DEVICE)   # (B, 4)
            date_ids = date_ids.to(DEVICE)   # (B, 10) 

            tgt_in  = date_ids[:, :-1]       # (B, 9)  
            tgt_out = date_ids[:, 1:]        # (B, 9) 
            logits = model(cond_ids, tgt_in) # (B, 9, V)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                tgt_out.reshape(B * T),
                ignore_index=PAD_ID,
            )
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item())

        model.eval()
        val_conds, val_preds = [], []
        with torch.no_grad():
            for cond_ids, _ in DataLoader(val_ds, batch_size=512):
                cond_ids = cond_ids.to(DEVICE)
                preds = _greedy_decode(model, cond_ids)
                for ci, pred in zip(cond_ids.cpu().numpy(), preds):
                    val_conds.append(" ".join(ID2TOKEN[x] for x in ci))
                    val_preds.append(pred)

        acc = score_predictions(val_conds, val_preds)
        print(f"[LSTM] Epoch {epoch:3d}/{EPOCHS}  "
              f"loss={np.mean(losses):.4f}")

    torch.save(model.state_dict(), os.path.join(save_dir, "lstm_weights.pt"))
    print(f"[LSTM] Saved to {save_dir}/lstm_weights.pt")



def _greedy_decode(model: ConditionalLSTM, cond_ids: torch.Tensor) -> list[str]:
    B = cond_ids.size(0)
    h, c = model.encode(cond_ids)                           

    tgt = torch.full((B, 1), SOS_ID, dtype=torch.long, device=cond_ids.device)
    generated = []

    for _ in range(8):  
        emb = model.drop(model.embed(tgt[:, -1:]))           # (B, 1, EMBED)
        out, (h, c) = model.decoder(emb, (h, c))             # (B, 1, H)
        logits = model.out_proj(model.drop(out))              # (B, 1, V)
        next_tok = logits[:, 0, :].argmax(-1)                 # (B,)
        tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)
        generated.append(next_tok)

    digit_mat = torch.stack(generated, dim=1).cpu().numpy()  # (B, 8)
    results = []
    for row in digit_mat:
        tokens = [ID2TOKEN.get(int(i), "0") for i in row]
        valid  = [t if t in DIGIT_TOKENS else "0" for t in tokens]
        dd   = int("".join(valid[0:2]))
        mm   = int("".join(valid[2:4]))
        yyyy = int("".join(valid[4:8]))
        results.append(f"{dd}-{mm}-{yyyy}")
    return results



def predict(conditions: list[str], weights_path: str) -> list[str]:
    model = ConditionalLSTM().to(DEVICE)
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
                preds    = _greedy_decode(model, cond_ids)
                date_str = preds[0]
                if validate_date(date_str, cond_str):
                    best = date_str; break
                if best is None:
                    best = date_str
            results.append(best)
    return results



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train",   action="store_true")
    p.add_argument("--data",    default="../data/data.txt")
    p.add_argument("--weights", default="lstm_weights.pt")
    p.add_argument("--epochs",  type=int, default=EPOCHS)
    args = p.parse_args()
    if args.train:
        EPOCHS = args.epochs
        train(args.data, save_dir=os.path.dirname(args.weights) or ".")
