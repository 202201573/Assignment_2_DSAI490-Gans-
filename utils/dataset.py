"""
PyTorch Dataset wrappers used by all four models.
"""

from __future__ import annotations
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from utils.tokenizer import DateTokenizer, parse_dataset, VOCAB_SIZE, PAD_ID

_tok = DateTokenizer()


class DateDataset(Dataset):
    """
    Returns (condition_ids, date_ids) tensors.
    condition_ids : (4,)   int64
    date_ids      : (10,)  int64   [SOS d d d d d d d d EOS]
    """

    def __init__(self, data_path: str):
        self.conditions, self.dates = parse_dataset(data_path)

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cond_ids = torch.tensor(_tok.encode_conditions(self.conditions[idx]), dtype=torch.long)
        date_ids = torch.tensor(_tok.encode_date(self.dates[idx]),            dtype=torch.long)
        return cond_ids, date_ids


class DateDatasetFlat(Dataset):
    """
    Flattened version: condition_ids (4,) + date digit ids (8,) as targets.
    Used by GAN and VAE which don't use seq2seq decoding.
    """

    def __init__(self, data_path: str):
        self.conditions, self.dates = parse_dataset(data_path)

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cond_ids  = torch.tensor(_tok.encode_conditions(self.conditions[idx]), dtype=torch.long)
        full_ids  = _tok.encode_date(self.dates[idx])      # [SOS d*8 EOS]
        digit_ids = torch.tensor(full_ids[1:9], dtype=torch.long)   # strip SOS/EOS
        return cond_ids, digit_ids
