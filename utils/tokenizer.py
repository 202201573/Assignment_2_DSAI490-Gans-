from __future__ import annotations
import re
from typing import List, Tuple, Optional
import datetime

# ── vocabulary ──────────────────────────────────────────────────────────────

DAY_TOKENS   = ["[MON]","[TUE]","[WED]","[THU]","[FRI]","[SAT]","[SUN]"]
MONTH_TOKENS = ["[JAN]","[FEB]","[MAR]","[APR]","[MAY]","[JUN]",
                "[JUL]","[AUG]","[SEP]","[OCT]","[NOV]","[DEC]"]
LEAP_TOKENS  = ["[False]","[True]"]
DECADE_TOKENS = [f"[{180+i}]" for i in range(41)]   # [180]..[220]
DIGIT_TOKENS  = [str(d) for d in range(10)]          # "0".."9"

SPECIAL = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

ALL_TOKENS = SPECIAL + DAY_TOKENS + MONTH_TOKENS + LEAP_TOKENS + DECADE_TOKENS + DIGIT_TOKENS

TOKEN2ID = {t: i for i, t in enumerate(ALL_TOKENS)}
ID2TOKEN = {i: t for t, i in TOKEN2ID.items()}

PAD_ID = TOKEN2ID["<PAD>"]
SOS_ID = TOKEN2ID["<SOS>"]
EOS_ID = TOKEN2ID["<EOS>"]
UNK_ID = TOKEN2ID["<UNK>"]

VOCAB_SIZE = len(ALL_TOKENS)

# ── month/day helpers ────────────────────────────────────────────────────────

MONTH_NAME_TO_NUM = {m: i+1 for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}
DAY_NAME_TO_NUM = {d: i for i, d in enumerate(
    ["MON","TUE","WED","THU","FRI","SAT","SUN"])}   # Monday=0 … Sunday=6


# ── core tokenizer ───────────────────────────────────────────────────────────

class DateTokenizer:
    """
    Encodes condition lines and date strings to integer token lists.

    Input (conditions):  "[WED] [JAN] [False] [180]"
    Output (date str):   "3-12-1962"  →  8 digit tokens (dd mm yyyy)
    """

    def encode_conditions(self, cond_str: str) -> List[int]:
        """
        '[WED] [JAN] [False] [180]' → list of 4 token ids
        """
        tokens = cond_str.strip().split()
        return [TOKEN2ID.get(t, UNK_ID) for t in tokens]

    def encode_date(self, date_str: str) -> List[int]:
        """
        '3-12-1962' → 8 digit token ids  (dd mm yyyy, zero-padded)
        Returns SOS + 8 digits + EOS  (total 10)
        """
        try:
            parts = date_str.strip().split("-")
            dd = int(parts[0])
            mm = int(parts[1])
            yyyy = int(parts[2])
            s = f"{dd:02d}{mm:02d}{yyyy:04d}"
            ids = [SOS_ID] + [TOKEN2ID[c] for c in s] + [EOS_ID]
            return ids
        except Exception:
            return [SOS_ID] + [PAD_ID]*8 + [EOS_ID]

    def decode_date_ids(self, ids: List[int]) -> Optional[str]:
        """
        List of 8 digit token ids → 'd-m-yyyy' string
        Skips SOS/EOS/PAD if present.
        """
        digits = []
        for i in ids:
            tok = ID2TOKEN.get(i, "")
            if tok in DIGIT_TOKENS:
                digits.append(tok)
        if len(digits) < 8:
            return None
        digits = digits[:8]
        dd   = int("".join(digits[0:2]))
        mm   = int("".join(digits[2:4]))
        yyyy = int("".join(digits[4:8]))
        return f"{dd}-{mm}-{yyyy}"

    def conditions_to_vector(self, cond_str: str) -> List[int]:
        """One-hot style numeric vector for GAN/VAE conditioning (4 ints → embed later)."""
        return self.encode_conditions(cond_str)


# ── dataset parser ───────────────────────────────────────────────────────────

def parse_dataset(path: str) -> Tuple[List[str], List[str]]:
    """
    Returns (conditions_list, dates_list).
    conditions_list[i] = '[WED] [JAN] [False] [180]'
    dates_list[i]      = '1-1-1800'
    """
    conditions, dates = [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cond = " ".join(parts[:4])
                date = parts[4]
                conditions.append(cond)
                dates.append(date)
    return conditions, dates


def parse_input_file(path: str) -> List[str]:
    """Parse an input file that has only condition columns (no date)."""
    conditions = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                conditions.append(" ".join(parts[:4]))
    return conditions


# ── validation helpers ───────────────────────────────────────────────────────

def is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def validate_date(date_str: str, cond_str: str) -> bool:
    """Return True if date_str satisfies all conditions in cond_str."""
    try:
        parts = date_str.strip().split("-")
        dd, mm, yyyy = int(parts[0]), int(parts[1]), int(parts[2])
        if not (1800 <= yyyy <= 2200):
            return False
        dt = datetime.date(yyyy, mm, dd)
    except Exception:
        return False

    tokens = cond_str.strip().split()
    day_tok, mon_tok, leap_tok, dec_tok = tokens[0], tokens[1], tokens[2], tokens[3]

    # day of week  (Monday=0 in datetime)
    expected_dow = DAY_NAME_TO_NUM.get(day_tok[1:-1], -1)
    if dt.weekday() != expected_dow:
        return False

    # month
    expected_month = MONTH_NAME_TO_NUM.get(mon_tok[1:-1], -1)
    if dt.month != expected_month:
        return False

    # leap year
    expected_leap = leap_tok == "[True]"
    if is_leap_year(yyyy) != expected_leap:
        return False

    # decade  e.g. [196] → 1960–1969
    dec_num = int(dec_tok[1:-1]) * 10   # [196] → 1960
    if not (dec_num <= yyyy < dec_num + 10):
        return False

    return True


def score_predictions(conditions: List[str], predictions: List[str]) -> float:
    """Fraction of predictions that satisfy all conditions."""
    correct = sum(validate_date(p, c) for c, p in zip(conditions, predictions))
    return correct / len(conditions) if conditions else 0.0
