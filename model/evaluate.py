from __future__ import annotations
import argparse
from collections import defaultdict
from utils.tokenizer import validate_date, parse_dataset


def evaluate(predictions_path: str, verbose: bool = False):
    conditions, predictions = parse_dataset(predictions_path)

    total = len(conditions)
    correct = 0
    day_correct  = defaultdict(lambda: [0, 0])
    mon_correct  = defaultdict(lambda: [0, 0])
    leap_correct = defaultdict(lambda: [0, 0])
    dec_correct  = defaultdict(lambda: [0, 0])

    for cond, pred in zip(conditions, predictions):
        tokens = cond.split()
        day_tok, mon_tok, leap_tok, dec_tok = tokens[0], tokens[1], tokens[2], tokens[3]
        ok = int(validate_date(pred, cond))
        correct += ok

        day_correct[day_tok][ok]   += 1;  day_correct[day_tok][1-ok]  # init both
        mon_correct[mon_tok][ok]   += 1;  mon_correct[mon_tok][1-ok]
        leap_correct[leap_tok][ok] += 1;  leap_correct[leap_tok][1-ok]
        dec_correct[dec_tok][ok]   += 1;  dec_correct[dec_tok][1-ok]

    # Overall
    print(f"\n{'='*50}")
    print(f"  Overall accuracy: {correct}/{total} = {100*correct/total:.2f}%")
    print(f"{'='*50}\n")

    if verbose:
        for label, d in [("Day", day_correct), ("Month", mon_correct),
                         ("Leap", leap_correct), ("Decade", dec_correct)]:
            print(f"--- {label} breakdown ---")
            for tok in sorted(d):
                c = d[tok][1]
                t = d[tok][0] + d[tok][1]
                if t > 0:
                    print(f"  {tok:12s}: {c}/{t} ({100*c/t:.1f}%)")
            print()

    failures = [(c, p) for c, p in zip(conditions, predictions) if not validate_date(p, c)]
    if failures and verbose:
        print("--- Sample failures ---")
        for c, p in failures[:10]:
            print(f"  Cond: {c}  →  Pred: {p}")

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Predictions file path")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()
    evaluate(args.predictions, verbose=args.verbose)


if __name__ == "__main__":
    main()
