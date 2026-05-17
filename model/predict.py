from __future__ import annotations
import os, sys, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ROOT)

from utils.tokenizer import parse_input_file, validate_date, score_predictions

WEIGHTS = {
    "vae":       os.path.join(ROOT, "vae",       "vae_weights.pt"),
    "lstm":      os.path.join(ROOT, "lstm",      "lstm_weights.pt"),
    "diffusion": os.path.join(ROOT, "diffusion", "diffusion_weights.pt"),
    "gan":       os.path.join(ROOT, "gan",       "gan_weights.pt"),
}


def load_predictor(model_name: str):
    if model_name == "vae":
        from vae.vae_model import predict
    elif model_name == "lstm":
        from lstm.lstm_model import predict
    elif model_name == "diffusion":
        from diffusion.diffusion_model import predict
    elif model_name == "gan":
        from gan.gan_model import predict
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(WEIGHTS)}")
    return predict


def run(input_path: str, output_path: str, model_name: str, limit: int = None):
    conditions = parse_input_file(input_path)
    if not conditions:
        print(f"[ERROR] No conditions parsed from {input_path}")
        sys.exit(1)

    if limit and limit < len(conditions):
        print(f"[predict] Limiting input to first {limit} lines.")
        conditions = conditions[:limit]

    weights_path = WEIGHTS[model_name]
    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights not found at {weights_path}")
        print(f"        Train the {model_name} model first.")
        sys.exit(1)

    print(f"[predict] Using model: {model_name}")
    print(f"[predict] Weights:     {weights_path}")
    print(f"[predict] Conditions:  {len(conditions)}")

    predict_fn = load_predictor(model_name)
    predictions = predict_fn(conditions, weights_path)

    valid_count = sum(validate_date(p, c) for c, p in zip(conditions, predictions))
    print(f"[predict] Valid predictions: {valid_count}/{len(conditions)} "
          f"({100*valid_count/len(conditions):.1f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        for cond, pred in zip(conditions, predictions):
            f.write(f"{cond} {pred}\n")

    print(f"[predict] Output written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Dates Generator — Inference")
    parser.add_argument("-i", "--input",  required=True, help="Path to input conditions file")
    parser.add_argument("-o", "--output", required=True, help="Path to output predictions file")
    parser.add_argument("--model", default="lstm",
                        choices=["vae", "gan", "lstm", "diffusion"],
                        help="Which model to use for prediction (default: lstm)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of input lines to process (default: all)")
    args = parser.parse_args()
    run(args.input, args.output, args.model, args.limit)


if __name__ == "__main__":
    main()
