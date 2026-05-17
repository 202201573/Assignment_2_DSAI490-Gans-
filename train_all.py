from __future__ import annotations
import os, sys, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_PATH = os.path.join(ROOT, "data", "data.txt")


def train_vae(data_path: str, epochs: int):
    from model.vae.vae_model import train
    import model.vae.vae_model as m
    m.EPOCHS = epochs
    train(data_path, save_dir=os.path.join(ROOT, "model", "vae"))


def train_lstm(data_path: str, epochs: int):
    from model.lstm.lstm_model import train
    import model.lstm.lstm_model as m
    m.EPOCHS = epochs
    train(data_path, save_dir=os.path.join(ROOT, "model", "lstm"))


def train_diffusion(data_path: str, epochs: int):
    from model.diffusion.diffusion_model import train
    import model.diffusion.diffusion_model as m
    m.EPOCHS = epochs
    train(data_path, save_dir=os.path.join(ROOT, "model", "diffusion"))


def train_gan(data_path: str, epochs: int):
    from model.gan.gan_model import train
    import model.gan.gan_model as m
    m.EPOCHS = epochs
    train(data_path, save_dir=os.path.join(ROOT, "model", "gan"))


TRAINERS = {
    "vae":       train_vae,
    "gan":       train_gan,
    "lstm":      train_lstm,
    "diffusion": train_diffusion,
}


def main():
    p = argparse.ArgumentParser(description="Train Dates Generator models")
    p.add_argument("--model",  default="all", choices=["all"] + list(TRAINERS))
    p.add_argument("--data",   default=DATA_PATH)
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()

    targets = list(TRAINERS) if args.model == "all" else [args.model]
    for name in targets:
        print(f"\n{'='*60}")
        print(f"  Training: {name.upper()}")
        print(f"{'='*60}\n")
        TRAINERS[name](args.data, args.epochs)

        # Automatically generate predictions for this model
        print(f"\n[train_all] Generating predictions for {name}...")
        from model.predict import run as run_predict
        out_file = os.path.join(ROOT, "data", f"predictions_{name}.txt")
        example_in = os.path.join(ROOT, "data", "input_1000.txt")
        if os.path.exists(example_in):
            run_predict(example_in, out_file, name)
        else:
            print(f"[train_all] Warning: {example_in} not found, skipping prediction.")

    print("\nAll done!")


if __name__ == "__main__":
    main()
