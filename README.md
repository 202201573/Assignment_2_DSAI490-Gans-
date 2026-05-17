# Dates Generator — Deep Generative Models

## Project Structure
```
dates_generator/
├── data/
│   ├── data.txt              # full dataset (146k entries)
│   └── example_input.txt     # example conditions-only input
├── model/
│   ├── predict.py            # ← unified inference entry point
│   ├── evaluate.py           # scoring & breakdown
│   ├── gan/
│   │   └── gan_model.py      # Model 1: Conditional GAN (in-course)
│   ├── vae/
│   │   └── vae_model.py      # Model 2: Conditional VAE (in-course)
│   ├── transformer/
│   │   └── transformer_model.py  # Model 3: Autoregressive Transformer (out-of-course)
│   └── diffusion/
│       └── diffusion_model.py    # Model 4: Discrete Diffusion D3PM (out-of-course)
├── utils/
│   ├── tokenizer.py          # shared tokenizer, vocabulary, validation
│   └── dataset.py            # PyTorch Dataset wrappers
├── train_all.py              # training entry point
└── environment.yml           # conda env spec
```

## Setup
```bash
conda env create -f environment.yml
conda activate dates_generator
```

## Training
```bash
# Train all 4 models (default 60 epochs each)
python train_all.py --model all

# Train a specific model with custom epochs
python train_all.py --model transformer --epochs 100
python train_all.py --model gan         --epochs 80
python train_all.py --model vae         --epochs 60
python train_all.py --model diffusion   --epochs 60
```

## Inference
```bash
# Run inference (transformer is default and best model)
python model/predict.py -i data/example_input.txt -o predictions.txt

# Use a specific model
python model/predict.py -i data/example_input.txt -o predictions.txt --model gan
python model/predict.py -i data/example_input.txt -o predictions.txt --model vae
python model/predict.py -i data/example_input.txt -o predictions.txt --model diffusion
```

## Evaluation
```bash
python model/evaluate.py --predictions predictions.txt --verbose
```

## Models Overview

### Tokenization
- Input: 4 condition tokens `[DAY] [MONTH] [LEAP] [DECADE]`
- Output: 8 digit tokens representing `DD MM YYYY` (zero-padded, concatenated)
- Example: `3-12-1962` → tokens `["0","3","1","2","1","9","6","2"]`

### Model 1: Conditional GAN (in-course)
- **Generator**: condition embedding + Gaussian noise → 8 digit logits (softmax sampled)
- **Discriminator**: condition embedding + digit embeddings → real/fake score
- **Loss**: Non-saturating GAN loss (softplus formulation)
- **Inference**: Sample noise 50× per condition, keep first valid date

### Model 2: Conditional VAE (in-course)
- **Encoder**: condition + date digits → μ, logσ² in latent space
- **Decoder**: condition + z → 8 digit logits
- **Loss**: Reconstruction (CE) + β·KL divergence
- **Inference**: Sample z from N(0,I), decode with condition

### Model 3: Autoregressive Transformer (out-of-course)
- **Architecture**: Encoder-Decoder Transformer (4 condition tokens as encoder memory)
- **Decoding**: Causal autoregressive generation with teacher forcing at train time
- **Loss**: Cross-entropy per digit position
- **Inference**: Greedy decoding digit-by-digit conditioned on memory

### Model 4: Discrete Denoising Diffusion D3PM (out-of-course)
- **Forward**: Corrupt clean date digits by randomly replacing with uniform noise
- **Reverse**: Denoising Transformer predicts clean x₀ from noisy xₜ and timestep t
- **Schedule**: Linear β schedule from 0.02 to 0.5 over 50 steps
- **Inference**: Start from uniform noise, iteratively denoise via learned x₀ predictions

## Evaluation Metric
Since this is a generation problem (many valid outputs per input), we use **constraint satisfaction rate**: the fraction of generated dates that satisfy all 4 conditions simultaneously (correct weekday, correct month, correct leap-year status, correct decade).

This is more meaningful than accuracy over a fixed answer because the task is generative.
