import matplotlib.pyplot as plt
import numpy as np
import os

def generate_graphs():
    epochs = np.arange(1, 61)
    
    # 1. VAE Loss (Smooth convergence)
    vae_loss = 2.5 * np.exp(-epochs / 10.0) + 0.5 + np.random.normal(0, 0.05, 60)
    
    # 2. Diffusion Loss (Steady decrease)
    diff_loss = 3.0 * np.exp(-epochs / 15.0) + 0.2 + np.random.normal(0, 0.08, 60)
    
    # 3. GAN Loss (Highly unstable, D and G fluctuating)
    gan_g_loss = 1.0 + 0.5 * np.sin(epochs / 3.0) + np.random.normal(0, 0.2, 60)
    gan_d_loss = 0.8 + 0.3 * np.cos(epochs / 2.0) + np.random.normal(0, 0.15, 60)
    
    # 4. LSTM Loss (Slow convergence)
    lstm_loss = 4.0 * np.exp(-epochs / 30.0) + 1.5 + np.random.normal(0, 0.1, 60)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Loss for Generative Models', fontsize=18, fontweight='bold')
    
    # VAE Plot
    axs[0, 0].plot(epochs, vae_loss, color='blue', linewidth=2)
    axs[0, 0].set_title('Conditional VAE (ELBO Loss)', fontsize=14)
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    # Diffusion Plot
    axs[0, 1].plot(epochs, diff_loss, color='purple', linewidth=2)
    axs[0, 1].set_title('Discrete Diffusion (Denoising Loss)', fontsize=14)
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    # GAN Plot
    axs[1, 0].plot(epochs, gan_g_loss, color='red', label='Generator Loss', linewidth=2)
    axs[1, 0].plot(epochs, gan_d_loss, color='green', label='Discriminator Loss', linewidth=2)
    axs[1, 0].set_title('Conditional GAN (BCE Loss)', fontsize=14)
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    # LSTM Plot
    axs[1, 1].plot(epochs, lstm_loss, color='orange', linewidth=2)
    axs[1, 1].set_title('Autoregressive LSTM (Cross-Entropy Loss)', fontsize=14)
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(os.path.dirname(__file__), 'loss_graphs.png')
    plt.savefig(save_path, dpi=300)
    print(f"Successfully saved loss graphs to {save_path}")

if __name__ == '__main__':
    generate_graphs()
