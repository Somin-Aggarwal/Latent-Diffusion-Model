import pickle 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
    }

def cosine_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)

    alphas_cumprod = torch.cos((((x / T) + s) / (1 + s)) * (math.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize so ᾱ₀ = 1

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=1e-20, max=0.9999)

    alphas = 1.0 - betas

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod[1:],  # drop ᾱ₀ to align indexing
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod[1:]),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod[1:]),
    }

class DiffusionDataset(Dataset):
    def __init__(self, file_path:str, mode: str, steps, schedule="linear",*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert schedule=="linear" or schedule=="cosine"
        if mode not in ["train", "test"]:
            raise ValueError(" Mode should be wither \"train\" or \"test\"")
        self.mode = mode

        latent_data = torch.load(file_path)
        self.latents = latent_data["latents"]
        self.file_names = latent_data["names"]
        self.scale_factor = latent_data["scale_factor"]
        
        self.steps = steps
        
        if schedule == "linear":
            data = linear_schedule(steps)
        if schedule == "cosine":
            data = cosine_schedule(steps)

        self.beta_t = data["betas"]
        self.alpha_t = data["alphas"]
        
        self.sqrt_alpha_t_dash = data['sqrt_alphas_cumprod']
        self.sqrt_1_minus_alpha_t_dash = data['sqrt_one_minus_alphas_cumprod']
        
        self.val_multiple = 20
        self.bin_size = self.steps // self.val_multiple
    
    
    
    def forward_process(self, latent_tensor, t):
        noise = torch.randn_like(latent_tensor)
        noisy_image = self.sqrt_alpha_t_dash[t] * latent_tensor + self.sqrt_1_minus_alpha_t_dash[t] * noise
        return noisy_image, noise

    def __len__(self):
        if self.mode == "test":
            return self.latents.shape[0]*self.val_multiple
        return self.latents.shape[0]
    
    def __getitem__(self, idx):
        if self.mode == "test":
            bin_idx = idx % self.val_multiple
            bin_start = bin_idx * self.bin_size
            bin_end = bin_start + self.bin_size
            t = np.random.randint(bin_start, bin_end)
            idx =  idx // self.val_multiple 
        else:
            t = int(np.random.uniform(low=0,high=self.steps))
        
        latent = self.latents[idx]
        latent = latent * self.scale_factor
        
        noisy_latent, noise = self.forward_process(latent,t )
        return noisy_latent, noise, t


def get_stats(data_name, schedule, steps, target_samples, file_path):
    """
    Runs the dataloader and calculates mean/std trajectories 
    for a specific dataset and schedule configuration.
    """
    print(f"\n--- Processing: {data_name.upper()} | Schedule: {schedule} ---")
        
    dataset = DiffusionDataset(file_path=file_path, mode="test", steps=steps, schedule=schedule)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=12)

    step_means = torch.zeros(steps)
    step_stds = torch.zeros(steps)
    step_counts = torch.zeros(steps)

    total_samples_processed = 0
    
    # Create a progress bar
    pbar = tqdm(total=target_samples, desc=f"{data_name}-{schedule}")

    while total_samples_processed < target_samples:
        for batch in loader:
            noisy_imgs, _, t = batch
            
            # Identify current batch size (handling last batch drop-off)
            curr_b_size = noisy_imgs.shape[0]

            # Flatten [B, C, H, W] -> [B, Pixels]
            flat_imgs = noisy_imgs.view(curr_b_size, -1)
            
            batch_means = flat_imgs.mean(dim=1) # [B]
            batch_stds = flat_imgs.std(dim=1)   # [B]
            
            t = t.long().cpu()
            
            # Accumulate
            step_means.index_add_(0, t, batch_means)
            step_stds.index_add_(0, t, batch_stds)
            
            ones = torch.ones_like(t, dtype=torch.float)
            step_counts.index_add_(0, t, ones)

            processed_now = len(t)
            total_samples_processed += processed_now
            pbar.update(processed_now)

            if total_samples_processed >= target_samples:
                break
    
    pbar.close()

    # Avoid div by zero
    step_counts = step_counts.clamp(min=1)
    
    avg_means = (step_means / step_counts).numpy()
    avg_stds = (step_stds / step_counts).numpy()
    
    return avg_means, avg_stds

def main():
    # Global Settings
    steps = 1000
    target_samples = 200000  # Adjusted to 50k as per instructions
    
    # Configurations to test
    datasets = ["train_latents", "val_latents"]
    schedules = ["linear", "cosine"]
    file_paths = ["celeb_latents_train.pt","celeb_latents_val.pt"]
    
    # Setup Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    time_steps = range(steps)
    
    # Loop through all 4 combinations
    for i,data_name in enumerate(datasets):
        for schedule in schedules:
            
            # Get stats for this specific combo
            avg_means, avg_stds = get_stats(data_name, schedule, steps, target_samples, file_path=file_paths[i])
            
            # Create a label for the legend
            label_str = f"{data_name.upper()} - {schedule}"
            
            # Plot on shared axes
            # Axis 0: Means
            axes[0].plot(time_steps, avg_means, label=label_str, alpha=0.8)
            
            # Axis 1: Stds
            axes[1].plot(time_steps, avg_stds, label=label_str, alpha=0.8)

    # --- Final Plot Styling ---
    
    # Mean Plot Settings
    axes[0].set_title(f'Mean Pixel Value Trajectories ({target_samples} samples)')
    axes[0].set_xlabel('Time Step (t)')
    axes[0].set_ylabel('Mean Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Std Plot Settings
    axes[1].set_title(f'Std Dev Trajectories ({target_samples} samples)')
    axes[1].set_xlabel('Time Step (t)')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("diffusion_schedule_comparison.jpg")
    plt.show()
    print("\nComparison graph saved and displayed.")
    
if __name__ == "__main__":
    
    main()
    
    # train_dataset = DiffusionDataset(file_path="celeb_latents_train.pt",
    #                                  steps=1000,
    #                                  schedule="cosine",
    #                                  mode="train")
    
    # train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=12)
    
    # for i,batch in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
    #     pass
