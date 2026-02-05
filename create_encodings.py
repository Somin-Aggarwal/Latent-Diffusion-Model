import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from VAE.model import VAE  

# --- 1. Custom Dataset to return (Image, Filename) ---
class InferenceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # Get all image files sorted to ensure deterministic order
        self.files = sorted([
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Map to [-1, 1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img_path = os.path.join(self.root_dir, filename)
        
        image = Image.open(img_path)
        image = self.transform(image)
        
        return image, filename

def create_latent_dataset(config, weights_path, output_path):
    device = config["device"]
    
    # --- 2. Load VAE Model ---
    print(f"Loading VAE from {weights_path}...")
    mcfg = config["model"]
    model = VAE(
        img_ch=mcfg["img_ch"],
        base_ch=mcfg["base_ch"],
        ch_mul=mcfg["ch_mul"],
        attn=mcfg["attn"],
        n_resblocks=mcfg["n_resblocks"],
        e_ch=mcfg["e_ch"]
    ).to(device)
    
    # Load weights (Handling potential DDP 'module.' prefix)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    # new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict,strict=True)
    model.eval()

    # --- 3. Setup Data Loader ---
    dataset = InferenceDataset(config["train_dir"])
    loader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=config["num_workers"],
        pin_memory=True
    )

    print(f"Processing {len(dataset)} images...")

    # --- 4. Pre-allocate Memory ---
    # We calculate size: (N, Channels, H, W)
    # 160k images, 8 channels, 16x16 resolution
    n_samples = len(dataset)
    latent_h = 64 // 4 # f=4
    latent_w = 64 // 4
    c_latent = mcfg["e_ch"]
    
    # We store on CPU to save GPU memory for the VAE
    all_latents = torch.zeros((n_samples, c_latent, latent_h, latent_w), dtype=torch.float32)
    all_names = []

    # --- 5. Inference Loop ---
    start_idx = 0
    with torch.no_grad():
        for images, filenames in tqdm(loader, total=len(loader),desc="Encoding"):
            images = images.to(device)
            current_batch_size = images.shape[0]
            
            # Run Encoder
            # Returns: x_rec, latent_encoding, mean, logvar
            # We use 'mean' as the grounded latent for the dataset
            x, mean, logvar, latent_encoding = model(images)
            
            # Move to CPU and store in the big tensor
            end_idx = start_idx + current_batch_size
            all_latents[start_idx:end_idx] = mean.cpu()
            all_names.extend(filenames)
            
            start_idx = end_idx

    # --- 6. Calculate Global Statistics ---
    print("Calculating Global Variance...")
    # Calculate variance across the entire dataset
    global_var = torch.var(all_latents)
    global_std = torch.std(all_latents)
    global_mean = torch.mean(all_latents)
    scale_factor = 1.0 / global_std.item()

    print(f"Global Mean: {global_mean:.6f}")
    print(f"Global Std: {global_std:.6f}")
    print(f"Recommended Scale Factor: {scale_factor:.6f}")

    # --- 7. Save to Single .pt File ---
    data_to_save = {
        "latents": all_latents,    # Tensor [160000, 8, 16, 16]
        "names": all_names,        # List of strings
        "scale_factor": scale_factor,
        "latent_shape": (c_latent, latent_h, latent_w)
    }

    torch.save(data_to_save, output_path)
    print(f"Saved successfully to {output_path} ({os.path.getsize(output_path)/1e9:.2f} GB)")

if __name__ == "__main__":
    # Configuration
    config = {
        "train_dir": "celeb_images/validation", # Path to your images
        "batch_size": 128, # Can go higher since no gradients
        "num_workers": 12,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": { 
            "img_ch": 3, 
            "base_ch": 16, 
            "ch_mul": [2,4,8], 
            "attn": [], 
            "n_resblocks": 2, 
            "e_ch": 8 
        }
    }
    
    # Path to your best VAE checkpoint
    weights_path = "VAE/celeb_vae/best_model.pt"
    
    create_latent_dataset(config, weights_path, output_path="celeb_latents_val.pt")