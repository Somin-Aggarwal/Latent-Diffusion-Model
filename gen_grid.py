import torch
import torchvision
import matplotlib.pyplot as plt
import os
from model import UNet
from VAE.model import VAE
from sampling import SamplingClass

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results", exist_ok=True)

# Paths
vae_weights_path = "VAE/celeb_vae/best_model.pt"
unet_weights_path = "ldm1_celeb/best_model.pt"
scale_factor = 0.9788325875351648

# --- Load Models ---
# 1. Load UNet
unet_weights = torch.load(unet_weights_path, map_location=device)
config = unet_weights['training_config']
mcfg = config['model']

unet = UNet(
    img_ch=mcfg["img_ch"],
    base_ch=mcfg["base_ch"],
    ch_mul=mcfg["ch_mul"],
    attn=mcfg["attn"],
    n_resblocks=mcfg["n_resblocks"],
    tdim=mcfg["tdim"],
    steps=config['steps'],
).to(device)
unet.load_state_dict(unet_weights["model_state_dict"])
unet.eval()

# 2. Load VAE
vae_weights = torch.load(vae_weights_path, map_location=device)
vae_config = vae_weights['training_config']
vae_mcfg = vae_config['model']

vae = VAE(
    img_ch=vae_mcfg["img_ch"],
    base_ch=vae_mcfg["base_ch"],
    ch_mul=vae_mcfg["ch_mul"],
    attn=vae_mcfg["attn"],
    n_resblocks=vae_mcfg["n_resblocks"],
    e_ch=vae_mcfg['e_ch']
).to(device)
vae.load_state_dict(vae_weights["model_state_dict"])
vae.eval()

# --- Helper: VAE Decoding ---
def decode_latents(latents, vae_model, scale_factor):
    """
    Decodes latent z back to image x using the VAE.
    Formula: x = Decoder(z / scale_factor)
    """
    # 1. Rescale latents
    latents = latents / scale_factor
    
    # 2. Decode
    with torch.no_grad():
        x = vae_model.post_quant_conv(latents)
        for layer in vae_model.decoder_blocks:
            x = layer(x)
        x = vae_model.proj_out(x)
    return x

# --- Generation Loop ---
print("Starting 10x10 Grid Generation...")

batch_size = 10
total_rows = 10
all_images = []

sampler = SamplingClass(
    model=unet,
    batch_size=batch_size,
    schedule=config.get('schedule', 'cosine'),
    steps=config['steps'],
    img_ch=mcfg["img_ch"],
    device=device
)

for row in range(total_rows):
    print(f"Generating Row {row + 1}/{total_rows}...")
    
    # 1. Run Sampling (returns list of steps)
    # Note: Ancestral2 is unconditional here as per your snippet
    x_ts = sampler.Ancestral2() 
    
    # 2. Extract Last Element (The final denoised latent)
    final_latent = x_ts[-1]
    
    # 3. Decode with VAE
    decoded_images = decode_latents(final_latent, vae, scale_factor)
    
    # 4. Store on CPU to save GPU memory
    all_images.append(decoded_images.cpu())

# --- Create & Save Grid ---
print("Stitching grid...")
full_batch = torch.cat(all_images, dim=0) # Shape: [100, 3, 32, 32] (or similar)

# 1. Denormalize & Clamp
# Assuming VAE output is [-1, 1], map to [0, 1]
full_batch = (full_batch + 1) / 2
full_batch = torch.clamp(full_batch, 0.0, 1.0)

# 2. Make Grid
grid_img = torchvision.utils.make_grid(full_batch, nrow=10, padding=0)

# 3. Save
save_path = "results/celeb_ldm2_10x10.png"
torchvision.utils.save_image(grid_img, save_path)
print(f"Done! Saved to {save_path}")

# 4. Display
plt.figure(figsize=(10, 10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis('off')
plt.title("CelebA LDM Generation (10x10)")
plt.show()