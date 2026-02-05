import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model import UNet
from VAE.model import VAE
from sampling import SamplingClass

device = "cuda" if torch.cuda.is_available() else "cpu"

vae_weights_path = "VAE/celeb_vae/best_model.pt"
unet_weights_path = "ldm1_celeb/best_model.pt"

unet_weights = torch.load(unet_weights_path, map_location=device)

config = unet_weights['training_config']
print("UNET")
print(config)
mcfg = config['model']

# ---- Initialize Model ----
mcfg = config["model"]
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

# ---- Setup Hyperparameters ----
batch_size = 5
steps = config['steps']
image_size = 32
schedule = config.get('schedule', 'cosine')

SamplingObject = SamplingClass(
    model=unet,
    batch_size=batch_size,
    schedule=schedule,
    steps=steps,
    img_ch=mcfg["img_ch"],
    device=device
)

vae_weights = torch.load(vae_weights_path, map_location=device)
vae_config = vae_weights['training_config']
print("VAE")
print(vae_config)
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

scale_factor = 0.9788325875351648

time_steps = [i for i in range(1,1001,1)]
time_steps.append(1000)

x_ts = SamplingObject.Ancestral2()
SamplingObject.visualize_stats(x_ts, sampling_strategy="Ancestral2")
x_ts = SamplingObject.process_latents(vae,x_ts, scale_factor)
SamplingObject.visualize_stats(x_ts, sampling_strategy="Ancestral2")

SamplingObject.visualize(x_ts)