import torch
import torch.nn as nn
from torch.optim import AdamW,lr_scheduler
import os 
from dataloader import FaceDataset
from torch.utils.data import DataLoader
from model import VAE
import argparse
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from utils import save_checkpoint, KLD
import random
import numpy as np
import lpips

def train(config, resume, weights_path):
    os.makedirs(config["logging"]["weights_dir"], exist_ok=True)

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Dataset
    train_dataset = FaceDataset(images_dir=config["train_dir"])
    val_dataset = FaceDataset(images_dir=config["val_dir"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle"],
        num_workers=config["num_workers"],
        pin_memory=config["device"].startswith("cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["device"].startswith("cuda"),
    )

    mcfg = config["model"]
    model = VAE(
        img_ch=mcfg["img_ch"],
        base_ch=mcfg["base_ch"],
        ch_mul=mcfg["ch_mul"],
        attn=mcfg["attn"],
        n_resblocks=mcfg["n_resblocks"],
        e_ch=mcfg["e_ch"]
    ).to(config["device"])

    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    
    mse = nn.MSELoss()
    perceptual_criterion = lpips.LPIPS(net='vgg').to(config["device"])

    epochs = config["training"]["epochs"]
    warmup_epochs = min(config["training"]["warmup_epochs"], epochs - 1)

    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_epochs
    )

    cosine_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=config["training"]["eta_min"],
    )
    
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    start_epoch = 0
    if resume:
        weights_dict = torch.load(weights_path, map_location=config['device'])
        model.load_state_dict(weights_dict['model_state_dict'])
        optimizer.load_state_dict(weights_dict['optimizer_state_dict'])
        scheduler.load_state_dict(weights_dict['scheduler_state_dict'])
        start_epoch = weights_dict['epoch'] + 1

    best_val_loss = float("inf")

    for epoch in tqdm(range(start_epoch,epochs), desc="Epochs"):
        model.train()
        epoch_loss = 0.0
        
        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        for i, images in enumerate(train_iterator):
            images = images.to(config["device"])            
            
            optimizer.zero_grad()
            recon_images, mean, logvar, encodings = model(images)
            
            mse_loss = mse(recon_images, images)
            p_loss = perceptual_criterion(images, recon_images).mean()     
            kld = 1e-6 * KLD(mean,logvar)       
            
            loss = mse_loss + p_loss + kld
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            current_lr = optimizer.param_groups[0]['lr']
            train_iterator.set_postfix({
                    "mse": f"{mse_loss.item():.4f}",
                    "p_loss": f"{p_loss.item():.4f}",
                    "kld": f"{kld.item():.4f}",
                    "epoch_avg": f"{(epoch_loss / (i + 1)):.4f}",
                    "lr" : current_lr
                })

        scheduler.step()

        # Validation
        if (epoch + 1) % config["logging"]["validate_every_epoch"] == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images in val_loader:
                    
                    images = images.to(config["device"])
                    
                    recon_images, mean, logvar, encodings = model(images)
                    
                    mse_loss = mse(recon_images, images)
                    p_loss = perceptual_criterion(images, recon_images).mean()     
                    kld = 1e-6 * KLD(mean,logvar)       
                    
                    loss = mse_loss + p_loss + kld
                    
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    "best_model.pt",
                    model.state_dict(),
                    optimizer.state_dict(),
                    scheduler.state_dict(),
                    epoch,
                    i,
                    val_loss,
                    config,
                    config["logging"]["weights_dir"],
                )
                print(f"Val loss : {val_loss} | BEST MODEL")
            else:
                print(f"Val loss : {val_loss}")

        # Epoch checkpoint
        if (epoch + 1) % config["logging"]["save_every_n_epochs"] == 0:
            save_checkpoint(
                f"epoch{epoch+1}.pt",
                model.state_dict(),
                optimizer.state_dict(),
                scheduler.state_dict(),
                epoch,
                i,
                epoch_loss / len(train_loader),
                config,
                config["logging"]["weights_dir"],
            )
           

if __name__ == "__main__":
    
    resume = False
    weights_path = None
    
    if resume:
        config = torch.load(weights_path)['training_config']
    else:
        config = {
            "train_dir": "../celeb_images/training",
            "val_dir": "../celeb_images/validation",
            "num_workers": 12,

            "model": {
                "img_ch" : 3,
                "base_ch": 16,
                "ch_mul": [2,4,8],
                "attn": [],
                "n_resblocks": 2,
                "e_ch" : 8
            },

            "training": {
                "learning_rate": 1e-3,
                "epochs": 200,
                "batch_size": 16,
                "shuffle": True,
                "warmup_epochs": 0,
                "eta_min": 1e-4,
            },

            "logging": {
                "save_every_n_epochs": 50,
                "validate_every_epoch": 25,
                "weights_dir": "celeb_vae",
            },

            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed": 42,
        }
    
    print(config)
    train(config, resume, weights_path)