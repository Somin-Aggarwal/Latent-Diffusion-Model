import os
import torch
import logging

def save_checkpoint(filename, model_state_dict, optimizer_state_dict,
                    scheduler_state_dict, epoch, iteration, loss, config, weights_dir):
    os.makedirs(weights_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict,
        'loss': loss,
        'training_config': config
    }
    path = os.path.join(weights_dir, filename)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
    logging.info(f"Saved checkpoint {filename} at epoch {epoch}, iter {iteration}, loss {loss}")

def KLD(mean,logvar):
    return 0.5 * torch.sum(torch.pow(mean, 2) + torch.exp(logvar) - 1.0 - logvar, dim=[1, 2, 3]).mean()