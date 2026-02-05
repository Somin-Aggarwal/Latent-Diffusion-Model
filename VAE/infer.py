from dataloader import FaceDataset
import torch
from model import VAE
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

def show_results(original_images, model_output):
    # Assuming both are tensors of shape (B, C, H, W)
    images_org = torch.permute(original_images, dims=[0, 2, 3, 1]).cpu().numpy()
    model_output = torch.permute(model_output, dims=[0, 2, 3, 1]).cpu().numpy()

    B = images_org.shape[0]
    fig, axes = plt.subplots(B, 2, figsize=(6, 3 * B))

    if B == 1:
        axes = axes.reshape(1, 2)  # Ensure axes is 2D even for B=1

    for i in range(B):
        axes[i, 0].imshow(images_org[i])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(model_output[i])
        axes[i, 1].set_title("Model Output")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()    
    
    
images_dir = "../celeb_images/testing"
test_dataset = FaceDataset(images_dir)

batch_size = 4
device = "cuda"

test_loader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=8)

weights_path = "celeb_vae/best_model.pt"
weights_dict = torch.load(weights_path)

config = weights_dict["training_config"]
mcfg = config["model"]
model = VAE(
    img_ch=mcfg["img_ch"],
    base_ch=mcfg["base_ch"],
    ch_mul=mcfg["ch_mul"],
    attn=mcfg["attn"],
    n_resblocks=mcfg["n_resblocks"],
    e_ch=mcfg["e_ch"]
).to(config["device"])
model.load_state_dict(state_dict=weights_dict['model_state_dict'],strict=True)

for i,data in enumerate(test_loader):
    with torch.no_grad():
        images =  data
        images = images.to(device)
        
        reconstructed_images, _, _, _ = model(images)
        
        reconstructed_images = (reconstructed_images+1)/2.0
        images = (images+1)/2.0
        
        show_results(images,reconstructed_images)
    
    
    
    
    