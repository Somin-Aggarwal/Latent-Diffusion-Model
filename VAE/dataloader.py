import torch
from torch.utils.data import Dataset,DataLoader
import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class FaceDataset(Dataset):
    def __init__(self, images_dir):
        super().__init__()
        self.img_dir = images_dir
        self.image_names = sorted(os.listdir(images_dir))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
        random.seed(42)
            
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(f"{self.img_dir}/{image_name}")
        image = self.transforms(image)        
        return image

if __name__=="__main__":
    dataset = FaceDataset(images_dir="../celeb_images/training")
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True,num_workers=8)
    for i,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
        pass
    