import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import random


class PairedImageDataset(Dataset):
    
    def __init__(self, 
                 photo_dir: str,
                 sketch_dir: str,
                 size: int = 256,
                 augment: bool = True):
        self.photo_dir = Path(photo_dir)
        self.sketch_dir = Path(sketch_dir)
        self.size = size
        self.augment = augment
        
        self.photo_files = sorted(list(self.photo_dir.glob('*.jpg')) + 
                                  list(self.photo_dir.glob('*.png')))
        self.sketch_files = sorted(list(self.sketch_dir.glob('*.jpg')) + 
                                   list(self.sketch_dir.glob('*.png')))
        
        assert len(self.photo_files) == len(self.sketch_files), \
            f"Mismatch: {len(self.photo_files)} photos vs {len(self.sketch_files)} sketches"
        
        self.photo_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.sketch_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])
    
    def __len__(self) -> int:
        return len(self.photo_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        photo = Image.open(self.photo_files[idx]).convert('RGB')
        sketch = Image.open(self.sketch_files[idx]).convert('L')
        
        if self.augment:
            seed = random.randint(0, 2**32 - 1)
            
            random.seed(seed)
            torch.manual_seed(seed)
            photo = self.augment_transform(photo)
            
            random.seed(seed)
            torch.manual_seed(seed)
            sketch = self.augment_transform(sketch)
        
        photo = self.photo_transform(photo)
        sketch = self.sketch_transform(sketch)
        
        return photo, sketch


class SingleImageDataset(Dataset):
    
    def __init__(self, image_paths: list, size: int = 256):
        self.image_paths = image_paths
        self.size = size
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, str(path)


def get_dataloaders(photo_dir: str,
                   sketch_dir: str,
                   batch_size: int = 8,
                   num_workers: int = 4,
                   train_split: float = 0.9,
                   size: int = 256) -> Tuple[torch.utils.data.DataLoader, 
                                             torch.utils.data.DataLoader]:
    full_dataset = PairedImageDataset(photo_dir, sketch_dir, size, augment=True)
    
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    val_dataset_no_aug = PairedImageDataset(photo_dir, sketch_dir, size, augment=False)
    val_indices = val_dataset.indices
    val_dataset_subset = torch.utils.data.Subset(val_dataset_no_aug, val_indices)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    from config import Config
    
    if Config.PHOTOS_DIR.exists() and Config.SKETCHES_DIR.exists():
        dataset = PairedImageDataset(
            Config.PHOTOS_DIR,
            Config.SKETCHES_DIR,
            size=Config.IMAGE_SIZE
        )
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            photo, sketch = dataset[0]
            print(f"Photo shape: {photo.shape}, range: [{photo.min():.2f}, {photo.max():.2f}]")
            print(f"Sketch shape: {sketch.shape}, range: [{sketch.min():.2f}, {sketch.max():.2f}]")
    else:
        print("Dataset directories not found. Run preprocessing first.")
