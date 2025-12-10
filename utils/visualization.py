import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional
import torchvision.utils as vutils


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor + 1) / 2


def save_comparison_grid(photos: torch.Tensor,
                         sketches_real: torch.Tensor,
                         sketches_fake: torch.Tensor,
                         save_path: str,
                         nrow: int = 4):
    photos = denormalize(photos)
    sketches_real = denormalize(sketches_real)
    sketches_fake = denormalize(sketches_fake)
    
    sketches_real = sketches_real.repeat(1, 3, 1, 1)
    sketches_fake = sketches_fake.repeat(1, 3, 1, 1)
    
    batch_size = photos.size(0)
    comparison = torch.cat([photos, sketches_real, sketches_fake], dim=0)
    
    grid = vutils.make_grid(comparison, nrow=nrow, padding=2, normalize=False)
    
    vutils.save_image(grid, save_path)


def plot_losses(g_losses: List[float],
               d_losses: List[float],
               save_path: str):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_batch(photos: torch.Tensor,
                   sketches: torch.Tensor,
                   save_path: Optional[str] = None,
                   nrow: int = 4):
    photos = denormalize(photos)
    sketches = denormalize(sketches).repeat(1, 3, 1, 1)
    
    batch_size = photos.size(0)
    combined = torch.stack([photos, sketches], dim=1).view(batch_size * 2, 3, 
                                                           photos.size(2), photos.size(3))
    
    grid = vutils.make_grid(combined, nrow=nrow, padding=2, normalize=False)
    
    if save_path:
        vutils.save_image(grid, save_path)
    
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_plt.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_single_comparison(photo: torch.Tensor,
                          sketch_real: Optional[torch.Tensor],
                          sketch_fake: torch.Tensor,
                          save_path: str):
    photo = denormalize(photo).cpu()
    sketch_fake = denormalize(sketch_fake).cpu()
    
    if sketch_real is not None:
        sketch_real = denormalize(sketch_real).cpu()
        ncols = 3
    else:
        ncols = 2
    
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, 4))
    
    axes[0].imshow(photo.permute(1, 2, 0).numpy())
    axes[0].set_title('Input Photo')
    axes[0].axis('off')
    
    idx = 1
    if sketch_real is not None:
        axes[idx].imshow(sketch_real.squeeze().numpy(), cmap='gray')
        axes[idx].set_title('Real Sketch')
        axes[idx].axis('off')
        idx += 1
    
    axes[idx].imshow(sketch_fake.squeeze().numpy(), cmap='gray')
    axes[idx].set_title('Generated Sketch')
    axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Visualization utilities loaded successfully")
