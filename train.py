import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import time

from config import Config
from models.diffusion import Unet, DiffusionModel
from utils.dataset import get_dataloaders
from utils.visualization import save_comparison_grid, plot_losses


class Trainer:
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        
        config.create_dirs()
        
        # Initialize U-Net
        self.unet = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=config.INPUT_CHANNELS
        ).to(self.device)
        
        # Initialize Diffusion Model
        self.model = DiffusionModel(
            self.unet,
            timesteps=config.TIMESTEPS,
            beta_start=config.BETA_START,
            beta_end=config.BETA_END
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2)
        )
        
        self.writer = SummaryWriter(log_dir=config.PROJECT_ROOT / 'runs')
        
        self.current_epoch = 0
        self.global_step = 0
        self.losses = []
    
    def train_epoch(self, dataloader):
        self.model.train()
        
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, (photos, sketches) in enumerate(pbar):
            
            
            photos = photos.to(self.device)
            sketches = sketches.to(self.device)
            
            
            
            batch_size = sketches.size(0)
            
            # Sample t
            t = torch.randint(0, self.config.TIMESTEPS, (batch_size,), device=self.device).long()
            
            # Noise
            noise = torch.randn_like(sketches)
            
            
            # Calculate x_t
            x_0 = sketches
            noise = torch.randn_like(x_0)
            
            # Get alpha terms
            sqrt_alphas_cumprod_t = self.model.extract(self.model.sqrt_alphas_cumprod, t, x_0.shape)
            sqrt_one_minus_alphas_cumprod_t = self.model.extract(self.model.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
            
            x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
            
            # Concatenate condition
            model_input = torch.cat([x_t, photos], dim=1)
            
            # Predict noise
            noise_pred = self.unet(model_input, t)
            
            # Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            self.losses.append(loss.item())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            if self.global_step % 50 == 0:
                self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
            
            self.global_step += 1
        
        avg_loss = epoch_loss / len(dataloader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        
        photos, sketches_real = next(iter(dataloader))
        photos = photos.to(self.device)
        sketches_real = sketches_real.to(self.device)
        
        batch_size = photos.size(0)
        shape = sketches_real.shape
        
        # Start from pure noise
        img = torch.randn(shape, device=self.device)
        
        for i in tqdm(reversed(range(0, self.config.TIMESTEPS)), desc="Sampling", leave=False):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # Predict noise
            model_input = torch.cat([img, photos], dim=1)
            noise_pred = self.unet(model_input, t)
            
            # Compute previous image (x_{t-1})
            betas_t = self.model.extract(self.model.betas, t, img.shape)
            sqrt_one_minus_alphas_cumprod_t = self.model.extract(self.model.sqrt_one_minus_alphas_cumprod, t, img.shape)
            sqrt_recip_alphas_t = self.model.extract(self.model.sqrt_recip_alphas, t, img.shape)
            
            model_mean = sqrt_recip_alphas_t * (
                img - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
            )
            
            if i > 0:
                posterior_variance_t = self.model.extract(self.model.posterior_variance, t, img.shape)
                noise = torch.randn_like(img)
                img = model_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                img = model_mean
        
        sketches_fake = img
        
        save_path = self.config.OUTPUT_DIR / f'epoch_{self.current_epoch:03d}.png'
        save_comparison_grid(
            photos[:8],
            sketches_real[:8],
            sketches_fake[:8],
            str(save_path),
            nrow=4
        )
        
        print(f"Saved validation images to {save_path}")
    
    def save_checkpoint(self, filename: str = None):
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch:03d}.pth'
        
        checkpoint_path = self.config.CHECKPOINT_DIR / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        latest_path = self.config.CHECKPOINT_DIR / 'latest.pth'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.losses = checkpoint.get('losses', [])
        
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {self.current_epoch})")
    
    def train(self, train_loader, val_loader):
        print("=" * 60)
        print("Starting training (LDM)...")
        self.config.print_config()
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("=" * 60)
        
        start_epoch = self.current_epoch + 1
        
        for epoch in range(start_epoch, self.config.NUM_EPOCHS + 1):
            self.current_epoch = epoch
            
            avg_loss = self.train_epoch(train_loader)
            
            print(f"\nEpoch {epoch}/{self.config.NUM_EPOCHS} - Loss: {avg_loss:.4f}")
            
            if epoch % self.config.VAL_EVERY == 0:
                self.validate(val_loader)
            
            if epoch % self.config.SAVE_EVERY == 0:
                self.save_checkpoint()
            
            if epoch % 5 == 0:
                plot_path = self.config.OUTPUT_DIR / 'losses.png'
                # plot_losses expects g_losses and d_losses. We only have one loss.
                # We can adapt plot_losses or just plot one line.
                # For now, let's just pass the same loss twice or modify plot_losses later.
                # Or just pass empty list for second arg.
                plot_losses(self.losses, [], str(plot_path))
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        
        self.save_checkpoint('final.pth')
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train DrawNet (LDM)')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    args = parser.parse_args()
    
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    
    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders(
        photo_dir=str(Config.PHOTOS_DIR),
        sketch_dir=str(Config.SKETCHES_DIR),
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        size=Config.IMAGE_SIZE
    )
    
    trainer = Trainer(Config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
