import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: pytorch-fid not installed. FID metric unavailable.")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not installed. LPIPS metric unavailable.")

from config import Config
from models.diffusion import Unet, DiffusionModel
from utils.dataset import PairedImageDataset
from torch.utils.data import DataLoader


class Evaluator:
    
    def __init__(self, checkpoint_path: str, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Initialize U-Net
        self.unet = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=Config.INPUT_CHANNELS
        ).to(self.device)
        
        # Initialize Diffusion Model
        self.model = DiffusionModel(
            self.unet,
            timesteps=Config.TIMESTEPS,
            beta_start=Config.BETA_START,
            beta_end=Config.BETA_END
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        else:
            self.lpips_fn = None
    
    @torch.no_grad()
    def generate_batch(self, photos):
        batch_size = photos.size(0)
        shape = (batch_size, 1, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        
        img = torch.randn(shape, device=self.device)
        
        for i in reversed(range(0, Config.TIMESTEPS)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            model_input = torch.cat([img, photos], dim=1)
            noise_pred = self.unet(model_input, t)
            
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
        
        return img

    @torch.no_grad()
    def compute_lpips(self, dataloader):
        if not LPIPS_AVAILABLE or self.lpips_fn is None:
            print("LPIPS not available")
            return None
        
        lpips_scores = []
        
        for photos, sketches_real in tqdm(dataloader, desc="Computing LPIPS"):
            photos = photos.to(self.device)
            sketches_real = sketches_real.to(self.device)
            
            sketches_fake = self.generate_batch(photos)
            
            sketches_real_rgb = sketches_real.repeat(1, 3, 1, 1)
            sketches_fake_rgb = sketches_fake.repeat(1, 3, 1, 1)
            
            lpips_val = self.lpips_fn(sketches_fake_rgb, sketches_real_rgb)
            lpips_scores.extend(lpips_val.cpu().numpy())
        
        avg_lpips = np.mean(lpips_scores)
        return avg_lpips
    
    @torch.no_grad()
    def generate_for_fid(self, dataloader, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        idx = 0
        for photos, _ in tqdm(dataloader, desc="Generating for FID"):
            photos = photos.to(self.device)
            sketches = self.generate_batch(photos)
            
            for i in range(sketches.size(0)):
                sketch = sketches[i]
                sketch = (sketch + 1) / 2
                sketch = sketch.squeeze().cpu().numpy()
                sketch = (sketch * 255).astype(np.uint8)
                
                from PIL import Image
                img = Image.fromarray(sketch, mode='L')
                img.save(output_path / f'{idx:05d}.png')
                idx += 1
        
        print(f"Generated {idx} images in {output_dir}")
    
    def compute_fid(self, real_dir: str, fake_dir: str):
        if not FID_AVAILABLE:
            print("FID not available")
            return None
        
        print("Computing FID...")
        fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir],
            batch_size=50,
            device=self.device,
            dims=2048
        )
        
        return fid_value
    
    def evaluate(self, photo_dir: str, sketch_dir: str, batch_size: int = 16):
        print("=" * 60)
        print("Running evaluation...")
        print("=" * 60)
        
        dataset = PairedImageDataset(photo_dir, sketch_dir, 
                                     size=Config.IMAGE_SIZE, augment=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=4)
        
        metrics = {}
        
        if LPIPS_AVAILABLE:
            lpips_score = self.compute_lpips(dataloader)
            metrics['lpips'] = lpips_score
            print(f"LPIPS: {lpips_score:.4f}")
        
        if FID_AVAILABLE:
            fake_dir = Config.OUTPUT_DIR / 'eval_generated'
            self.generate_for_fid(dataloader, str(fake_dir))
            
            fid = self.compute_fid(sketch_dir, str(fake_dir))
            metrics['fid'] = fid
            print(f"FID: {fid:.2f}")
        
        print("=" * 60)
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate DrawNet (LDM)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--photo_dir', type=str, default=None,
                       help='Directory with photos (default: from config)')
    parser.add_argument('--sketch_dir', type=str, default=None,
                       help='Directory with sketches (default: from config)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    args = parser.parse_args()
    
    photo_dir = args.photo_dir or str(Config.PHOTOS_DIR)
    sketch_dir = args.sketch_dir or str(Config.SKETCHES_DIR)
    
    evaluator = Evaluator(args.checkpoint)
    metrics = evaluator.evaluate(photo_dir, sketch_dir, args.batch_size)
    
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        print(f"  {key.upper()}: {value:.4f}")


if __name__ == "__main__":
    main()
