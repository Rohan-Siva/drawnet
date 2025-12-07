import torch
from PIL import Image
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from config import Config
from models.diffusion import Unet, DiffusionModel
from utils.visualization import save_single_comparison
from torchvision import transforms


class DrawNetInference:
    
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
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Trained for {checkpoint['epoch']} epochs")
        
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    @torch.no_grad()
    def generate_sketch(self, image_path: str, output_path: str = None):
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Sampling
        batch_size = 1
        shape = (batch_size, 1, Config.IMAGE_SIZE, Config.IMAGE_SIZE) # Assuming 1 channel sketch
        
        img = torch.randn(shape, device=self.device)
        
        for i in tqdm(reversed(range(0, Config.TIMESTEPS)), desc="Generating", leave=False):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            model_input = torch.cat([img, image_tensor], dim=1)
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
        
        sketch_tensor = img
        
        sketch_tensor = (sketch_tensor + 1) / 2
        sketch_tensor = sketch_tensor.squeeze().cpu()
        
        sketch_np = (sketch_tensor.numpy() * 255).astype(np.uint8)
        sketch_pil = Image.fromarray(sketch_np, mode='L')
        
        sketch_pil = sketch_pil.resize(original_size, Image.LANCZOS)
        
        if output_path:
            sketch_pil.save(output_path)
            print(f"Saved sketch to {output_path}")
            
            comparison_path = str(output_path).replace('.png', '_comparison.png')
            save_single_comparison(
                image_tensor.squeeze(),
                None,
                sketch_tensor.unsqueeze(0),
                comparison_path
            )
            print(f"Saved comparison to {comparison_path}")
        
        return sketch_pil
    
    def batch_generate(self, input_dir: str, output_dir: str):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(ext))
        
        print(f"Found {len(image_files)} images")
        
        for img_file in image_files:
            output_file = output_path / f"{img_file.stem}_sketch.png"
            print(f"Processing {img_file.name}...")
            self.generate_sketch(str(img_file), str(output_file))
        
        print(f"\nCompleted! Generated {len(image_files)} sketches in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='DrawNet Inference (LDM)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path or directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    inference = DrawNetInference(args.checkpoint, device=args.device)
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        inference.batch_generate(args.input, args.output)
    else:
        inference.generate_sketch(args.input, args.output)


if __name__ == "__main__":
    main()
