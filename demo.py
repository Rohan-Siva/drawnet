import gradio as gr
import torch
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import Config
from models.diffusion import Unet, DiffusionModel
from torchvision import transforms


class DrawNetDemo:
    
    def __init__(self, checkpoint_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.unet = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=Config.INPUT_CHANNELS
        ).to(self.device)
        
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
        
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    @torch.no_grad()
    def generate(self, image):
        if image is None:
            return None
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.convert('RGB')
        original_size = image.size
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        batch_size = 1
        shape = (batch_size, 1, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        
        img = torch.randn(shape, device=self.device)
        
        for i in reversed(range(0, Config.TIMESTEPS)):
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
        sketch_np = sketch_tensor.squeeze().cpu().numpy()
        sketch_np = (sketch_np * 255).astype(np.uint8)
        
        sketch_pil = Image.fromarray(sketch_np, mode='L')
        sketch_pil = sketch_pil.resize(original_size, Image.LANCZOS)
        
        return np.array(sketch_pil)
    
    def create_interface(self):
        
        css = """
        .container {
            max-width: 1200px;
            margin: auto;
        }
        .title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 0.5em;
        }
        .description {
            text-align: center;
            font-size: 1.2em;
            margin-bottom: 1em;
        }
        """
        
        with gr.Blocks(css=css, title="DrawNet: Photo to Sketch") as demo:
            gr.Markdown(
                """
                <div class="title">🎨 DrawNet: Photo to Sketch</div>
                <div class="description">
                Transform your photos into artistic line drawings using AI (LDM)
                </div>
                """
            )
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Upload Photo",
                        type="pil",
                        height=400
                    )
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear", variant="secondary")
                        generate_btn = gr.Button("Generate Sketch", variant="primary")
                
                with gr.Column():
                    output_image = gr.Image(
                        label="Generated Sketch",
                        type="numpy",
                        height=400
                    )
            
            gr.Markdown(
                """
                ### How to use:
                1. Upload a photo using the left panel
                2. Click "Generate Sketch" to create a line drawing
                3. Download your sketch using the download button
                
                ### Tips for best results:
                - Use photos with clear subjects and good contrast
                - Works well with portraits, landscapes, and objects
                - Higher resolution images may take longer to process
                """
            )
            
            example_dir = Path("examples")
            if example_dir.exists():
                examples = [[str(f)] for f in example_dir.glob("*.jpg")]
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=input_image,
                        outputs=output_image,
                        fn=self.generate,
                        cache_examples=False
                    )
            
            generate_btn.click(
                fn=self.generate,
                inputs=input_image,
                outputs=output_image
            )
            
            clear_btn.click(
                fn=lambda: (None, None),
                inputs=None,
                outputs=[input_image, output_image]
            )
        
        return demo


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DrawNet Gradio Demo (LDM)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--share', action='store_true',
                       help='Create public share link')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to run on')
    args = parser.parse_args()
    
    if args.checkpoint is None:
        checkpoint_dir = Config.CHECKPOINT_DIR
        if (checkpoint_dir / 'latest.pth').exists():
            checkpoint_path = str(checkpoint_dir / 'latest.pth')
        elif (checkpoint_dir / 'final.pth').exists():
            checkpoint_path = str(checkpoint_dir / 'final.pth')
        else:
            checkpoints = list(checkpoint_dir.glob('*.pth'))
            if checkpoints:
                checkpoint_path = str(checkpoints[0])
            else:
                raise ValueError("No checkpoint found. Please train the model first or specify --checkpoint")
    else:
        checkpoint_path = args.checkpoint
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    demo_app = DrawNetDemo(checkpoint_path)
    interface = demo_app.create_interface()
    
    interface.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
