import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class GANLoss(nn.Module):
    
    def __init__(self, gan_mode: str = 'lsgan'):
        super().__init__()
        self.gan_mode = gan_mode
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown GAN mode: {gan_mode}")
    
    def __call__(self, prediction, target_is_real):
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        
        return self.loss(prediction, target)


class PerceptualLoss(nn.Module):
    
    def __init__(self, layers: list = None):
        super().__init__()
        
        if layers is None:
            layers = [3, 8, 15, 22]
        
        self.layers = layers
        
        vgg = models.vgg16(pretrained=True).features
        self.vgg = vgg.eval()
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.criterion = nn.L1Loss()
    
    def normalize(self, x):
        x = (x + 1) / 2
        return (x - self.mean) / self.std
    
    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features
    
    def forward(self, generated, target):
        if generated.size(1) == 1:
            generated = generated.repeat(1, 3, 1, 1)
        if target.size(1) == 1:
            target = target.repeat(1, 3, 1, 1)
        
        generated = self.normalize(generated)
        target = self.normalize(target)
        
        gen_features = self.extract_features(generated)
        target_features = self.extract_features(target)
        
        loss = 0
        for gen_feat, target_feat in zip(gen_features, target_features):
            loss += self.criterion(gen_feat, target_feat)
        
        return loss / len(self.layers)


class CombinedLoss(nn.Module):
    
    def __init__(self, 
                 lambda_l1: float = 100.0,
                 lambda_perceptual: float = 1.0,
                 use_perceptual: bool = False,
                 gan_mode: str = 'lsgan'):
        super().__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.use_perceptual = use_perceptual
        
        self.gan_loss = GANLoss(gan_mode)
        self.l1_loss = nn.L1Loss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(self, 
                generated: torch.Tensor,
                target: torch.Tensor,
                d_fake: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        loss_gan = self.gan_loss(d_fake, target_is_real=True)
        
        loss_l1 = self.l1_loss(generated, target) * self.lambda_l1
        
        total_loss = loss_gan + loss_l1
        
        losses = {
            'gan': loss_gan.item(),
            'l1': loss_l1.item()
        }
        
        if self.use_perceptual:
            loss_perceptual = self.perceptual_loss(generated, target) * self.lambda_perceptual
            total_loss += loss_perceptual
            losses['perceptual'] = loss_perceptual.item()
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses


def test_losses():
    print("Testing GANLoss...")
    gan_loss = GANLoss('lsgan')
    pred = torch.randn(2, 1, 30, 30)
    loss_real = gan_loss(pred, target_is_real=True)
    loss_fake = gan_loss(pred, target_is_real=False)
    print(f"Real loss: {loss_real.item():.4f}")
    print(f"Fake loss: {loss_fake.item():.4f}")
    
    print("\nTesting PerceptualLoss...")
    perceptual_loss = PerceptualLoss()
    gen = torch.randn(2, 1, 256, 256)
    target = torch.randn(2, 1, 256, 256)
    loss = perceptual_loss(gen, target)
    print(f"Perceptual loss: {loss.item():.4f}")
    
    print("\nTesting CombinedLoss...")
    combined_loss = CombinedLoss(lambda_l1=100.0, use_perceptual=False)
    d_fake = torch.randn(2, 1, 30, 30)
    total_loss, losses = combined_loss(gen, target, d_fake)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss breakdown: {losses}")


if __name__ == "__main__":
    test_losses()
