import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    
    def __init__(self, in_channels: int = 4, ndf: int = 64):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, photo, sketch):
        x = torch.cat([photo, sketch], dim=1)
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    
    def __init__(self, in_channels: int = 4, ndf: int = 64, num_D: int = 3):
        super().__init__()
        
        self.num_D = num_D
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels, ndf) for _ in range(num_D)
        ])
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, photo, sketch):
        results = []
        
        photo_input = photo
        sketch_input = sketch
        
        for i in range(self.num_D):
            results.append(self.discriminators[i](photo_input, sketch_input))
            if i != self.num_D - 1:
                photo_input = self.downsample(photo_input)
                sketch_input = self.downsample(sketch_input)
        
        return results


def test_discriminator():
    model = PatchDiscriminator(in_channels=4, ndf=64)
    photo = torch.randn(2, 3, 256, 256)
    sketch = torch.randn(2, 1, 256, 256)
    output = model(photo, sketch)
    print(f"Photo shape: {photo.shape}")
    print(f"Sketch shape: {sketch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n--- Testing Multi-Scale Discriminator ---")
    ms_model = MultiScaleDiscriminator(in_channels=4, ndf=64, num_D=3)
    outputs = ms_model(photo, sketch)
    print(f"Number of scales: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Scale {i} output shape: {out.shape}")


if __name__ == "__main__":
    test_discriminator()
