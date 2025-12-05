import torch
import torch.nn as nn


class UNetDown(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat([x, skip_input], dim=1)
        return x


class UNetGenerator(nn.Module):
    
    def __init__(self, in_channels: int = 3, out_channels: int = 1, ngf: int = 64):
        super().__init__()
        
        self.down1 = UNetDown(in_channels, ngf, normalize=False)
        self.down2 = UNetDown(ngf, ngf * 2)
        self.down3 = UNetDown(ngf * 2, ngf * 4)
        self.down4 = UNetDown(ngf * 4, ngf * 8)
        self.down5 = UNetDown(ngf * 8, ngf * 8)
        self.down6 = UNetDown(ngf * 8, ngf * 8)
        self.down7 = UNetDown(ngf * 8, ngf * 8)
        self.down8 = UNetDown(ngf * 8, ngf * 8, normalize=False)
        
        self.up1 = UNetUp(ngf * 8, ngf * 8, dropout=0.5)
        self.up2 = UNetUp(ngf * 16, ngf * 8, dropout=0.5)
        self.up3 = UNetUp(ngf * 16, ngf * 8, dropout=0.5)
        self.up4 = UNetUp(ngf * 16, ngf * 8)
        self.up5 = UNetUp(ngf * 16, ngf * 4)
        self.up6 = UNetUp(ngf * 8, ngf * 2)
        self.up7 = UNetUp(ngf * 4, ngf)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)


def test_generator():
    model = UNetGenerator(in_channels=3, out_channels=1, ngf=64)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    test_generator()
