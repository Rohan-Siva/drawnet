# DrawNet: Photo to Sketch Translation

Transform photos into artistic line drawings using deep learning. Built with PyTorch and Pix2Pix-style conditional GANs.

![DrawNet Banner](https://img.shields.io/badge/DrawNet-Photo%20to%20Sketch-blue)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

DrawNet is an end-to-end deep learning system that translates photographs into artistic sketches using conditional Generative Adversarial Networks (GANs). The project implements the Pix2Pix architecture with a U-Net generator and PatchGAN discriminator.

### Key Features

- **Pix2Pix Architecture**: U-Net generator with skip connections + PatchGAN discriminator
- **Synthetic Data Generation**: Automatic sketch creation using Canny edge detection
- **Training Pipeline**: Complete training loop with checkpointing and TensorBoard logging
- **Evaluation Metrics**: FID and LPIPS for quantitative assessment
- **Interactive Demo**: Gradio web interface for easy testing
- **Inference Tools**: Single image and batch processing scripts

## Architecture

```
Input Photo (RGB) → U-Net Generator → Generated Sketch (Grayscale)
                          ↓
                   PatchGAN Discriminator
                          ↓
                   Real/Fake Classification
```

### U-Net Generator
- 8 encoder blocks (downsampling with Conv + BatchNorm + LeakyReLU)
- 8 decoder blocks (upsampling with ConvTranspose + BatchNorm + ReLU)
- Skip connections between encoder-decoder pairs
- Input: 256×256 RGB image
- Output: 256×256 grayscale sketch

### PatchGAN Discriminator
- 70×70 receptive field for high-frequency detail
- 5 convolutional layers with increasing channels
- Classifies overlapping patches as real/fake
- Input: Concatenated photo + sketch (4 channels)

### Loss Functions
- **Adversarial Loss**: LSGAN (MSE-based) for stable training
- **L1 Pixel Loss**: Weighted λ=100 for structural preservation
- **Perceptual Loss** (optional): VGG16 features for style matching

## Tech Stack

- **Framework**: PyTorch 2.0+
- **Computer Vision**: OpenCV (edge detection)
- **Visualization**: Matplotlib, TensorBoard
- **Web Interface**: Gradio
- **Evaluation**: pytorch-fid, LPIPS

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Setup

```bash
# Clone the repository
cd /Users/rohansiva/Desktop/ML_Projects/drawnet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Dataset

Add your photos to `data/raw/` directory, then run:

```bash
python prepare_data.py --process
```

This will:
- Resize images to 256×256
- Generate sketch pairs using Canny edge detection
- Save processed photos to `data/photos/`
- Save generated sketches to `data/sketches/`

**Dataset Sources**:
- Your own photos
- [Unsplash](https://unsplash.com) (free high-quality images)
- [Pexels](https://www.pexels.com) (free stock photos)
- [COCO Dataset](https://cocodataset.org) (large-scale dataset)

### 2. Train Model

```bash
# Train with default settings (100 epochs, batch size 8)
python train.py

# Custom training
python train.py --epochs 50 --batch_size 16

# Resume from checkpoint
python train.py --resume checkpoints/latest.pth
```

**Training monitors**:
- Progress bar with loss values
- Validation images saved to `outputs/`
- TensorBoard logs in `runs/`

```bash
# View training progress
tensorboard --logdir runs
```

### 3. Generate Sketches

**Single image**:
```bash
python inference.py \
  --checkpoint checkpoints/final.pth \
  --input path/to/photo.jpg \
  --output sketch.png
```

**Batch processing**:
```bash
python inference.py \
  --checkpoint checkpoints/final.pth \
  --input path/to/photos/ \
  --output path/to/outputs/
```

### 4. Launch Demo

```bash
# Local demo
python demo.py

# Public share link
python demo.py --share

# Custom checkpoint
python demo.py --checkpoint checkpoints/best.pth
```

Access the web interface at `http://localhost:7860`

## Project Structure

```
drawnet/
├── config.py                 # Configuration and hyperparameters
├── prepare_data.py          # Dataset preparation script
├── train.py                 # Training script
├── inference.py             # Inference for single/batch images
├── evaluate.py              # Evaluation metrics (FID, LPIPS)
├── demo.py                  # Gradio web interface
├── requirements.txt         # Python dependencies
│
├── models/
│   ├── generator.py         # U-Net generator
│   ├── discriminator.py     # PatchGAN discriminator
│   └── losses.py            # Loss functions
│
├── utils/
│   ├── dataset.py           # PyTorch dataset loaders
│   ├── preprocessing.py     # Edge detection and augmentation
│   └── visualization.py     # Plotting and image grids
│
├── data/
│   ├── raw/                 # Original photos (user-provided)
│   ├── photos/              # Processed photos
│   └── sketches/            # Generated/real sketches
│
├── checkpoints/             # Saved model weights
├── outputs/                 # Validation images and plots
└── runs/                    # TensorBoard logs
```

## Training Details

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 256×256 | Input/output resolution |
| Batch Size | 8 | Images per batch |
| Learning Rate | 2e-4 | Adam optimizer |
| Betas | (0.5, 0.999) | Adam momentum |
| Epochs | 100 | Training iterations |
| λ L1 | 100.0 | L1 loss weight |
| GAN Mode | LSGAN | MSE-based adversarial loss |

### Training Time
- **Single GPU (RTX 3090)**: ~2-3 hours for 100 epochs (1000 images)
- **CPU**: ~10-15 hours (not recommended)

### Data Augmentation
- Random horizontal flip (p=0.5)
- Random rotation (±10 degrees)
- Applied consistently to photo-sketch pairs

## Evaluation

Run evaluation on test set:

```bash
python evaluate.py --checkpoint checkpoints/final.pth
```

**Metrics**:
- **FID (Fréchet Inception Distance)**: Measures distribution similarity (lower is better)
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual similarity (lower is better)

## Results

### Sample Outputs

*Add your generated sketches here after training*

### Performance

| Metric | Value |
|--------|-------|
| FID | TBD after training |
| LPIPS | TBD after training |
| Training Time | ~2-3 hours (100 epochs) |

## Tips for Best Results

1. **Dataset Quality**:
   - Use high-resolution photos (will be resized to 256×256)
   - Diverse subjects (portraits, landscapes, objects)
   - Good contrast and clear edges
   - Minimum 500-1000 images for good results

2. **Training**:
   - Monitor loss curves - G and D should be balanced
   - Check validation images every 5 epochs
   - If outputs are blurry, increase adversarial weight
   - If training is unstable, reduce learning rate

3. **Inference**:
   - Works best on images similar to training data
   - Higher resolution can be achieved by training at 512×512
   - Post-process sketches with image editing tools if needed

## Advanced Usage

### Custom Edge Detection

Modify `utils/preprocessing.py` to use different edge detection:

```python
# Adjust Canny thresholds
Config.CANNY_LOW_THRESHOLD = 30
Config.CANNY_HIGH_THRESHOLD = 100
```

### Multi-Scale Discriminator

Use `MultiScaleDiscriminator` in `models/discriminator.py` for better quality:

```python
from models.discriminator import MultiScaleDiscriminator
discriminator = MultiScaleDiscriminator(in_channels=4, ndf=64, num_D=3)
```

### Perceptual Loss

Enable perceptual loss in training:

```python
from models.losses import CombinedLoss
loss_fn = CombinedLoss(lambda_l1=100.0, lambda_perceptual=1.0, use_perceptual=True)
```

## Future Improvements

- [ ] CycleGAN for unpaired training
- [ ] Diffusion models for higher quality
- [ ] Style transfer (charcoal, ink, watercolor)
- [ ] Higher resolution (512×512, 1024×1024)
- [ ] Mobile deployment (ONNX, TensorFlow Lite)
- [ ] Real-time video processing

## Troubleshooting

**Out of memory**:
- Reduce batch size: `python train.py --batch_size 4`
- Reduce image size in `config.py`

**Poor quality outputs**:
- Train longer (more epochs)
- Increase dataset size
- Adjust loss weights
- Try different edge detection parameters

**Training instability**:
- Reduce learning rate
- Use gradient clipping
- Check data normalization

## Citation

If you use this code, please cite the original Pix2Pix paper:

```bibtex
@inproceedings{isola2017image,
  title={Image-to-image translation with conditional adversarial networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={CVPR},
  year={2017}
}
```

## License

MIT License - feel free to use for personal or commercial projects.

## Acknowledgments

- Pix2Pix architecture by Isola et al.
- U-Net architecture by Ronneberger et al.
- PatchGAN discriminator design

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.

---

**Built with ❤️ using PyTorch**
