# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the automated setup:
```bash
chmod +x setup.sh
./setup.sh
```

## 2. Prepare Dataset

Add your photos to `data/raw/`, then:

```bash
python prepare_data.py --process
```

## 3. Train

```bash
# Quick test (10 epochs, ~20 min)
python train.py --epochs 10

# Full training (100 epochs, ~2-3 hours)
python train.py --epochs 100
```

Monitor training:
```bash
tensorboard --logdir runs
```

## 4. Generate Sketches

Single image:
```bash
python inference.py \
  --checkpoint checkpoints/final.pth \
  --input photo.jpg \
  --output sketch.png
```

Batch:
```bash
python inference.py \
  --checkpoint checkpoints/final.pth \
  --input photos/ \
  --output sketches/
```

## 5. Launch Demo

```bash
python demo.py
```

Visit `http://localhost:7860`

## Tips

- **Dataset**: 500+ images recommended (1000+ ideal)
- **GPU**: CUDA recommended for fast training
- **CPU**: Works but slower (~10x)
- **Memory**: 8GB GPU or 16GB RAM minimum

## Troubleshooting

**Import errors**:
```bash
pip install -r requirements.txt
```

**Out of memory**:
```bash
python train.py --batch_size 4
```

**Poor quality**:
- Train longer (more epochs)
- Add more training data
- Adjust edge detection thresholds in `config.py`

For full documentation, see `README.md`
