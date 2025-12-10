
import sys
from pathlib import Path

def test_imports():
    print("Testing imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"  ✓ torchvision {torchvision.__version__}")
        
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__}")
        
        from PIL import Image
        print(f"  ✓ Pillow")
        
        import gradio
        print(f"  ✓ Gradio {gradio.__version__}")
        
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
        
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_models():
    """Test model initialization."""
    print("\nTesting models...")
    try:
        import torch
        from models.generator import UNetGenerator
        from models.discriminator import PatchDiscriminator
        
        # Test generator
        G = UNetGenerator(in_channels=3, out_channels=1, ngf=64)
        x = torch.randn(1, 3, 256, 256)
        y = G(x)
        assert y.shape == (1, 1, 256, 256), f"Generator output shape mismatch: {y.shape}"
        print(f"  ✓ Generator: {sum(p.numel() for p in G.parameters()):,} parameters")
        
        # Test discriminator
        D = PatchDiscriminator(in_channels=4, ndf=64)
        photo = torch.randn(1, 3, 256, 256)
        sketch = torch.randn(1, 1, 256, 256)
        pred = D(photo, sketch)
        print(f"  ✓ Discriminator: {sum(p.numel() for p in D.parameters()):,} parameters")
        print(f"    Output shape: {pred.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_preprocessing():
    """Test preprocessing utilities."""
    print("\nTesting preprocessing...")
    try:
        import numpy as np
        from utils.preprocessing import generate_sketch_canny
        
        # Create dummy image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        sketch = generate_sketch_canny(image)
        
        assert sketch.shape == (256, 256), f"Sketch shape mismatch: {sketch.shape}"
        print(f"  ✓ Canny edge detection")
        
        return True
    except Exception as e:
        print(f"  ✗ Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    try:
        from config import Config
        
        Config.create_dirs()
        
        assert Config.DATA_DIR.exists(), "Data directory not created"
        assert Config.CHECKPOINT_DIR.exists(), "Checkpoint directory not created"
        
        print(f"  ✓ Configuration loaded")
        print(f"    Device: {Config.DEVICE}")
        print(f"    Image size: {Config.IMAGE_SIZE}")
        print(f"    Batch size: {Config.BATCH_SIZE}")
        
        return True
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    Device: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"  ⚠ CUDA not available (will use CPU)")
        
        return True
    except Exception as e:
        print(f"  ✗ CUDA error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DrawNet Installation Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Models", test_models()))
    results.append(("Preprocessing", test_preprocessing()))
    results.append(("CUDA", test_cuda()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! DrawNet is ready to use.")
        print("\nNext steps:")
        print("  1. Add images to data/raw/")
        print("  2. Run: python prepare_data.py --process")
        print("  3. Run: python train.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTry:")
        print("  pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
