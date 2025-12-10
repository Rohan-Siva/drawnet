import torch
from pathlib import Path

class Config:
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    PHOTOS_DIR = DATA_DIR / "photos"
    SKETCHES_DIR = DATA_DIR / "sketches"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    
    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 1
    NGF = 64
    NDF = 64
    
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 2e-4
    BETA1 = 0.5
    BETA2 = 0.999
    
    LAMBDA_L1 = 100.0
    LAMBDA_PERCEPTUAL = 1.0
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    SAVE_EVERY = 10
    VAL_EVERY = 5
    
    CANNY_LOW_THRESHOLD = 50
    CANNY_HIGH_THRESHOLD = 150
    
    FLIP_PROB = 0.5
    ROTATION_DEGREES = 10
    
    TIMESTEPS = 1000
    BETA_START = 0.0001
    BETA_END = 0.02
    
    @classmethod
    def create_dirs(cls):
        for dir_path in [cls.DATA_DIR, cls.RAW_DIR, cls.PHOTOS_DIR, 
                         cls.SKETCHES_DIR, cls.CHECKPOINT_DIR, cls.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        print("=" * 50)
        print("DrawNet Configuration")
        print("=" * 50)
        print(f"Device: {cls.DEVICE}")
        print(f"Image Size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"L1 Weight: {cls.LAMBDA_L1}")
        print("=" * 50)
