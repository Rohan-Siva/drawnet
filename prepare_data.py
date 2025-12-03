import argparse
from pathlib import Path
import urllib.request
import zipfile
from tqdm import tqdm

from config import Config
from utils.preprocessing import process_dataset


def download_sample_dataset():
    print("For this demo, please add your own images to data/raw/")
    print("You can use any photos - portraits, landscapes, objects, etc.")
    print("\nSuggested sources:")
    print("  - Your own photos")
    print("  - Unsplash (https://unsplash.com)")
    print("  - Pexels (https://www.pexels.com)")
    print("  - COCO dataset (https://cocodataset.org)")
    print("\nOnce you have images in data/raw/, run this script again with --process")


def prepare_dataset(method='canny', size=256):
    Config.create_dirs()
    
    raw_files = list(Config.RAW_DIR.glob('*.jpg')) + \
                list(Config.RAW_DIR.glob('*.png')) + \
                list(Config.RAW_DIR.glob('*.jpeg'))
    
    if not raw_files:
        print(f"\nNo images found in {Config.RAW_DIR}")
        download_sample_dataset()
        return
    
    print(f"\nFound {len(raw_files)} images in {Config.RAW_DIR}")
    print(f"Processing with {method} edge detection...")
    
    count = process_dataset(
        input_dir=Config.RAW_DIR,
        photo_dir=Config.PHOTOS_DIR,
        sketch_dir=Config.SKETCHES_DIR,
        size=size,
        method=method
    )
    
    print(f"\n{'='*60}")
    print(f"Dataset preparation complete!")
    print(f"  - Processed: {count} image pairs")
    print(f"  - Photos: {Config.PHOTOS_DIR}")
    print(f"  - Sketches: {Config.SKETCHES_DIR}")
    print(f"{'='*60}")
    print("\nYou can now start training with:")
    print("  python train.py")


def main():
    parser = argparse.ArgumentParser(description='Prepare DrawNet dataset')
    parser.add_argument('--download', action='store_true',
                       help='Download sample dataset')
    parser.add_argument('--process', action='store_true',
                       help='Process raw images to create paired dataset')
    parser.add_argument('--method', type=str, default='canny',
                       choices=['canny', 'hed'],
                       help='Edge detection method')
    parser.add_argument('--size', type=int, default=256,
                       help='Image size')
    args = parser.parse_args()
    
    if args.download:
        download_sample_dataset()
    elif args.process:
        prepare_dataset(method=args.method, size=args.size)
    else:
        raw_files = list(Config.RAW_DIR.glob('*.jpg')) + \
                    list(Config.RAW_DIR.glob('*.png'))
        if raw_files:
            prepare_dataset(method=args.method, size=args.size)
        else:
            download_sample_dataset()


if __name__ == "__main__":
    main()
