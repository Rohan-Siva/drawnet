import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple
from config import Config


def generate_sketch_canny(image: np.ndarray, 
                          low_threshold: int = None,
                          high_threshold: int = None) -> np.ndarray:
    if low_threshold is None:
        low_threshold = Config.CANNY_LOW_THRESHOLD
    if high_threshold is None:
        high_threshold = Config.CANNY_HIGH_THRESHOLD
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    sketch = 255 - edges
    
    return sketch


def generate_sketch_hed(image: np.ndarray, model_path: str = None) -> np.ndarray:
    print("HED not implemented, using Canny instead")
    return generate_sketch_canny(image)


def preprocess_image(image_path: Union[str, Path], 
                     size: int = None) -> np.ndarray:
    if size is None:
        size = Config.IMAGE_SIZE
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    
    return image


def process_dataset(input_dir: Union[str, Path],
                    photo_dir: Union[str, Path],
                    sketch_dir: Union[str, Path],
                    size: int = None,
                    method: str = 'canny') -> int:
    if size is None:
        size = Config.IMAGE_SIZE
    
    input_path = Path(input_dir)
    photo_path = Path(photo_dir)
    sketch_path = Path(sketch_dir)
    
    photo_path.mkdir(parents=True, exist_ok=True)
    sketch_path.mkdir(parents=True, exist_ok=True)
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    count = 0
    for ext in extensions:
        for img_file in input_path.glob(ext):
            try:
                image = preprocess_image(img_file, size)
                
                if method == 'canny':
                    sketch = generate_sketch_canny(image)
                elif method == 'hed':
                    sketch = generate_sketch_hed(image)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                photo_output = photo_path / img_file.name
                cv2.imwrite(str(photo_output), image)
                
                sketch_output = sketch_path / img_file.name
                cv2.imwrite(str(sketch_output), sketch)
                
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} images...")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
    
    print(f"Successfully processed {count} images")
    return count


def augment_image_pair(photo: np.ndarray, 
                       sketch: np.ndarray,
                       flip: bool = True,
                       rotate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if flip and np.random.rand() > 0.5:
        photo = cv2.flip(photo, 1)
        sketch = cv2.flip(sketch, 1)
    
    if rotate and np.random.rand() > 0.5:
        angle = np.random.uniform(-Config.ROTATION_DEGREES, Config.ROTATION_DEGREES)
        h, w = photo.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        photo = cv2.warpAffine(photo, M, (w, h))
        sketch = cv2.warpAffine(sketch, M, (w, h))
    
    return photo, sketch


if __name__ == "__main__":
    Config.create_dirs()
    
    if Config.RAW_DIR.exists() and any(Config.RAW_DIR.iterdir()):
        print("Processing dataset...")
        count = process_dataset(
            Config.RAW_DIR,
            Config.PHOTOS_DIR,
            Config.SKETCHES_DIR,
            method='canny'
        )
        print(f"Dataset ready: {count} image pairs")
    else:
        print(f"No images found in {Config.RAW_DIR}")
        print("Add images to data/raw/ and run this script again")
