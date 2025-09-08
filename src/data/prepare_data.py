import argparse
import cv2
import os
from pathlib import Path
from tqdm import tqdm

def generate_lr_images(hr_dir, output_dir, scale=4):
    """
    Generates low-resolution (LR) images from HR images.
    
    Args:
        hr_dir (str): Path to HR images
        output_dir (str): Output folder for LR images
        scale (int): Reduction factor (e.g., 2, 3, 4)
    """
    os.makedirs(output_dir, exist_ok=True)
    hr_paths = list(Path(hr_dir).glob("*.png"))  # images in .png format

    for hr_path in tqdm(hr_paths, desc=f"↓ Generation LR x{scale}"):
        img = cv2.imread(str(hr_path))
        h, w = img.shape[:2]

        # Reduced size
        new_h, new_w = h // scale, w // scale
        lr_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Output Name
        lr_path = Path(output_dir) / hr_path.name
        cv2.imwrite(str(lr_path), lr_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preparation DIV2K (HR -> LR)")
    parser.add_argument("--data_dir", type=str, default="data/raw/DIV2K",
                        help="Path to downloaded HR data")
    parser.add_argument("--output_dir", type=str, default="data/processed/DIV2K",
                        help="Folder where to store generated LR images")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 3, 4],
                        help="Scale factors to generate (eg: 2 3 4)")

    args = parser.parse_args()

    # For each split (train/valid)
    splits = ["DIV2K_train_HR", "DIV2K_valid_HR"]

    for split in splits:
        hr_split_dir = Path(args.data_dir) / split
        if not hr_split_dir.exists():
            print(f"Directory {hr_split_dir} not found, check that you have run download_data.sh")
            continue

        for scale in args.scales:
            out_dir = Path(args.output_dir) / f"{split.replace('_HR','_LR_x'+str(scale))}"
            generate_lr_images(hr_split_dir, out_dir, scale)
    
    print(f"LR data generated in {args.output_dir}")
