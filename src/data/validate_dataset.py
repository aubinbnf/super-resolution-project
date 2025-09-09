import os
import glob
from PIL import Image
import numpy as np
import json

def validate_dataset(hr_dir, lr_dir, scales=[2, 4, 8]):
    report = {"missing_lr": [], "wrong_size": [], "wrong_mode": [], "stats": {}}
    hr_images = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
    for hr_path in hr_images:
        image_id = os.path.splitext(os.path.basename(hr_path))[0]
        hr = Image.open(hr_path)
        
        # Check that the image is in RGB mode
        if hr.mode != "RGB":
            report["wrong_mode"].append(hr_path)
        
        # Stats
        report["stats"][image_id] = {
            "hr_size": hr.size,
        }
        
        # Checks of the corresponding LRs
        for s in scales:
            base_dir = lr_dir + f"_x{s}"
            lr_path = os.path.join(base_dir, f"{image_id}.png")
            if not os.path.exists(lr_path):
                report["missing_lr"].append(lr_path)
                continue
            lr = Image.open(lr_path)
            
            # Check that the image is in RGB mode
            if lr.mode != "RGB":
                report["wrong_mode"].append(lr_path)
            
            # Dimension checks
            expected_size = (hr.size[0] // s, hr.size[1] // s)
            if lr.size != expected_size:
                report["wrong_size"].append({
                    "hr": hr_path,
                    "lr": lr_path,
                    "expected": expected_size,
                    "found": lr.size
                })
    
    return report

if __name__ == "__main__":
    hr_dir = "data/raw/DIV2K/DIV2K_train_HR"
    lr_dir = "data/processed/DIV2K/DIV2K_train_LR"

    report = validate_dataset(hr_dir, lr_dir)
    
    with open("src/data/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("Validation complete. Report saved in validation_report.json")
