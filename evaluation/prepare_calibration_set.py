# prepare_calibration_set.py

import os
import shutil
import random
from pathlib import Path

# --- Configuration ---

# Set the sampling fraction (e.g., 0.2 = 20% of the data)
SAMPLING_FRACTION = 0.2

SOURCE_DIR = Path("data/test")
DEST_DIR = Path("data/calibration")
EXCLUDE_FOLDERS = ['bothcells'] # Case-insensitive

# ---------------------

def create_representative_calibration_set():
    print(f"Creating new calibration set at: {DEST_DIR}")
    print(f"Source: {SOURCE_DIR}")
    print(f"Sampling Fraction: {SAMPLING_FRACTION * 100:.0f}%")
    print(f"Excluding: {EXCLUDE_FOLDERS}\n")

    # Clear out any old calibration data
    if DEST_DIR.exists():
        print(f"Removing old directory: {DEST_DIR}")
        shutil.rmtree(DEST_DIR)
    
    DEST_DIR.mkdir(parents=True)
    
    total_source = 0
    total_copied = 0

    # Get a list of valid class directories from the source
    valid_class_dirs = [
        d for d in SOURCE_DIR.iterdir() 
        if d.is_dir() and d.name.lower() not in EXCLUDE_FOLDERS
    ]

    if not valid_class_dirs:
        print(f"Error: No valid class folders found in {SOURCE_DIR} (or all were excluded).")
        return

    print("--- Summary ---")
    for class_dir in valid_class_dirs:
        class_name = class_dir.name
        
        # Get all image files
        image_files = [
            f for f in class_dir.glob('*') 
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]
        
        if not image_files:
            print(f"  {class_name}: No images found, skipping.")
            continue
        
        # Determine how many images to sample
        num_to_sample = max(1, int(len(image_files) * SAMPLING_FRACTION))
        
        # Randomly select the files
        sampled_files = random.sample(image_files, num_to_sample)
        
        # Create the new destination subfolder
        dest_class_dir = DEST_DIR / class_name
        dest_class_dir.mkdir()

        # Copy the files
        for f in sampled_files:
            shutil.copy(f, dest_class_dir / f.name)
            
        print(f"  {class_name}: Copied {num_to_sample} of {len(image_files)} images.")
        
        total_source += len(image_files)
        total_copied += num_to_sample

    print("\n--- Done ---")
    print(f"Total source images (non-excluded): {total_source}")
    print(f"Total calibration images created: {total_copied}")
    print(f"New calibration set is ready at: {DEST_DIR}")

if __name__ == "__main__":
    create_representative_calibration_set()