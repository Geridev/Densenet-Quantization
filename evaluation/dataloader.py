import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image

class CustomImageFolder(Dataset):
    """
    A custom data loader that skips specified subfolders.
    """
    def __init__(self, root_dir, transform=None, exclude_folders=None):
        self.root_dir = root_dir
        self.transform = transform
        self.exclude_folders = [] if exclude_folders is None else [f.lower() for f in exclude_folders]
        
        self.samples = []  # This will store (image_path, class_index)
        self.class_to_idx = {}
        self.classes = []
        
        self._scan_directory()

    def _scan_directory(self):
        """Scans the directory, builds class map, and finds samples."""
        print(f"Scanning directory: {self.root_dir}")
        print(f"Excluding folders: {self.exclude_folders}")

        valid_class_idx = 0
        for class_name in sorted(os.listdir(self.root_dir)):
            if not os.path.isdir(os.path.join(self.root_dir, class_name)):
                continue
            
            if class_name.lower() in self.exclude_folders:
                print(f"  ... Skipping excluded folder: {class_name}")
                continue
            
            # This is a valid class
            self.class_to_idx[class_name] = valid_class_idx
            self.classes.append(class_name)
            
            class_dir = os.path.join(self.root_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, file_name)
                    self.samples.append((img_path, valid_class_idx))
            
            valid_class_idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning blank image.")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            label = 0 # Return a default label to avoid crash

        if self.transform:
            image = self.transform(image)
            
        return image, label


def create_dataloader(data_dir: str, 
                      batch_size: int = 32, 
                      num_workers: int = 4, 
                      img_size: int = 224,
                      exclude_folders: list = None) -> DataLoader:
    """
    Creates an efficient data loader for the test dataset.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # --- THIS IS THE FIX ---
    # We now ONLY resize and use ToTensor().
    # ToTensor() automatically scales image data from [0, 255] to [0.0, 1.0],
    # which exactly matches your original run_densenet.py logic.
    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    # -----------------------
    
    # Use our new CustomImageFolder
    image_dataset = CustomImageFolder(
        root_dir=data_dir,
        transform=data_transform,
        exclude_folders=exclude_folders
    )
    
    if len(image_dataset) == 0:
        raise ValueError(f"No images found in directory: {data_dir}. "
                         "Check subfolders and exclude_folders list.")

    print(f"Found {len(image_dataset)} images in {len(image_dataset.classes)} (non-excluded) classes.")
    print(f"Class mapping: {image_dataset.class_to_idx}")

    # Create the DataLoader
    loader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for evaluation
        num_workers=num_workers,
        pin_memory=True  # Speeds up HtoD data transfer
    )
    
    return loader