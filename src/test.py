import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config

class DentalXrayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: List of labels (integers)
            transform: torchvision transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), color='black')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(augment=False):
    """
    Enhanced transforms with stronger augmentation for medical images
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            
            # Geometric transformations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # Increased from 0.3
            transforms.RandomRotation(degrees=20),  # Increased from 15
            transforms.RandomAffine(
                degrees=10,
                translate=(0.15, 0.15),  # Increased from 0.1
                scale=(0.85, 1.15),      # Increased range
                shear=10                 # Added shear
            ),
            
            # Color/Intensity transformations (important for X-rays)
            transforms.ColorJitter(
                brightness=0.3,    # Increased from 0.2
                contrast=0.3,      # Increased from 0.2
                saturation=0.2,    # Added
                hue=0.1           # Added
            ),
            
            # Additional augmentations
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            
            # Normalization
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # Random erasing
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform

def load_dataset(data_dir):
    """
    Load image paths and labels from directory structure
    Automatically detects all class folders in the data directory
    
    Args:
        data_dir: Root directory containing class folders
    
    Returns:
        image_paths: List of image paths
        labels: List of labels (integers)
        class_names: List of class names (sorted alphabetically)
    """
    data_dir = Path(data_dir)
    image_paths = []
    labels = []
    
    # Get all subdirectories (class folders)
    class_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if len(class_folders) == 0:
        raise ValueError(f"No class folders found in {data_dir}")
    
    # Extract class names from folder names
    class_names = [d.name for d in class_folders]
    
    print(f"\n{'='*70}")
    print(f"Found {len(class_names)} classes:")
    print(f"{'='*70}")
    for idx, name in enumerate(class_names):
        print(f"  Class {idx}: {name}")
    print(f"{'='*70}\n")
    
    # Supported image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    # Load images from each class folder
    total_images = 0
    for label, class_folder in enumerate(class_folders):
        class_images = []
        
        # Collect all images with supported extensions
        for ext in image_extensions:
            class_images.extend(list(class_folder.glob(ext)))
        
        # Remove duplicates
        class_images = list(set(class_images))
        
        print(f"Class '{class_folder.name}': {len(class_images)} images")
        
        if len(class_images) == 0:
            print(f"  ⚠️  WARNING: No images found in {class_folder}")
        
        for img_path in class_images:
            image_paths.append(str(img_path))
            labels.append(label)
        
        total_images += len(class_images)
    
    print(f"\nTotal images loaded: {total_images}")
    
    if total_images == 0:
        raise ValueError("No images found in any class folder!")
    
    return image_paths, labels, class_names

def create_dataloaders(data_dir, batch_size=None):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Root directory containing class folders
        batch_size: Batch size (default: Config.BATCH_SIZE)
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    
    print(f"\n{'='*70}")
    print("LOADING DATASET")
    print(f"{'='*70}")
    
    # Load dataset
    image_paths, labels, class_names = load_dataset(data_dir)
    
    # Check if we have enough samples
    min_samples_per_class = 10
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    for label, count in zip(unique_labels, counts):
        if count < min_samples_per_class:
            print(f"⚠️  WARNING: Class '{class_names[label]}' has only {count} images.")
            print(f"   Recommended: at least {min_samples_per_class} images per class")
    
    # Split dataset
    print(f"\n{'='*70}")
    print("SPLITTING DATASET")
    print(f"{'='*70}")
    print(f"Train: {Config.TRAIN_SPLIT*100:.0f}%")
    print(f"Validation: {Config.VAL_SPLIT*100:.0f}%")
    print(f"Test: {Config.TEST_SPLIT*100:.0f}%")
    
    try:
        # First split: train vs (val + test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels,
            test_size=(1 - Config.TRAIN_SPLIT),
            random_state=42,
            stratify=labels  # Maintain class distribution
        )
        
        # Second split: val vs test
        val_size = Config.VAL_SPLIT / (Config.VAL_SPLIT + Config.TEST_SPLIT)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(1 - val_size),
            random_state=42,
            stratify=temp_labels
        )
    except ValueError as e:
        print(f"\n⚠️  Error during split: {e}")
        print("This usually happens when you don't have enough samples per class.")
        print("Falling back to simple random split without stratification...")
        
        # Fallback without stratification
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels,
            test_size=(1 - Config.TRAIN_SPLIT),
            random_state=42
        )
        
        val_size = Config.VAL_SPLIT / (Config.VAL_SPLIT + Config.TEST_SPLIT)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(1 - val_size),
            random_state=42
        )
    
    print(f"\nDataset Split Results:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val:   {len(val_paths)} images")
    print(f"  Test:  {len(test_paths)} images")
    
    # Print class distribution in each split
    print(f"\nClass Distribution:")
    print(f"{'Class':<45} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-"*70)
    
    for idx, class_name in enumerate(class_names):
        train_count = sum(1 for label in train_labels if label == idx)
        val_count = sum(1 for label in val_labels if label == idx)
        test_count = sum(1 for label in test_labels if label == idx)
        
        # Truncate long class names
        display_name = class_name[:42] + "..." if len(class_name) > 45 else class_name
        print(f"{display_name:<45} {train_count:<10} {val_count:<10} {test_count:<10}")
    
    print(f"{'='*70}\n")
    
    # Create datasets
    train_dataset = DentalXrayDataset(
        train_paths, train_labels,
        transform=get_transforms(augment=True)
    )
    
    val_dataset = DentalXrayDataset(
        val_paths, val_labels,
        transform=get_transforms(augment=False)
    )
    
    test_dataset = DentalXrayDataset(
        test_paths, test_labels,
        transform=get_transforms(augment=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Test the dataset loading
    print("Testing dataset loading...")
    
    try:
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            Config.DATA_DIR,
            batch_size=4
        )
        
        print("\n✓ Dataset loading successful!")
        print(f"\nTesting data loading:")
        
        # Get one batch
        images, labels = next(iter(train_loader))
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels: {labels}")
        print(f"  Label classes: {[class_names[l.item()] for l in labels]}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()