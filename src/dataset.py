"""
Dataset module for Periapical vs Non-Periapical X-ray Classification
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from .config import Config

class DentalXrayDataset(Dataset):
    """Dataset for dental X-rays"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), color='black')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(augment=True):
    """Get image transforms"""
    if augment:
        transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.15, 0.15),
                scale=(0.85, 1.15),
                shear=10
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
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

def organize_periapical_dataset():
    """
    Organize dataset for Periapical vs Non-Periapical classification
    
    Expected structure:
    data/raw/
        ├── periapical/       (put all periapical X-rays here)
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── non_periapical/   (put all non-periapical X-rays here)
            ├── image1.jpg
            ├── image2.jpg
            └── ...
    
    If you have disease folders, it will copy all to 'periapical' folder
    """
    
    print("\n" + "="*80)
    print("ORGANIZING PERIAPICAL DATASET")
    print("="*80)
    
    raw_dir = Config.DATA_DIR
    
    periapical_dir = raw_dir / "periapical"
    non_periapical_dir = raw_dir / "non_periapical"
    
    if periapical_dir.exists() and non_periapical_dir.exists():
        print("\n✓ Dataset already organized!")
        print(f"  Periapical folder: {periapical_dir}")
        print(f"  Non-Periapical folder: {non_periapical_dir}")
        return True
    
    disease_folders = [
        raw_dir / 'Primary Endo with Secondary Perio',
        raw_dir / 'Primary Endodontic Lesion',
        raw_dir / 'Primary Perio with Secondary Endo',
        raw_dir / 'Primary Periodontal Lesion',
        raw_dir / 'True Combined Lesions'
    ]
    
    existing_disease_folders = [f for f in disease_folders if f.exists()]
    
    if existing_disease_folders:
        print("\n✓ Found disease category folders!")
        print("  These will be treated as PERIAPICAL X-rays")
        

        periapical_dir.mkdir(exist_ok=True)
        

        total_copied = 0
        for disease_folder in existing_disease_folders:
            images = list(disease_folder.glob('*.jpg')) + \
                    list(disease_folder.glob('*.png')) + \
                    list(disease_folder.glob('*.jpeg'))
            
            print(f"\n  Copying {len(images)} images from '{disease_folder.name}'...")
            
            for img_path in images:
                dest_path = periapical_dir / f"{disease_folder.name}_{img_path.name}"
                shutil.copy2(img_path, dest_path)
                total_copied += 1
        
        print(f"\n✓ Copied {total_copied} periapical images")

        non_periapical_dir.mkdir(exist_ok=True)
        
        print(f"\n⚠️  IMPORTANT: Please add non-periapical X-ray images to:")
        print(f"    {non_periapical_dir}")
        print("\n    Non-periapical X-rays include:")
        print("      - Bitewing X-rays")
        print("      - Panoramic X-rays")
        print("      - Occlusal X-rays")
        print("      - Any other dental X-rays that are NOT periapical")
        
        return False
    
    print("\n⚠️  Dataset not organized!")
    print("\nPlease create the following structure:")
    print(f"\n{raw_dir}/")
    print("├── periapical/")
    print("│   ├── image1.jpg")
    print("│   ├── image2.jpg")
    print("│   └── ...")
    print("└── non_periapical/")
    print("    ├── image1.jpg")
    print("    ├── image2.jpg")
    print("    └── ...")
    
    return False

def load_dataset(data_dir):
    data_dir = Path(data_dir)
    image_paths = []
    labels = []
    
    # Define classes
    # class_folders = [
    #     data_dir / "non_periapical",
    #     data_dir / "periapical"
    # ]
    class_folders = [
        data_dir / Config.CLASS_NAMES[0],
        data_dir / Config.CLASS_NAMES[1]
    ]
    class_names = ["Non-Periapical", "Periapical"]
    # class_names=Config.CLASS_NAMES
    
    print(f"\n{'='*80}")
    print(f"LOADING PERIAPICAL CLASSIFICATION DATASET")
    print(f"{'='*80}")
    print(f"Classes: {class_names}")
    print(f"{'='*80}")
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    
    for label, (class_folder, class_name) in enumerate(zip(class_folders, class_names)):
        if not class_folder.exists():
            print(f"\n⚠️  Warning: Folder not found: {class_folder}")
            continue
        
        class_images = []
        for ext in image_extensions:
            class_images.extend(list(class_folder.glob(ext)))
        
        class_images = list(set(class_images))
        
        print(f"\n{class_name}: {len(class_images)} images")
        
        if len(class_images) == 0:
            print(f"  ⚠️  WARNING: No images found in {class_folder}")
        
        for img_path in class_images:
            image_paths.append(str(img_path))
            labels.append(label)
    
    print(f"\nTotal images: {len(image_paths)}")
    print(f"{'='*80}\n")
    
    if len(image_paths) == 0:
        raise ValueError("No images found! Please organize your dataset correctly.")
    
    return image_paths, labels, class_names

def create_dataloaders(data_dir, batch_size=None):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Root directory containing periapical/ and non_periapical/ folders
        batch_size: Batch size (default: Config.BATCH_SIZE)
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    

    image_paths, labels, class_names = load_dataset(Config.DATA_DIR)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Class Distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {class_names[label]}: {count} images ({count/len(labels)*100:.1f}%)")
    
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 classes! Please add images to both folders.")
    

    try:
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels,
            test_size=(1 - Config.TRAIN_SPLIT),
            random_state=42,
            stratify=labels
        )
        
        val_size = Config.VAL_SPLIT / (Config.VAL_SPLIT + Config.TEST_SPLIT)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(1 - val_size),
            random_state=42,
            stratify=temp_labels
        )
    except ValueError as e:
        print(f"\n⚠️  Stratified split failed: {e}")
        print("Using random split instead...")
        
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
    
    print(f"\nDataset Split:")
    print(f"  Train: {len(train_paths)} images ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Val:   {len(val_paths)} images ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Test:  {len(test_paths)} images ({len(test_paths)/len(image_paths)*100:.1f}%)")
    

    print(f"\nClass Distribution in Splits:")
    print(f"{'Class':<20} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-"*50)
    
    for idx, class_name in enumerate(class_names):
        train_count = sum(1 for label in train_labels if label == idx)
        val_count = sum(1 for label in val_labels if label == idx)
        test_count = sum(1 for label in test_labels if label == idx)
        print(f"{class_name:<20} {train_count:<10} {val_count:<10} {test_count:<10}")
    
    print(f"{'='*80}\n")

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
    print("Testing Periapical Dataset Module...")
    
    # Organize dataset
    organize_periapical_dataset()
    
    # Test loading
    try:
        print("\nTesting dataloader creation...")
        train_loader, val_loader, test_loader, class_names = create_dataloaders(
            Config.DATA_DIR,
            batch_size=4
        )
        
        print("\n✓ Dataloaders created successfully!")

        images, labels = next(iter(train_loader))
        print(f"\nTest Batch:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels: {labels}")
        print(f"  Label classes: {[class_names[l.item()] for l in labels]}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()