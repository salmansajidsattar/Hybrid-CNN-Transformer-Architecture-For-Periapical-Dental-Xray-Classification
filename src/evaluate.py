import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
from PIL import Image
import torchvision.transforms as transforms


sys.path.append(str(Path(__file__).parent))

from config import Config
from dataset import create_dataloaders
from model import create_model
from utils import (
    load_checkpoint, plot_confusion_matrix,
    calculate_metrics, print_metrics, save_metrics,
    plot_sample_predictions
)

def load_test_images_from_folder(test_folder, transform=None):
    """
    Load test images from folder structure: test/class_name/images
    Specifically handles periapical and non_periapical folders
    
    Args:
        test_folder: Path to test folder
        transform: Image transformations to apply
        
    Returns:
        images: List of transformed images
        labels: List of corresponding labels
        class_names: List of class names
    """
    test_path = Path(test_folder)
    
    if not test_path.exists():
        raise ValueError(f"Test folder not found: {test_path}")

    class_folders = sorted([d for d in test_path.iterdir() if d.is_dir()])
    
    if len(class_folders) == 0:
        raise ValueError(f"No class folders found in {test_path}")
    
    class_names = [d.name for d in class_folders]
    
    print(f"\nFound {len(class_names)} classes:")
    for idx, name in enumerate(class_names):
        print(f"  Class {idx}: {name}")
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    images = []
    labels = []
    image_paths = []
    
    for class_idx, class_folder in enumerate(class_folders):
        class_name = class_folder.name
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(class_folder.glob(ext)))
        
        print(f"\nLoading class '{class_name}' (label={class_idx})...")
        print(f"  Found {len(image_files)} images")
        
        loaded_count = 0
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                
                images.append(img_tensor)
                labels.append(class_idx)
                image_paths.append(img_path)
                loaded_count += 1
            except Exception as e:
                print(f"  ⚠ Error loading {img_path.name}: {e}")
        
        print(f"  Successfully loaded: {loaded_count} images")
    
    print(f"\nTotal images loaded: {len(images)}")
    
    return images, labels, class_names, image_paths

def evaluate_model(model, images, labels, device, batch_size=32):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    
    print("\nEvaluating model...")
    
    num_images = len(images)
    num_batches = (num_images + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc='Evaluation'):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_images)
            
            batch_images = torch.stack(images[start_idx:end_idx]).to(device)
            batch_labels = labels[start_idx:end_idx]
            
            
            outputs = model(batch_images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(batch_labels)
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    return all_labels, all_predictions, all_probs

def calculate_detailed_metrics(y_true, y_pred, class_names):
    accuracy = accuracy_score(y_true, y_pred)
    
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    

    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm.tolist(),
        'per_class': {}
    }
    
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i])
        }
    
    return metrics

def print_detailed_metrics(metrics, class_names):
    """Print metrics in a formatted way"""
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    print(f"Accuracy:           {metrics['accuracy']*100:6.2f}%")
    print(f"\nPrecision (Macro):  {metrics['precision_macro']*100:6.2f}%")
    print(f"Precision (Weighted): {metrics['precision_weighted']*100:6.2f}%")
    print(f"\nRecall (Macro):     {metrics['recall_macro']*100:6.2f}%")
    print(f"Recall (Weighted):  {metrics['recall_weighted']*100:6.2f}%")
    print(f"\nF1 Score (Macro):   {metrics['f1_macro']*100:6.2f}%")
    print(f"F1 Score (Weighted): {metrics['f1_weighted']*100:6.2f}%")
    
    print("\n" + "="*80)
    print("PER-CLASS METRICS")
    print("="*80)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    
    for class_name in class_names:
        class_metrics = metrics['per_class'][class_name]
        print(f"{class_name:<20} "
              f"{class_metrics['precision']*100:>6.2f}%      "
              f"{class_metrics['recall']*100:>6.2f}%      "
              f"{class_metrics['f1_score']*100:>6.2f}%")

def main():
    """Main evaluation function"""
    
    print("\n" + "="*80)
    print("DENTAL X-RAY CLASSIFICATION - EVALUATION")
    print("="*80)
    
    
    test_folder = Path(Config.PROJECT_ROOT) / 'test/processed'
    
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    
    print("\nLoading test images from folder structure...")
    images, labels, class_names, image_paths = load_test_images_from_folder(
        test_folder, 
        transform=transform
    )
    
    if len(images) == 0:
        print("\n⚠ Error: No test images found!")
        print(f"Please check that test images exist in: {test_folder}")
        print("Expected structure: test/class_name/image.jpg")
        return None
    
    
    print("\nCreating model...")
    model = create_model(num_classes=len(class_names))
    model = model.to(Config.DEVICE)
    
    checkpoint_path = Config.CHECKPOINT_DIR / 'best_model.pth'
    if checkpoint_path.exists():
        print(f"\nLoading checkpoint: {checkpoint_path}")
        optimizer = torch.optim.AdamW(model.parameters())
        epoch, checkpoint_metrics = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Loaded model from epoch {epoch}")
        if 'val_acc' in checkpoint_metrics:
            print(f"Validation accuracy: {checkpoint_metrics['val_acc']:.2f}%")
    else:
        print(f"\n⚠ Warning: No checkpoint found at {checkpoint_path}")
        print("Evaluating untrained model...")
    

    print("\n" + "-"*80)
    print("EVALUATING ON TEST SET")
    print("-"*80)
    y_true, y_pred, y_probs = evaluate_model(
        model, images, labels, Config.DEVICE, batch_size=Config.BATCH_SIZE
    )
    

    metrics = calculate_detailed_metrics(y_true, y_pred, class_names)
    
    print_detailed_metrics(metrics, class_names)
    
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    ))
    
    print("\nGenerating confusion matrix...")
    cm_path = Config.RESULTS_DIR / 'confusion_matrix.png'
    cm = plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    
    metrics_path = Config.RESULTS_DIR / 'test_metrics.json'
    save_metrics(metrics, metrics_path)
    print(f"Metrics saved to: {metrics_path}")
    
    print("\n" + "="*80)
    print("PER-CLASS ACCURACY")
    print("="*80)
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == y_true[class_mask]).mean() * 100
            class_count = class_mask.sum()
            print(f"{class_name:20s}: {class_acc:6.2f}% ({class_count} samples)")
        else:
            print(f"{class_name:20s}: No samples")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    print(f"\n✓ All results saved to: {Config.RESULTS_DIR}")
    print(f"  - Confusion matrix: {cm_path}")
    print(f"  - Test metrics: {metrics_path}")
    print("\nSUMMARY:")
    print(f"  - Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  - Precision (Macro): {metrics['precision_macro']*100:.2f}%")
    print(f"  - Recall (Macro): {metrics['recall_macro']*100:.2f}%")
    print(f"  - F1 Score (Macro): {metrics['f1_macro']*100:.2f}%")
    print("="*80 + "\n")
    
    return metrics

if __name__ == "__main__":
    main()