import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import sys
from .config import Config
from .dataset import create_dataloaders
from .model import create_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from .utils import (
    save_checkpoint, plot_training_history,
    EarlyStopping, save_metrics, AverageMeter
)
sys.path.append(str(Path(__file__).parent))



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss

def save_confusion_matrix(model, dataloader, device, class_names, epoch, save_dir):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.savefig(save_dir / f'confusion_matrix_epoch_{epoch}.png')
    plt.close()
    print(f"✓ Confusion matrix saved for epoch {epoch}")

def find_latest_checkpoint():
    checkpoint_dir = Config.CHECKPOINT_DIR
    best_path = checkpoint_dir / 'best_model.pth'
    if best_path.exists():
        return best_path, 'best'

    final_path = checkpoint_dir / 'final_model.pth'
    if final_path.exists():
        return final_path, 'final'
    epoch_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if epoch_checkpoints:
        epoch_checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return epoch_checkpoints[-1], 'epoch'
    return None, None


def load_checkpoint_for_resume(model, optimizer, scheduler, scaler=None):
    checkpoint_path, checkpoint_type = find_latest_checkpoint()
    if checkpoint_path is None:
        print("\n" + "="*80)
        print("NO CHECKPOINT FOUND - STARTING FRESH TRAINING")
        print("="*80)
        return 0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}, 0.0
    
    print("\n" + "="*80)
    print(f"RESUMING FROM CHECKPOINT: {checkpoint_type.upper()}")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model state loaded")
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
        
        
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✓ Scheduler state loaded")
            except(Exception):
                print("⚠ Could not load scheduler state (will use default)")
        
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("✓ Scaler state loaded")
            except(Exception):
                print("⚠ Could not load scaler state (will use default)")
        
        
        start_epoch = checkpoint.get('epoch', 0) + 1 
        
        history = checkpoint.get('history', {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
        })
        
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        print(f"\nResume Information:")
        print(f"  Starting from epoch: {start_epoch}")
        print(f"  Best validation accuracy: {best_val_acc:.2f}%")
        print(f"  Training history length: {len(history['train_loss'])} epochs")
        print("="*80 + "\n")
        
        return start_epoch, history, best_val_acc
        
    except Exception as e:
        print(f"\n⚠️  Error loading checkpoint: {e}")
        print("Starting fresh training instead...")
        import traceback
        traceback.print_exc()
        return 0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}, 0.0


def save_checkpoint_enhanced(model, optimizer, scheduler, scaler, epoch, history,best_val_acc, metrics, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'history': history,
        'best_val_acc': best_val_acc,
        'metrics': metrics,
        'config': {
            'num_classes': Config.NUM_CLASSES,
            'img_size': Config.IMG_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'batch_size': Config.BATCH_SIZE,
        }
    }
    
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [TRAIN]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        optimizer.zero_grad()
        if scaler is not None and Config.USE_AMP:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100.0 * correct / batch_size
        
        losses.update(loss.item(), batch_size)
        accuracies.update(accuracy, batch_size)
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.2f}%'
        })
    
    return losses.avg, accuracies.avg


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [VALID]')
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = 100.0 * correct / batch_size
            
            losses.update(loss.item(), batch_size)
            accuracies.update(accuracy, batch_size)
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.2f}%'
            })
    
    return losses.avg, accuracies.avg


def train_model():
    print("\n" + "="*80)
    print("DENTAL X-RAY CLASSIFICATION - TRAINING")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Device: {Config.DEVICE}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Total Epochs: {Config.NUM_EPOCHS}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print(f"  Image Size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"  Number of Classes: {Config.NUM_CLASSES}")
    print(f"  Mixed Precision: {Config.USE_AMP}")
    
    print("\n" + "-"*80)
    print("Loading Dataset...")
    print("-"*80)
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        Config.DATA_DIR,
        Config.BATCH_SIZE
    )
    
    print("\n" + "-"*80)
    print("Creating Model...")
    print("-"*80)
    model = create_model(num_classes=Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)
    
    if hasattr(Config, 'LABEL_SMOOTHING') and Config.LABEL_SMOOTHING > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=Config.LABEL_SMOOTHING)
        print(f"Using Label Smoothing: {Config.LABEL_SMOOTHING}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss")
    
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    
    if hasattr(Config, 'LR_SCHEDULER'):
        if Config.LR_SCHEDULER == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=Config.NUM_EPOCHS,
                eta_min=Config.LR_MIN if hasattr(Config, 'LR_MIN') else 1e-6
            )
            print(f"Using Cosine Annealing LR Scheduler")
        elif Config.LR_SCHEDULER == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
            print(f"Using Step LR Scheduler")
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max',
                factor=0.5, 
                patience=10,
                verbose=True
            )
            print(f"Using ReduceLROnPlateau Scheduler")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        print("Using default ReduceLROnPlateau Scheduler")
    
    
    scaler = GradScaler() if (hasattr(Config, 'USE_AMP') and Config.USE_AMP) else None
    
    
    start_epoch, history, best_val_acc = load_checkpoint_for_resume(
        model, optimizer, scheduler, scaler
    )
    
    
    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE,
        mode='max'
    )
    
    
    if start_epoch > 0 and len(history['val_acc']) > 0:
        early_stopping.best_score = best_val_acc
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch}")
    print(f"Training until epoch {Config.NUM_EPOCHS}")
    print("="*80 + "\n")
    
    last_train_loss = 0.0
    last_train_acc = 0.0
    last_val_loss = 0.0
    last_val_acc = 0.0

    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE, epoch+1, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, Config.DEVICE, epoch+1
        )
        
        # Store last epoch metrics (FIX: Update these every epoch)
        last_train_loss = train_loss
        last_train_acc = train_acc
        last_val_loss = val_loss
        last_val_acc = val_acc
        
        # Update learning rate
        if hasattr(Config, 'LR_SCHEDULER') and Config.LR_SCHEDULER == 'plateau':
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        # Save to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = Config.CHECKPOINT_DIR / 'best_model.pth'
            save_checkpoint_enhanced(
                model, optimizer, scheduler, scaler, epoch,
                history, best_val_acc,
                {'val_acc': val_acc, 'val_loss': val_loss, 'train_acc': train_acc, 'train_loss': train_loss},
                best_model_path
            )

            print(f"  🌟 New best model! Val Acc: {val_acc:.2f}%")
        
        print(f"  Best Val Acc: {best_val_acc:.2f}%")
        print(f"{'='*80}")
        
        # Save periodic checkpoint
        if (epoch + 1) % Config.SAVE_EVERY == 0:
            checkpoint_path = Config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint_enhanced(
                model, optimizer, scheduler, scaler, epoch,
                history, best_val_acc,
                {'val_acc': val_acc, 'val_loss': val_loss, 'train_acc': train_acc, 'train_loss': train_loss},
                checkpoint_path
            )
            save_confusion_matrix(model, val_loader, Config.DEVICE, class_names, epoch+1, Config.LOG_DIR)
        
        # Early stopping check
        if early_stopping(val_acc):
            print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
            break
    
    # FIX: Use last_val_acc instead of val_acc for final save
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Total Epochs Trained: {len(history['train_loss'])}")
    
    # Save final model (FIX: Use last epoch metrics)
    final_model_path = Config.CHECKPOINT_DIR / 'final_model.pth'
    save_checkpoint_enhanced(
        model, optimizer, scheduler, scaler, 
        len(history['train_loss']) - 1,  # Last epoch index
        history, best_val_acc,
        {
            'val_acc': last_val_acc, 
            'val_loss': last_val_loss, 
            'train_acc': last_train_acc, 
            'train_loss': last_train_loss
        },
        final_model_path
    )
    
    # Save training history
    history_path = Config.RESULTS_DIR / 'training_history.json'
    save_metrics(history, history_path)
    
    # Plot training history
    print("\nGenerating training history plot...")
    plot_path = Config.RESULTS_DIR / 'training_history.png'
    plot_training_history(history, plot_path)
    
    print("\n✓ Training complete! All results saved.")
    print(f"  - Best model: {Config.CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"  - Final model: {final_model_path}")
    print(f"  - Training history: {history_path}")
    print(f"  - Training plot: {plot_path}")
    
    return model, history


if __name__ == "__main__":
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Progress has been saved. Run again to resume.")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()