import torch
from pathlib import Path

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data/processed"
    PROCESSED_DIR = PROJECT_ROOT / "data/processed"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    LOG_DIR = PROJECT_ROOT / "logs"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    for dir_path in [PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR, RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    IMG_SIZE = 384
    NUM_CLASSES = 2
    CLASS_NAMES = ['non_periapical', 'periapical']
    # CLASS_NAMES=['Primary Endo with Secondary Perio','Primary Endodontic Lesion','Primary Perio with Secondary Endo','Primary Periodontal Lesion','True Combined Lesions']

    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 4e-4
    WEIGHT_DECAY = 5e-5

    
    

    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    

    CNN_CHANNELS = [64, 128, 256, 512]
    EMBED_DIM = 512
    NUM_HEADS = 16
    NUM_TRANSFORMER_LAYERS = 4  # Increased
    MLP_DIM = 2048
    DROPOUT = 0.25  # Increased for regularization
    

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Checkpoint settings
    SAVE_EVERY = 1
    SAVE_BEST = True
    EARLY_STOPPING_PATIENCE = 100
    
    ROTATION = 10
    BRIGHTNESS = 0.1
    CONTRAST = 0.1
    USE_MIXUP = False
    USE_CUTMIX = False

    
    # Advanced settings
    LR_SCHEDULER = 'cosine'
    LR_MIN = 1e-6
    LABEL_SMOOTHING = 0.05
    USE_AMP = True
    GRAD_CLIP = 1.0
    
    # Advanced augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    USE_CUTMIX = True
    CUTMIX_PROB = 0.5