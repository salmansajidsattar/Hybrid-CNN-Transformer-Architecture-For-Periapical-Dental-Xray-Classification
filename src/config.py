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

    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 4e-4
    WEIGHT_DECAY = 5e-5

    
    

    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    

    # NOTE: CNN_CHANNELS / NUM_HEADS / NUM_TRANSFORMER_LAYERS / MLP_DIM / DROPOUT
    # below are read ONLY by the legacy V1 model (model.py / train.py).
    # The HCNT model reported in the paper (model_v2.py / train_v2.py /
    # run_seeds.py) uses the separate "V2 model settings" block further
    # down (NUM_HEADS_V2, NUM_LAYERS_V2, etc.) and does NOT read these.
    CNN_CHANNELS = [64, 128, 256, 512]          # V1 only
    EMBED_DIM = 512                              # shared: also read by V2 (see below)
    NUM_HEADS = 16                                # V1 only
    NUM_TRANSFORMER_LAYERS = 4  # Increased       # V1 only
    MLP_DIM = 2048                                # V1 only
    DROPOUT = 0.25  # Increased for regularization  # V1 only
    

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

    # ── V2 model settings (model_v2.py / train_v2.py / ensemble.py) ────────
    # Backbone: 'resnet50' | 'radimagen' | 'densenet121' | 'efficientnet'
    #   'radimagen'   → ResNet-50 pretrained on 1.35M radiology images (best)
    #   'densenet121' → DenseNet-121 pretrained on radiology images (set RADIMAGEN_CKPT to DenseNet121.pt)
    BACKBONE           = 'radimagen'
    # Portable path: resolved relative to the project root so this works
    # unchanged on Windows, Linux, and Kaggle (where the project root is
    # wherever the dataset/notebook working directory is mounted), instead
    # of the previous hardcoded 'F:\Dental_Xray_Classification\...' path.
    # Override by setting the RADIMAGEN_CKPT environment variable.
    import os as _os
    RADIMAGEN_CKPT     = Path(_os.environ.get(
        'RADIMAGEN_CKPT', str(PROJECT_ROOT / 'RadImageNet_pytorch' / 'ResNet50.pt')))
    # To switch to DenseNet-121 backbone, change to:
    # BACKBONE       = 'densenet121'
    # RADIMAGEN_CKPT = Path(_os.environ.get(
    #     'RADIMAGEN_CKPT', str(PROJECT_ROOT / 'RadImageNet_pytorch' / 'DenseNet121.pt')))

    # Backbone LR scale (10× lower than Transformer to avoid forgetting)
    BACKBONE_LR_SCALE  = 0.1
    FREEZE_STAGES      = 1            # freeze stem+layer1 initially
    UNFREEZE_EPOCH     = 10           # epoch to fully unfreeze backbone

    # Transformer architecture (v2)
    NUM_HEADS_V2       = 8
    NUM_LAYERS_V2      = 6
    MLP_RATIO          = 4.0
    ATTN_DROP          = 0.0
    DROP_PATH_RATE     = 0.1          # stochastic depth
    LAYER_SCALE_INIT   = 1e-5         # LayerScale init

    # LR schedule
    WARMUP_EPOCHS      = 5            # linear warmup epochs

    # Focal Loss + OHEM
    FOCAL_GAMMA        = 2.0          # Focal Loss gamma (2.0 = standard)
    FOCAL_ALPHA        = 0.25         # Focal Loss alpha
    OHEM_KEEP_RATIO    = 0.7          # keep top-70% hardest per batch

    # Test-Time Augmentation
    USE_TTA            = True
    TTA_N_AUG          = 5            # number of augmented views

    # Ablation toggles (see src/run_seeds.py ABLATION_CONFIGS) -- all True
    # by default, i.e. the exact full HCNT configuration described in the
    # paper. run_seeds.py flips these one at a time via setattr/getattr.
    USE_TRANSFORMER    = True         # False -> GAP + linear head (no Transformer)
    USE_POS_EMBED      = True         # False -> no positional embeddings
    USE_LAYER_SCALE    = True         # False -> LayerScale disabled (fixed unit gain)
    USE_FOCAL_OHEM     = True         # False -> plain CrossEntropyLoss