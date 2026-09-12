import os

BASE = os.path.dirname(os.path.abspath(__file__))

class Config:
    # ---------- Paths (relative to project root) ----------
    TRAIN_DIR = os.path.join(BASE, "train")     # enhanced train images
    TEST_DIR  = os.path.join(BASE, "test")      # enhanced test images
    ENHANCED_DIR = TRAIN_DIR                    # generic alias

    TRAIN_CSV = os.path.join(BASE, "train_labels_left.csv")
    TEST_CSV  = os.path.join(BASE, "test_fixed.csv")   # filtered left-only
    TEST_CSV_FULL = os.path.join(BASE, "testLabels.csv")

    CHECKPOINT = os.path.join(BASE, "checkpoints", "best_model.pth")
    OUTPUT_DIR = os.path.join(BASE, "outputs")
    CACHE_DIR  = os.path.join(BASE, "cache")

    # ---------- Image ----------
    IMG_SIZE = 512
    MIN_SEGMENT_LEN = 8

    # ---------- Training ----------
    BATCH_SIZE   = 16  # Increase to 16 or 32 now that AMP is active
    EPOCHS       = 30  # CosineAnnealing converges fast on frozen heads
    LR           = 3e-4 # Slightly higher LR suited for frozen backbone + linear head
    NUM_WORKERS  = 4
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES  = 5
    SEED         = 42

    # ---------- Feature dims ----------
    MORPH_DIM = 30
    CNN_DIM   = 1280
    FUSED_DIM = MORPH_DIM + CNN_DIM

    # ---------- Attention ----------
    ATTN_HEADS = 8
    DROPOUT    = 0.4

    DEVICE = "cuda"

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(Config.CHECKPOINT), exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)