import os
import torch

class Config:
    BASE_DATA = "/kaggle/input/datasets/yuva876/dr-dataset/traintest"
    TRAIN_DIR = os.path.join(BASE_DATA, "train")
    TEST_DIR  = os.path.join(BASE_DATA, "test")
    ENHANCED_DIR = TRAIN_DIR

    TRAIN_CSV = "/kaggle/input/datasets/yuva876/dr-dataset/traintest/train_labels_left.csv"
    TEST_CSV  = "/kaggle/input/datasets/yuva876/dr-dataset/traintest/test_fixed.csv"
    TEST_CSV_FULL = os.path.join(BASE_DATA, "testLabels.csv")

    BASE_WORK = "/kaggle/working"
    CHECKPOINT = os.path.join(BASE_WORK, "checkpoints", "best_model.pth")
    OUTPUT_DIR = os.path.join(BASE_WORK, "outputs")
    CACHE_DIR  = os.path.join(BASE_WORK, "cache")

    IMG_SIZE = 512
    MIN_SEGMENT_LEN = 8

    # Hardware (Dual T4 x2)
    NUM_GPUS     = torch.cuda.device_count()
    BATCH_SIZE   = 32 if NUM_GPUS > 1 else 16
    EPOCHS       = 30
    LR           = 6e-4 if NUM_GPUS > 1 else 3e-4
    NUM_WORKERS  = 4
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES  = 5
    SEED         = 42

    MORPH_DIM = 30
    CNN_DIM   = 1280
    FUSED_DIM = MORPH_DIM + CNN_DIM

    ATTN_HEADS = 10
    DROPOUT    = 0.4
    DEVICE     = "cuda"

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(Config.CHECKPOINT), exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)
