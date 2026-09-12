import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import DRDataset, _load_label_csv
from model import DRClassifier
from matched_filter import matched_filter_vessels
from feature_extractor import extract_all_features


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def compute_morph_batch(gray_tensor):
    feats = []
    for i in range(gray_tensor.size(0)):
        g = (gray_tensor[i, 0].cpu().numpy() * 255).astype(np.uint8)
        mask = matched_filter_vessels(g)
        f = extract_all_features(mask > 0, orig_gray=g,
                                 min_segment_len=Config.MIN_SEGMENT_LEN)
        feats.append(f)
    return torch.from_numpy(np.stack(feats)).float().to(gray_tensor.device)


def train_one_epoch(model, loader, optim, crit, scaler, device, use_cache):
    model.train()
    running, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="train", leave=False)

    for batch in pbar:
        img = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        if use_cache and "morph" in batch:
            morph = batch["morph"].to(device, non_blocking=True)
        else:
            morph = compute_morph_batch(batch["gray"].to(device, non_blocking=True))

        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(img, morph)
            loss = crit(logits, label)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        running += loss.item() * img.size(0)
        correct += (logits.argmax(1) == label).sum().item()
        total += img.size(0)
        pbar.set_postfix(loss=f"{running/total:.4f}", acc=f"{correct/total:.3f}")

    return running / total, correct / total


@torch.no_grad()
def evaluate(model, loader, crit, device, use_cache):
    model.eval()
    running, correct, total = 0.0, 0, 0
    for batch in tqdm(loader, desc="val", leave=False):
        img = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        if use_cache and "morph" in batch:
            morph = batch["morph"].to(device, non_blocking=True)
        else:
            morph = compute_morph_batch(batch["gray"].to(device, non_blocking=True))

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(img, morph)
            loss = crit(logits, label)

        running += loss.item() * img.size(0)
        correct += (logits.argmax(1) == label).sum().item()
        total += img.size(0)
    return running / total, correct / total


def main():
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = _load_label_csv(Config.TRAIN_CSV, split="left")
    print(f"Train rows (left): {len(df)}")
    print(df["level"].value_counts().sort_index())

    morph_cache = None
    name_to_row = None
    use_cache = False
    cache_path = os.path.join(Config.CACHE_DIR, "train_morph.npy")
    if os.path.exists(cache_path):
        morph_cache = np.load(cache_path)
        idx_df = pd.read_csv(os.path.join(Config.CACHE_DIR, "train_index.csv"),
                             encoding="utf-8-sig")
        idx_df.columns = [c.strip() for c in idx_df.columns]
        name_to_row = {n: i for i, n in enumerate(idx_df["image"].tolist())}
        use_cache = True
        print(f"Using cache: {cache_path} {morph_cache.shape}")
    else:
        print("[WARNING] Cache not found! Run precompute.py first to avoid severe slowdowns.")

    n_val = max(1, int(0.2 * len(df)))
    n_tr = len(df) - n_val
    train_df = df.iloc[:n_tr].reset_index(drop=True)
    val_df = df.iloc[n_tr:].reset_index(drop=True)

    train_ds = DRDataset(train_df, Config.TRAIN_DIR, Config.IMG_SIZE, train=True,
                         morph_cache=morph_cache, name_to_row=name_to_row)
    val_ds = DRDataset(val_df, Config.TRAIN_DIR, Config.IMG_SIZE, train=False,
                       morph_cache=morph_cache, name_to_row=name_to_row)

    loader_kwargs = {
        "batch_size": Config.BATCH_SIZE,
        "num_workers": Config.NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": (Config.NUM_WORKERS > 0),
    }

    tr_ld = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    va_ld = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = DRClassifier(num_classes=Config.NUM_CLASSES).to(device)

    # Freeze EfficientNet backbone parameters for fast initial training
    # (Unfreeze later if fine-tuning accuracy requires it)
    for param in model.cnn.parameters():
        param.requires_grad = False

    counts = df["level"].value_counts().sort_index().reindex(
        range(Config.NUM_CLASSES), fill_value=1).values
    weights = torch.tensor(counts.max() / counts, dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=Config.LR,
                              weight_decay=Config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=Config.EPOCHS)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    best_acc = 0.0
    for ep in range(1, Config.EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_ld, optim, crit, scaler, device, use_cache)
        va_loss, va_acc = evaluate(model, va_ld, crit, device, use_cache)
        sched.step()
        print(f"Epoch {ep:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), Config.CHECKPOINT)
            print(f"  ✔ saved best (val_acc={best_acc:.4f})")
    print("Done. Best val acc:", best_acc)


if __name__ == "__main__":
    main()