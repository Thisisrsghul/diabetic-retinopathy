import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

from config import Config
from dataset import DRDataset, _load_label_csv
from model import DRClassifier
from matched_filter import matched_filter_vessels
from feature_extractor import extract_all_features


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


class OptimizedRounder:
    """Finds optimal decision boundaries to map continuous predictions to discrete DR grades."""
    def __init__(self):
        self.coef_ = 0

    def _loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred < coef[1]:
                X_p[i] = 1
            elif pred < coef[2]:
                X_p[i] = 2
            elif pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return -cohen_kappa_score(y, X_p, weights="quadratic")

    def fit(self, X, y):
        init_coef = [0.5, 1.5, 2.5, 3.5]
        res = minimize(self._loss, init_coef, args=(X, y), method='Nelder-Mead')
        self.coef_ = res.x

    def predict(self, X):
        X_p = np.copy(X)
        res = np.zeros(len(X_p), dtype=int)
        for i, pred in enumerate(X_p):
            if pred < self.coef_[0]:
                res[i] = 0
            elif pred < self.coef_[1]:
                res[i] = 1
            elif pred < self.coef_[2]:
                res[i] = 2
            elif pred < self.coef_[3]:
                res[i] = 3
            else:
                res[i] = 4
        return res


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
    running, total = 0.0, 0
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
            preds = model(img, morph)
            loss = crit(preds, label)

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optim)
        scaler.update()

        running += loss.item() * img.size(0)
        total += img.size(0)
        pbar.set_postfix(loss=f"{running/total:.4f}")

    return running / total


@torch.no_grad()
def evaluate(model, loader, crit, device, use_cache):
    model.eval()
    running, total = 0.0, 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="val", leave=False):
        img = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        if use_cache and "morph" in batch:
            morph = batch["morph"].to(device, non_blocking=True)
        else:
            morph = compute_morph_batch(batch["gray"].to(device, non_blocking=True))

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            preds = model(img, morph)
            loss = crit(preds, label)

        running += loss.item() * img.size(0)
        total += img.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(label.cpu().numpy().tolist())

    return running / total, np.array(all_preds), np.array(all_labels)


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

    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=Config.SEED,
        stratify=df["level"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

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

    model = DRClassifier(num_classes=1).to(device)

    for param in model.cnn.parameters():
        param.requires_grad = True

    crit = nn.SmoothL1Loss()

    backbone_params = list(model.cnn.parameters())
    fusion_params = [p for n, p in model.named_parameters() if not n.startswith("cnn")]

    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": 2e-5},
        {"params": fusion_params,   "lr": 2e-4}
    ], weight_decay=Config.WEIGHT_DECAY)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    best_qwk = -1.0
    best_thresholds = [0.5, 1.5, 2.5, 3.5]
    rounder = OptimizedRounder()

    for ep in range(1, 32):
        tr_loss = train_one_epoch(model, tr_ld, optim, crit, scaler, device, use_cache)
        va_loss, val_preds, val_labels = evaluate(model, va_ld, crit, device, use_cache)
        sched.step()

        rounder.fit(val_preds, val_labels)
        discrete_preds = rounder.predict(val_preds)
        val_qwk = cohen_kappa_score(val_labels, discrete_preds, weights="quadratic")
        val_acc = (discrete_preds == val_labels).mean()

        print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f} "
              f"acc {val_acc:.4f} qwk {val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_thresholds = rounder.coef_.tolist()
            torch.save(model.state_dict(), Config.CHECKPOINT)
            # Save thresholds alongside weights for inference
            thresh_file = os.path.join(Config.BASE_WORK, "thresholds.json")
            with open(thresh_file, "w") as f:
                json.dump(best_thresholds, f)
            print(f"  ✔ saved best (QWK={best_qwk:.4f}, thresholds={best_thresholds})")

    print(f"Training Complete. Best val QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    main()
