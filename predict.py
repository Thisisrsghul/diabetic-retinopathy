import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt

from config import Config
from dataset import DRDataset, _load_label_csv
from model import DRClassifier
from train import compute_morph_batch


def plot_and_save_confusion_matrix(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True Label',
        xlabel='Predicted Label',
        title='Diabetic Retinopathy Confusion Matrix'
    )

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix plot saved → {output_path}")


@torch.no_grad()
def main():
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = _load_label_csv(Config.TEST_CSV, split="left")
    print(f"Test rows (left): {len(df)}")

    morph_cache = None
    name_to_row = None
    use_cache = False
    cache_path = os.path.join(Config.CACHE_DIR, "test_morph.npy")
    if os.path.exists(cache_path):
        morph_cache = np.load(cache_path)
        idx_df = pd.read_csv(os.path.join(Config.CACHE_DIR, "test_index.csv"), encoding="utf-8-sig")
        idx_df.columns = [c.strip() for c in idx_df.columns]
        name_to_row = {n: i for i, n in enumerate(idx_df["image"].tolist())}
        use_cache = True
        print(f"Using cache: {cache_path} {morph_cache.shape}")

    ds = DRDataset(df, Config.TEST_DIR, Config.IMG_SIZE, train=False,
                   morph_cache=morph_cache, name_to_row=name_to_row)
    ld = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                    num_workers=Config.NUM_WORKERS)

    model = DRClassifier(num_classes=1).to(device)
    model.load_state_dict(torch.load(Config.CHECKPOINT, map_location=device))
    model.eval()

    # Load calibrated decision thresholds
    thresh_file = os.path.join(Config.BASE_WORK, "thresholds.json")
    if os.path.exists(thresh_file):
        with open(thresh_file, "r") as f:
            coef = json.load(f)
        print(f"Loaded decision thresholds: {coef}")
    else:
        coef = [0.5, 1.5, 2.5, 3.5]
        print("Using standard thresholds [0.5, 1.5, 2.5, 3.5]")

    raw_preds = []
    for batch in tqdm(ld, desc="predict"):
        img = batch["image"].to(device)
        if use_cache and "morph" in batch:
            morph = batch["morph"].to(device)
        else:
            morph = compute_morph_batch(batch["gray"].to(device))

        # Test-Time Augmentation: Original, H-Flip, V-Flip
        pred1 = model(img, morph)
        pred2 = model(torch.flip(img, dims=[3]), morph)
        pred3 = model(torch.flip(img, dims=[2]), morph)
        avg_pred = (pred1 + pred2 + pred3) / 3.0

        raw_preds.extend(avg_pred.cpu().numpy().tolist())

    # Map continuous regression outputs to discrete classes
    discrete_preds = []
    for pred in raw_preds:
        if pred < coef[0]:
            discrete_preds.append(0)
        elif pred < coef[1]:
            discrete_preds.append(1)
        elif pred < coef[2]:
            discrete_preds.append(2)
        elif pred < coef[3]:
            discrete_preds.append(3)
        else:
            discrete_preds.append(4)

    df["predicted_level"] = discrete_preds
    df["raw_continuous_score"] = raw_preds
    out_csv = os.path.join(Config.OUTPUT_DIR, "predictions_left.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved →", out_csv)

    if "level" in df.columns:
        y_true = df["level"].to_numpy().astype(int)
        y_pred = np.array(discrete_preds)
        classes = [f"Class {i}" for i in range(5)]

        print("\n" + "=" * 55)
        print("                 CLASSIFICATION REPORT")
        print("=" * 55)
        print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
        print("=" * 55)
        print("                   CONFUSION MATRIX")
        print("=" * 55)
        cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in classes], columns=[f"Pred {c}" for c in classes])
        print(cm_df.to_string())
        print("=" * 55)

        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")

        cm_png = os.path.join(Config.OUTPUT_DIR, "confusion_matrix.png")
        plot_and_save_confusion_matrix(cm, classes, cm_png)


if __name__ == "__main__":
    main()
