import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import DRDataset, _load_label_csv
from model import DRClassifier
from train import compute_morph_batch


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
        idx_df = pd.read_csv(os.path.join(Config.CACHE_DIR, "test_index.csv"),
                             encoding="utf-8-sig")
        idx_df.columns = [c.strip() for c in idx_df.columns]
        name_to_row = {n: i for i, n in enumerate(idx_df["image"].tolist())}
        use_cache = True
        print(f"Using cache: {cache_path} {morph_cache.shape}")

    ds = DRDataset(df, Config.TEST_DIR, Config.IMG_SIZE, train=False,
                   morph_cache=morph_cache, name_to_row=name_to_row)
    ld = DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                    num_workers=Config.NUM_WORKERS)

    model = DRClassifier(Config.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(Config.CHECKPOINT, map_location=device))
    model.eval()

    preds = []
    for batch in tqdm(ld, desc="predict"):
        img = batch["image"].to(device)
        if use_cache and "morph" in batch:
            morph = batch["morph"].to(device)
        else:
            morph = compute_morph_batch(batch["gray"].to(device))
        preds.extend(model(img, morph).argmax(1).cpu().numpy().tolist())

    df["predicted_level"] = preds
    out = os.path.join(Config.OUTPUT_DIR, "predictions_left.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("Saved →", out)


if __name__ == "__main__":
    main()