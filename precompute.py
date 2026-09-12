import os
import cv2
import numpy as np
from tqdm import tqdm
from config import Config
from dataset import _load_label_csv
from matched_filter import matched_filter_vessels
from feature_extractor import extract_all_features


def _find_image(name, img_dir):
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".bmp", ".JPG", ".JPEG"):
        p = os.path.join(img_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


def precompute_one(csv_path, tag, img_dir):
    df = _load_label_csv(csv_path, split="left")
    print(f"[{tag}] rows in CSV: {len(df)}  |  image dir: {img_dir}")

    feats = np.zeros((len(df), Config.MORPH_DIM), dtype=np.float32)
    kept_rows = []
    n_found = n_missing = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc=tag):
        name = row["image"]
        path = _find_image(name, img_dir)
        if path is None:
            n_missing += 1
            continue

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            n_missing += 1
            continue
        if img.shape[:2] != (Config.IMG_SIZE, Config.IMG_SIZE):
            img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))

        mask = matched_filter_vessels(img)
        feats[i] = extract_all_features(mask > 0, orig_gray=img,
                                        min_segment_len=Config.MIN_SEGMENT_LEN)
        kept_rows.append(i)
        n_found += 1

    feats = feats[kept_rows]
    df_found = df.iloc[kept_rows].reset_index(drop=True)

    np.save(os.path.join(Config.CACHE_DIR, f"{tag}_morph.npy"), feats)
    df_found.to_csv(os.path.join(Config.CACHE_DIR, f"{tag}_index.csv"),
                    index=False, encoding="utf-8-sig")
    print(f"[{tag}] found: {n_found}  missing: {n_missing}  saved: {feats.shape}")


if __name__ == "__main__":
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    precompute_one(Config.TRAIN_CSV, "train", Config.TRAIN_DIR)
    precompute_one(Config.TEST_CSV,  "test",  Config.TEST_DIR)