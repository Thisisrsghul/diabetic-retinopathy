import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import Config


def _load_label_csv(csv_path, split="left"):
    """Read CSV; strip BOM; keep only '<id>_<split>' rows."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df[df["image"].astype(str).str.endswith(f"_{split}")].reset_index(drop=True)
    return df


class DRDataset(Dataset):
    def __init__(self, df, img_dir, img_size=512, train=True,
                 morph_cache=None, name_to_row=None):
        self.df = df
        self.img_dir = img_dir
        self.img_size = img_size
        self.train = train
        self.morph_cache = morph_cache
        self.name_to_row = name_to_row

        self.tf_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.tf_eval = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def _find_image(self, name):
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".bmp", ".JPG", ".JPEG"):
            p = os.path.join(self.img_dir, name + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"No image for {name} in {self.img_dir}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["image"]
        label = int(row["level"]) if "level" in row else -1

        img_path = self._find_image(name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise RuntimeError(f"cv2 could not read {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (self.img_size, self.img_size))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        tf = self.tf_train if self.train else self.tf_eval
        img_tensor = tf(img_rgb)

        sample = {
            "image": img_tensor,
            "gray": torch.from_numpy(gray).float().unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "name": name,
        }
        if self.morph_cache is not None and self.name_to_row is not None:
            r = self.name_to_row.get(name, -1)
            if r >= 0:
                sample["morph"] = torch.from_numpy(self.morph_cache[r]).float()
        return sample