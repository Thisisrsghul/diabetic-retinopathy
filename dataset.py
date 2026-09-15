import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms


def _load_label_csv(csv_path, split="left"):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    
    img_col = next((c for c in df.columns if "image" in c.lower()), df.columns[0])
    target_col = next((c for c in ["level", "label", "dr_level"] if c in df.columns), None)
    
    if split in ("left", "right"):
        df = df[df[img_col].str.endswith(f"_{split}")].copy()
    
    rename_dict = {img_col: "image"}
    if target_col:
        rename_dict[target_col] = "level"
    return df.rename(columns=rename_dict).reset_index(drop=True)


class DRDataset(Dataset):
    def __init__(self, df, img_dir, img_size, train=True, morph_cache=None, name_to_row=None):
        self.df = df
        self.img_dir = img_dir
        self.img_size = img_size
        self.train = train
        self.morph_cache = morph_cache
        self.name_to_row = name_to_row

        # Standard augmentations suitable for enhanced grayscale fundus images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.RandomVerticalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(360) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row["image"])
        
        # Check extensions
        img_path = os.path.join(self.img_dir, f"{img_name}.jpeg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, f"{img_name}.png")

        # Read the enhanced image as grayscale (since MATLAB locallapfilt saved single-channel)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if gray is None:
            gray = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # Duplicate single channel to 3 channels (RGB) for EfficientNet compatibility
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        img_tensor = self.transform(img_rgb)
        gray_tensor = torch.from_numpy(gray).float().unsqueeze(0) / 255.0

        item = {
            "image": img_tensor,
            "gray": gray_tensor,
            "image_name": img_name
        }

        if "level" in row:
            item["label"] = torch.tensor(float(row["level"]), dtype=torch.float32)

        if self.morph_cache is not None and self.name_to_row is not None:
            if img_name in self.name_to_row:
                cache_idx = self.name_to_row[img_name]
                item["morph"] = torch.from_numpy(self.morph_cache[cache_idx]).float()

        return item
