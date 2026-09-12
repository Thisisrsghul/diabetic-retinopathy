import os
import pandas as pd

csv_path = "test_labels_left.csv"
image_folder = "test"

# 1. Read the CSV
df = pd.read_csv(csv_path)

# 2. Collect existing file stems from the disk into a set for O(1) lookup
# Strips '.jpg' (or checks with extension if your image names already include it)
existing_images = {
    os.path.splitext(f)[0]
    for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
}

# 3. Filter DataFrame
mask = df["image"].astype(str).isin(existing_images)
df_fixed = df[mask]
missing = df[~mask]

# 4. Save the cleaned CSV
df_fixed.to_csv("test_fixed.csv", index=False)

# 5. Summary
print(f"Original rows: {len(df):,}")
print(f"Missing rows:  {len(missing):,}")
print(f"Final rows:    {len(df_fixed):,}")