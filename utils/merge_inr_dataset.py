import shutil
from pathlib import Path

SRC_BASE = Path("dataset/raw/INR")
DST_BASE = Path("dataset/raw/INR_clean")

DST_BASE.mkdir(exist_ok=True)

for split in ["training", "validation", "test"]:
    split_dir = SRC_BASE / split
    if not split_dir.exists():
        continue

    for denom_dir in split_dir.iterdir():
        if not denom_dir.is_dir():
            continue

        dst_denom = DST_BASE / denom_dir.name
        dst_denom.mkdir(parents=True, exist_ok=True)

        for img in denom_dir.glob("*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                shutil.copy(img, dst_denom / img.name)

print("✅ INR dataset merged successfully")