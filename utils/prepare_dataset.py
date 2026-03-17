import random
import shutil
from pathlib import Path

RAW_DIR = Path("dataset/raw")
OUT_DIR = Path("dataset/processed")

SPLIT = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

random.seed(42)

def prepare_country(country_dir):
    country = country_dir.name

    for denom_dir in country_dir.iterdir():
        if not denom_dir.is_dir():
            continue

        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images.extend(denom_dir.glob(ext))

        if len(images) < 5:
            continue

        random.shuffle(images)
        n = len(images)
        n_train = int(n * SPLIT["train"])
        n_val = int(n * SPLIT["val"])

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, files in splits.items():
            out_dir = OUT_DIR / split / f"{country}_{denom_dir.name}"
            out_dir.mkdir(parents=True, exist_ok=True)

            for f in files:
                shutil.copy(f, out_dir / f.name)

def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for country_dir in RAW_DIR.iterdir():
        if country_dir.is_dir():
            prepare_country(country_dir)

    print("✅ Dataset prepared successfully!")

if __name__ == "__main__":
    main()