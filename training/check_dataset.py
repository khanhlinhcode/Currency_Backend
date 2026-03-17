import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
import cv2

DATA_DIR = Path("dataset/processed/train")
IMG_SIZE = (224, 224)
COLS = 5

# ======================
# GROUP CLASSES
# ======================
classes = [c for c in os.listdir(DATA_DIR) if (DATA_DIR / c).is_dir()]

vn_classes  = [c for c in classes if c.startswith("VN")]
usa_classes = [c for c in classes if c.startswith("USA")]
inr_classes = [c for c in classes if c.startswith("INR")]

# 👉 lấy mỗi nhóm 4 class
selected_classes = (
    sorted(vn_classes)[:4] +
    sorted(usa_classes)[:4] +
    sorted(inr_classes)[:4]
)

print("Selected:", selected_classes)

rows = (len(selected_classes) // COLS) + 1

plt.figure(figsize=(COLS * 3, rows * 3))

plot_index = 1

# ======================
# LOAD IMAGE
# ======================
for cls in selected_classes:
    class_path = DATA_DIR / cls
    images = [img for img in os.listdir(class_path) if not img.startswith('.')]

    if not images:
        continue

    img_path = class_path / random.choice(images)
    img = cv2.imread(str(img_path))

    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    plt.subplot(rows, COLS, plot_index)
    plt.imshow(img)
    plt.title(cls, fontsize=9)
    plt.axis("off")

    plot_index += 1

# ======================
# SAVE
# ======================
plt.tight_layout()
plt.savefig("dataset_balanced.png", dpi=300)
plt.show()

print("✅ Saved: dataset_balanced.png")