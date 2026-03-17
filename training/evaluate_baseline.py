import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score
)
from sklearn.preprocessing import label_binarize

# ======================
# CONFIG
# ======================
DATA_DIR = Path("dataset/processed")
MODEL_PATH = Path("models/mobilenetv2_transfer.keras")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# ======================
# AUTO PREPROCESS BY MODEL NAME
# ======================
model_name = MODEL_PATH.name.lower()

if "mobilenet" in model_name:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    PREPROCESS_NAME = "mobilenet_v2.preprocess_input"
elif "efficientnetv2" in model_name:
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    PREPROCESS_NAME = "efficientnet_v2.preprocess_input"
elif "efficientnet" in model_name:
    from tensorflow.keras.applications.efficientnet import preprocess_input
    PREPROCESS_NAME = "efficientnet.preprocess_input"
else:
    preprocess_input = None
    PREPROCESS_NAME = "rescale_1_over_255"

print(f"🔧 Using preprocess: {PREPROCESS_NAME}")

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")

# ======================
# GENERATOR FACTORY
# ======================
if preprocess_input is None:
    datagen = ImageDataGenerator(rescale=1. / 255)
else:
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def make_gen(split: str, shuffle: bool):
    return datagen.flow_from_directory(
        DATA_DIR / split,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=shuffle,
        seed=SEED
    )

train_gen = make_gen("train", shuffle=False)
val_gen   = make_gen("val", shuffle=False)
test_gen  = make_gen("test", shuffle=False)

class_indices = test_gen.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

# ======================
# EVALUATE TRAIN / VAL / TEST
# ======================
train_loss, train_acc = model.evaluate(train_gen, verbose=1)
val_loss, val_acc = model.evaluate(val_gen, verbose=1)
test_loss, test_acc = model.evaluate(test_gen, verbose=1)

print("\n===== OVERALL METRICS =====")
print(f"Train accuracy: {train_acc:.4f} | loss: {train_loss:.4f}")
print(f"Val   accuracy: {val_acc:.4f} | loss: {val_loss:.4f}")
print(f"Test  accuracy: {test_acc:.4f} | loss: {test_loss:.4f}")

# ======================
# PREDICTION ON TEST
# ======================
y_prob = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_gen.classes

print("\nUnique y_pred:", np.unique(y_pred, return_counts=True))
print("Unique y_true:", np.unique(y_true, return_counts=True))
print("Acc from sklearn:", accuracy_score(y_true, y_pred))

# ======================
# CONFUSION MATRIX (TEST)
# ======================
cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="viridis",
    xticklabels=class_names, yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Test)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULT_DIR / "confusion_matrix_test.png", dpi=250)
plt.show()

# ======================
# NORMALIZED CONFUSION MATRIX
# ======================
cm_sum = cm.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm_norm, annot=False, cmap="Blues", vmin=0, vmax=1,
    xticklabels=class_names, yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix (Test)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULT_DIR / "confusion_matrix_test_normalized.png", dpi=250)
plt.show()

# ======================
# CLASSIFICATION REPORT (DETAIL)
# ======================
print("\n📊 Classification Report (Test)\n")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# ======================
# ACCURACY PER CLASS
# ======================
class_acc = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros(num_classes), where=cm.sum(axis=1) != 0)

plt.figure(figsize=(14, 5))
plt.bar(class_names, class_acc)
plt.xticks(rotation=90)
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Accuracy per Class (Test)")
plt.tight_layout()
plt.savefig(RESULT_DIR / "accuracy_per_class_test.png", dpi=250)
plt.show()

# ======================
# ROC CURVE (MACRO)
# ======================
y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
fpr, tpr = {}, {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)

for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= num_classes
macro_auc = auc(all_fpr, mean_tpr)

plt.figure(figsize=(7, 6))
plt.plot(all_fpr, mean_tpr, label=f"Macro ROC (AUC = {macro_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Macro Average - Test)")
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "roc_curve_macro_test.png", dpi=250)
plt.show()

print(f"✅ Macro AUC (test): {macro_auc:.4f}")

# ======================
# EXAMPLE PREDICTIONS (HIỂN THỊ ẢNH GỐC)
# ======================
sample_paths = test_gen.filepaths[:6]
sample_labels = test_gen.classes[:6]

plt.figure(figsize=(12, 8))
for i, fp in enumerate(sample_paths):
    img_pil = load_img(fp, target_size=IMG_SIZE)
    img_arr = img_to_array(img_pil).astype("float32")

    x = np.expand_dims(img_arr.copy(), axis=0)
    if preprocess_input is None:
        x = x / 255.0
    else:
        x = preprocess_input(x)

    pred = model.predict(x, verbose=0)[0]
    pred_class = np.argmax(pred)
    true_class = sample_labels[i]

    plt.subplot(2, 3, i + 1)
    plt.imshow(img_arr.astype("uint8"))
    plt.title(f"T: {class_names[true_class]}\nP: {class_names[pred_class]}")
    plt.axis("off")

plt.suptitle("Example Predictions (Test)")
plt.tight_layout()
plt.savefig(RESULT_DIR / "example_predictions_test.png", dpi=250)
plt.show()

print("\n✅ Evaluation completed")
print("📁 All charts saved to:", RESULT_DIR.resolve())