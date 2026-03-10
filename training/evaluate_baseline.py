import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ======================
# CONFIG
# ======================
DATA_DIR = Path("dataset/processed")
# MODEL_PATH = Path("models/baseline_m0pp_tuned.keras")
MODEL_PATH = Path("models/mobilenetv2_transfer.keras")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# ======================
# LOAD TEST DATA
# ======================
use_mobilenet_preprocess = "mobilenetv2" in MODEL_PATH.name.lower()
if use_mobilenet_preprocess:
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
else:
    test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    DATA_DIR / "test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_gen.class_indices.keys())
num_classes = len(class_names)

# ======================
# PREDICT
# ======================
y_prob = model.predict(test_gen)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_gen.classes

# ======================
# CONFUSION MATRIX
# ======================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,10))
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(xticks_rotation=90)
plt.title("Confusion Matrix - Baseline (M0++)")
plt.show()

# ======================
# CLASSIFICATION REPORT
# ======================
print("\n📊 Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

# ======================
# ROC + AUC (MACRO)
# ======================
y_true_bin = label_binarize(y_true, classes=range(num_classes))

fpr = dict()
tpr = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])

# Macro-average
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)

for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= num_classes
macro_auc = auc(all_fpr, mean_tpr)

plt.figure(figsize=(7,6))
plt.plot(all_fpr, mean_tpr, label=f"Macro ROC (AUC = {macro_auc:.3f})")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Macro-average) - Baseline")
plt.legend()
plt.show()

print("✅ Macro AUC:", macro_auc)
