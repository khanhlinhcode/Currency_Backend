import tensorflow as tf
from pathlib import Path
import numpy as np

from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# ======================
# REPRODUCIBILITY
# ======================
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# ======================
# CONFIG
# ======================
DATA_DIR = Path("dataset/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 35

# Phase 1 nên cao hơn để head học nhanh
LR_PHASE1 = 3e-4
LR_PHASE2 = 1e-5

train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"
test_dir = DATA_DIR / "test"

# ======================
# DATA GENERATORS
# ======================
# preprocess_input dùng 1 lần duy nhất ở generator
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_gen = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

test_gen = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

num_classes = train_gen.num_classes

print("Classes:", train_gen.class_indices)
print("Train samples:", train_gen.samples)
print("Validation samples:", val_gen.samples)
print("Test samples:", test_gen.samples)
print("Number of classes:", num_classes)

# ======================
# CLASS WEIGHT (chống lệch lớp)
# ======================
classes = np.unique(train_gen.classes)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_gen.classes
)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
print("Class weights:", class_weight)

# ======================
# BACKBONE
# ======================
base_model = EfficientNetV2B0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    include_preprocessing=False
)
base_model.trainable = False

# ======================
# CLASSIFIER HEAD
# ======================
inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# ======================
# CALLBACKS
# ======================
best_path = MODEL_DIR / "efficientnetv2b0_best.keras"

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=str(best_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# ======================
# PHASE 1: Train head
# ======================
print("\n🚀 PHASE 1: Train classifier head")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_PHASE1),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# ======================
# PHASE 2: Fine tune
# ======================
print("\n🔥 PHASE 2: Fine tuning EfficientNetV2B0")
base_model.trainable = True

# Freeze phần đầu, train phần cuối
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Freeze BN để fine-tune ổn định
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_PHASE2),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# ======================
# TEST
# ======================
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"\n✅ EfficientNetV2B0 test accuracy: {test_acc:.4f}")
print(f"✅ EfficientNetV2B0 test loss: {test_loss:.4f}")

# ======================
# SAVE
# ======================
final_path = MODEL_DIR / "efficientnetv2b0_transfer.keras"
model.save(final_path)

print(f"✅ Final model saved: {final_path}")
print(f"✅ Best model checkpoint: {best_path}")