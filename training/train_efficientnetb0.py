import tensorflow as tf
from pathlib import Path
import numpy as np

from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ======================
# CONFIG
# ======================
DATA_DIR = Path("dataset/processed")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS_PHASE1 = 25
EPOCHS_PHASE2 = 10

LR_PHASE1 = 1e-4
LR_PHASE2 = 5e-6
SEED = 42

train_dir = DATA_DIR / "train"
val_dir   = DATA_DIR / "val"
test_dir  = DATA_DIR / "test"

# ======================
# DATA AUGMENTATION
# ======================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    seed=SEED
)

num_classes = train_gen.num_classes
print("Number of classes:", num_classes)

# ======================
# CLASS WEIGHT (IMPORTANT)
# ======================
class_counts = np.bincount(train_gen.classes)
class_weight = {
    i: float(class_counts.sum()) / (num_classes * class_counts[i])
    for i in range(num_classes)
}
print("Class weights:", class_weight)

# ======================
# EFFICIENTNET BACKBONE
# ======================
base_model = EfficientNetB0(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False   # PHASE 1 FREEZE

# ======================
# CLASSIFIER HEAD
# ======================
inputs = layers.Input(shape=(224,224,3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# ======================
# COMPILE – PHASE 1
# ======================
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE1),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# TRAIN – PHASE 1
# ======================
print("\n🚀 PHASE 1: Training head only")
callbacks_phase1 = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
]
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks_phase1,
    class_weight=class_weight
)

# ======================
# PHASE 2 – FINE TUNE
# ======================
print("\n🔥 PHASE 2: Fine-tuning EfficientNetB0")

base_model.trainable = True

# Freeze most layers (giữ 40 layer cuối trainable)
for layer in base_model.layers[:-40]:
    layer.trainable = False

# Keep BatchNorm frozen (ổn định với batch nhỏ)
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_PHASE2),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
]
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks_phase2,
    class_weight=class_weight
)

# ======================
# TEST
# ======================
test_gen = val_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_gen)
print("🔥 EfficientNetB0 Test accuracy:", test_acc)

Path("models").mkdir(exist_ok=True)
model.save("models/efficientnetb0_transfer.keras")
print("✅ Model saved to models/efficientnetb0_transfer.keras")
