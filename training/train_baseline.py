# import tensorflow as tf
# from pathlib import Path
# import matplotlib.pyplot as plt
# import numpy as np

# from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping

# # ======================
# # CẤU HÌNH
# # ======================
# DATA_DIR = Path("dataset/processed")
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32
# EPOCHS = 20
# LR = 1e-4

# train_dir = DATA_DIR / "train"
# val_dir   = DATA_DIR / "val"
# test_dir  = DATA_DIR / "test"

# # ======================
# # DATA AUGMENTATION
# # ======================
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=15,
#     zoom_range=0.15,
#     brightness_range=[0.7, 1.3],
#     horizontal_flip=True
# )

# val_datagen = ImageDataGenerator(rescale=1./255)

# train_gen = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="categorical"
# )

# val_gen = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="categorical"
# )

# num_classes = train_gen.num_classes
# print("Number of classes:", num_classes)
# print("Class mapping:", train_gen.class_indices)

# # ======================
# # MODEL M0++
# # ======================
# model = models.Sequential([
#     layers.Input(shape=(224,224,3)),

#     layers.Conv2D(32, 3, padding="same"),
#     layers.BatchNormalization(),
#     layers.Activation("relu"),
#     layers.MaxPooling2D(),

#     layers.Conv2D(64, 3, padding="same"),
#     layers.BatchNormalization(),
#     layers.Activation("relu"),
#     layers.MaxPooling2D(),

#     layers.Conv2D(128, 3, padding="same"),
#     layers.BatchNormalization(),
#     layers.Activation("relu"),
#     layers.MaxPooling2D(),

#     layers.Conv2D(256, 3, padding="same"),
#     layers.BatchNormalization(),
#     layers.Activation("relu"),
#     layers.MaxPooling2D(),

#     layers.GlobalAveragePooling2D(),

#     layers.Dense(256, activation="relu"),
#     layers.Dropout(0.4),

#     layers.Dense(num_classes, activation="softmax")
# ])

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # ======================
# # CALLBACKS
# # ======================
# early_stop = EarlyStopping(
#     monitor="val_loss",
#     patience=4,
#     restore_best_weights=True
# )

# # ======================
# # TRAIN
# # ======================
# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=EPOCHS,
#     callbacks=[early_stop]
# )

# # ======================
# # TEST
# # ======================
# test_gen = val_datagen.flow_from_directory(
#     test_dir,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="categorical",
#     shuffle=False
# )

# test_loss, test_acc = model.evaluate(test_gen)
# print("✅ Test accuracy:", test_acc)

# # ======================
# # PLOT ACC & LOSS
# # ======================
# plt.figure(figsize=(12,5))

# plt.subplot(1,2,1)
# plt.plot(history.history["accuracy"], label="Train Accuracy")
# plt.plot(history.history["val_accuracy"], label="Val Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Accuracy Curve")
# plt.legend()

# plt.subplot(1,2,2)
# plt.plot(history.history["loss"], label="Train Loss")
# plt.plot(history.history["val_loss"], label="Val Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Loss Curve")
# plt.legend()

# plt.tight_layout()
# plt.show()

# # ======================
# # SAVE MODEL (FORMAT MỚI)
# # ======================
# Path("models").mkdir(exist_ok=True)
# model.save("models/baseline_m0pp.keras")
# print("✅ Model saved to models/baseline_m0pp.keras")
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ======================
# CONFIG
# ======================
DATA_DIR = Path("dataset/processed")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LR = 5e-4  # start a bit higher, will reduce automatically

train_dir = DATA_DIR / "train"
val_dir   = DATA_DIR / "val"
test_dir  = DATA_DIR / "test"

# ======================
# AUGMENTATION (nhẹ, ổn định)
# ======================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

num_classes = train_gen.num_classes
print("Number of classes:", num_classes)
print("Class mapping:", train_gen.class_indices)

# ======================
# MODEL M0++ (tuned nhẹ)
# ======================
l2 = regularizers.l2(1e-4)

model = models.Sequential([
    layers.Input(shape=(224,224,3)),

    layers.Conv2D(32, 3, padding="same", kernel_regularizer=l2),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", kernel_regularizer=l2),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", kernel_regularizer=l2),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, padding="same", kernel_regularizer=l2),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(256, activation="relu", kernel_regularizer=l2),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# CALLBACKS
# ======================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# ======================
# TRAIN
# ======================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
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
print("✅ Test accuracy:", test_acc)

# ======================
# SAVE MODEL
# ======================
Path("models").mkdir(exist_ok=True)
model.save("models/baseline_m0pp_tuned.keras")
print("✅ Model saved to models/baseline_m0pp_tuned.keras")