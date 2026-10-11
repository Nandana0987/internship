import tensorflow as tf
from tensorflow.keras import layers, models
import pickle

# ==========================
# SETTINGS
# ==========================
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 15

# ==========================
# LOAD DATASET
# ==========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names

print("\nEmployees Found:")
for employee in class_names:
    print(employee)

# Save employee names
with open("class_names.pkl", "wb") as f:
    pickle.dump(class_names, f)

# ==========================
# PERFORMANCE OPTIMIZATION
# ==========================
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)

# ==========================
# DATA AUGMENTATION
# ==========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# ==========================
# CNN MODEL
# ==========================
model = models.Sequential([

    data_augmentation,

    layers.Rescaling(1./255),

    layers.Conv2D(
        32,
        (3,3),
        activation="relu"
    ),
    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        (3,3),
        activation="relu"
    ),
    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        (3,3),
        activation="relu"
    ),
    layers.MaxPooling2D(),

    layers.Conv2D(
        256,
        (3,3),
        activation="relu"
    ),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

# ==========================
# COMPILE MODEL
# ==========================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# MODEL SUMMARY
# ==========================
model.summary()

# ==========================
# TRAIN MODEL
# ==========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ==========================
# SAVE MODEL
# ==========================
model.save("employee_model.h5")

print("\nModel Saved Successfully")
print("employee_model.h5")
print("class_names.pkl")