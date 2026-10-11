import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_PATH = "UTKFace"

IMG_SIZE = 96 
BATCH_SIZE = 32
EPOCHS = 10

# ==========================================
# LOAD DATASET
# ==========================================

images = []
ages = []
genders = []
races = []

print("Loading UTKFace Dataset...")

files = os.listdir(DATASET_PATH)

for file in files:

    try:

        parts = file.split("_")

        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])

        img_path = os.path.join(
            DATASET_PATH,
            file
        )

        img = Image.open(img_path)

        img = img.convert("RGB")

        img = img.resize(
            (IMG_SIZE, IMG_SIZE)
        )

        img = np.array(img)

        images.append(img)

        ages.append(age)

        genders.append(gender)

        races.append(race)

    except:
        continue

print(f"Images Loaded : {len(images)}")

# ==========================================
# CONVERT TO NUMPY
# ==========================================

X = np.array(images, dtype=np.float32)

X = X / 255.0

age_y = np.array(
    ages,
    dtype=np.float32
)

# Normalize age
age_y = age_y / 116.0

gender_y = np.array(
    genders,
    dtype=np.int32
)

race_y = np.array(
    races,
    dtype=np.int32
)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

(
    X_train,
    X_val,
    age_train,
    age_val,
    gender_train,
    gender_val,
    race_train,
    race_val

) = train_test_split(

    X,
    age_y,
    gender_y,
    race_y,

    test_size=0.2,
    random_state=42

)

print("Training Samples :", len(X_train))
print("Validation Samples :", len(X_val))

# ==========================================
# BUILD MODEL
# ==========================================

base_model = MobileNetV2(

    weights="imagenet",

    include_top=False,

    input_shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    )

)

# First Stage
base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.3)(x)

# AGE HEAD

age_head = Dense(
    256,
    activation="relu"
)(x)

age_head = Dropout(0.3)(age_head)

age_output = Dense(
    1,
    activation="linear",
    name="age"
)(age_head)

# GENDER HEAD

gender_head = Dense(
    128,
    activation="relu"
)(x)

gender_output = Dense(
    2,
    activation="softmax",
    name="gender"
)(gender_head)

# RACE HEAD

race_head = Dense(
    128,
    activation="relu"
)(x)

race_output = Dense(
    5,
    activation="softmax",
    name="race"
)(race_head)

model = Model(

    inputs=base_model.input,

    outputs=[
        age_output,
        gender_output,
        race_output
    ]

)

# ==========================================
# COMPILE
# ==========================================

model.compile(

    optimizer=Adam(
        learning_rate=0.001
    ),

    loss={

        "age": "mse",

        "gender":
        "sparse_categorical_crossentropy",

        "race":
        "sparse_categorical_crossentropy"

    },

    metrics={

        "age": ["mae"],

        "gender": ["accuracy"],

        "race": ["accuracy"]

    }

)

model.summary()

# ==========================================
# CALLBACKS
# ==========================================

checkpoint = ModelCheckpoint(

    "best_model.keras",

    monitor="val_loss",

    save_best_only=True,

    verbose=1

)

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)

# ==========================================
# TRAIN
# ==========================================

history = model.fit(

    X_train,

    {

        "age": age_train,

        "gender": gender_train,

        "race": race_train

    },

    validation_data=(

        X_val,

        {

            "age": age_val,

            "gender": gender_val,

            "race": race_val

        }

    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        checkpoint,
        early_stop
    ]

)

# ==========================================
# FINE TUNING
# ==========================================

print("\nStarting Fine Tuning...\n")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(

    optimizer=Adam(
        learning_rate=1e-5
    ),

    loss={

        "age": "mse",

        "gender":
        "sparse_categorical_crossentropy",

        "race":
        "sparse_categorical_crossentropy"

    },

    metrics={

        "age": ["mae"],

        "gender": ["accuracy"],

        "race": ["accuracy"]

    }

)

history_fine = model.fit(

    X_train,

    {

        "age": age_train,

        "gender": gender_train,

        "race": race_train

    },

    validation_data=(

        X_val,

        {

            "age": age_val,

            "gender": gender_val,

            "race": race_val

        }

    ),

    epochs=10,

    batch_size=BATCH_SIZE

)

# ==========================================
# SAVE MODEL
# ==========================================

model.save(
    "age_gender_ethnicity.keras"
)

print("\nModel Saved Successfully")

# ==========================================
# PLOTS
# ==========================================

os.makedirs(
    "static",
    exist_ok=True
)

# Gender Accuracy

plt.figure(figsize=(8,5))

plt.plot(
    history.history["gender_accuracy"]
)

plt.plot(
    history.history["val_gender_accuracy"]
)

plt.title(
    "Gender Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    "Train",
    "Validation"
])

plt.savefig(
    "static/gender_accuracy.png"
)

plt.close()

# Race Accuracy

plt.figure(figsize=(8,5))

plt.plot(
    history.history["race_accuracy"]
)

plt.plot(
    history.history["val_race_accuracy"]
)

plt.title(
    "Ethnicity Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    "Train",
    "Validation"
])

plt.savefig(
    "static/race_accuracy.png"
)

plt.close()

# Age MAE

plt.figure(figsize=(8,5))

plt.plot(
    history.history["age_mae"]
)

plt.plot(
    history.history["val_age_mae"]
)

plt.title(
    "Age Mean Absolute Error"
)

plt.xlabel("Epoch")
plt.ylabel("MAE")

plt.legend([
    "Train",
    "Validation"
])

plt.savefig(
    "static/age_mae.png"
)

plt.close()

print("\nGraphs Saved Inside static/")