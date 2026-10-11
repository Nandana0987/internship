import cv2
import os
import numpy as np

path = "dataset/admin"

features = []

for img_name in os.listdir(path):
    img_path = os.path.join(path, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # ⭐ important fix

    features.append(img.flatten())

model = np.mean(features, axis=0)

np.save("model/admin_model.npy", model)

print("Model trained successfully")