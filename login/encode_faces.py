import cv2
import os
import numpy as np

path = "dataset/admin"

vectors = []

for img_name in os.listdir(path):
    img_path = os.path.join(path, img_name)

    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.resize(img, (80,80))  # smaller = faster
    vectors.append(img.flatten())

# 🔥 Create SINGLE average face vector
avg_face = np.mean(vectors, axis=0)

os.makedirs("encodings", exist_ok=True)

np.save("encodings/admin_face.npy", avg_face)

print("Fast encoding done!")