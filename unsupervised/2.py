import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load image
image = cv2.imread("unsupervised/image.jpg")



if image is None:
    print("Image not found!")
    exit()

# Convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape into pixel feature vectors
pixels = image.reshape((-1, 3))

# Apply K-Means
k = 7  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(pixels)

# Create segmented image
centers = np.uint8(kmeans.cluster_centers_)
segmented = centers[labels]
segmented = segmented.reshape(image.shape)

# Show results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmented)
plt.title("Clustered Image")
plt.axis("off")

plt.show()