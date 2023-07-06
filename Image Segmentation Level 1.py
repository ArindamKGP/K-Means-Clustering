from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread("/Users/arindamiitkgp/Desktop/KDAG/K Clustering Image Segmentation/Birla Planetarium.jpeg")
vectorized = img.reshape((-1,3))
kmeans = KMeans(n_clusters=5, random_state = 0, n_init=5).fit(vectorized)
centers = np.uint8(kmeans.cluster_centers_)
segmented_data = centers[kmeans.labels_.flatten()]
 
segmented_image = segmented_data.reshape((img.shape))
plt.imshow(segmented_image)
plt.title("K = 5")
plt.show()