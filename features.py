import numpy as np
import compas
from sklearn.decomposition import PCA

with open('data/005.xyz', 'r') as xyz:
    lines = xyz.readlines()
    pc = np.array([line.split() for line in lines], dtype=float)

print(pc)
# minimum oriented bounding box
# obb = compas.geometry.oriented_bounding_box_numpy(coords)

# height
height = np.max(pc[:, 2])
print(height)

# height_spread
height_object = np.max(pc[:, 2]) - np.min(pc[:, 2])
print(height_object)

# PCA
# pca = PCA(n_components=3)
# pca.fit(pc)
#
# eigenvalues = pca.explained_variance_
#
# eigenvalues_sorted = np.sort(eigenvalues)[::-1]
#
# linearity = eigenvalues_sorted[0] / np.sum(eigenvalues_sorted)
#
# print("linearity:", linearity)