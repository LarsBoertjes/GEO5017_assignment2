import numpy as np
from read_data import get_pointclouds, read_point_clouds

pointclouds, labels = read_point_clouds('data')

for pointcloud, label in zip(pointclouds, labels):
    print(f"size {len(pointcloud)}  has label: {label}")


""""centroid = np.median(points, axis=0)

cov_matrix = np.cov(points - centroid, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

sum = eigenvalues[0] + eigenvalues[1] + eigenvalues[2]
omnivariance = (eigenvalues[0] * eigenvalues[1] * eigenvalues[2]) ** (1/3)

planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]

print("Eigenvalues: ", eigenvalues)
print("Eigenvectors: ", eigenvectors)

print("Sum: ", sum)
print("Omnivariance: ", omnivariance)

print("Planarity: ", planarity)"""




