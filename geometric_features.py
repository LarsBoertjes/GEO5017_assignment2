import numpy as np
import math
from read_data import get_pointclouds, read_point_clouds
import matplotlib.pyplot as plt
from plotting_features import plot_distributions
from feature_selection import compute_scatter_matrices, compute_trace_ratio


def compute_eigen(pointcloud):
    """" Computes the eigenvalues for a pointcloud based on the covariance matrix of the distribution of the points
    around the median point
    """

    # get centroid
    centroid = np.median(pointcloud, axis=0)

    # computing of covariance
    # rowvar -> each column of the arrays corresponds to dimension
    cov_matrix = np.cov(pointcloud - centroid, rowvar=False)

    # compute eigenvalues & eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # sorting the eigenvalues & vectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def geometric_features(eigenvalues, eigenvectors):
    """
    Input: eigenvalues of object

    takes in eigenvalues and returns in following order geometric features:
    sum, omnivariance, eigenentropy, linearity, planarity, sphericity,
    anisotropy, verticality
    """
    e1, e2, e3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]
    v1, v2, v3 = eigenvectors[0], eigenvectors[1], eigenvectors[2]
    ez = [0, 0, 1]

    _sum = e1 + e2 + e3

    omnivariance = (e1 * e2 * e3) ** (1/3)

    if _sum == 0:
        return 0
    else:
        eigenentropy = (-1 / math.log(3)) * (
            (e1 / _sum) * math.log(e1 / _sum) +
            (e2 / _sum) * math.log(e2 / _sum) +
            (e3 / _sum) * math.log(e3 / _sum)
        )

    linearity = (e1 - e2) / e1
    planarity = (e2 - e3) / e1
    sphericity = e3 / e1
    anisotropy = (e1 - e3) / e1
    verticality = 1 - abs(np.dot(ez, v3))

    return _sum, omnivariance, eigenentropy, linearity, planarity, sphericity, anisotropy, verticality


#  importing of pointcloud and labels array
pointclouds, labels = read_point_clouds('data')

# get eigenvalues, eigenvectors from pointcloud objects
eigenvalues_pointclouds = []
eigenvectors_pointclouds = []

for pointcloud in pointclouds:
    eigenvalues, eigenvectors = compute_eigen(pointcloud)
    eigenvalues_pointclouds.append(eigenvalues)
    eigenvectors_pointclouds.append(eigenvectors)

# compute geometric features
_sum, omnivariance, eigenentropy, linearity, planarity, sphericity, anisotropy, verticality = [], [], [], [], [], [], [], []

for eigenvalues in eigenvalues_pointclouds:
    geometricFeatures = geometric_features(eigenvalues, eigenvectors)
    _sum.append(geometricFeatures[0])
    omnivariance.append(geometricFeatures[1])
    eigenentropy.append(geometricFeatures[2])
    linearity.append(geometricFeatures[3])
    planarity.append(geometricFeatures[4])
    sphericity.append(geometricFeatures[5])
    anisotropy.append(geometricFeatures[6])
    verticality.append(geometricFeatures[7])


# plot the feature distributions for each label
features = [_sum, omnivariance, eigenentropy, linearity, planarity, sphericity, anisotropy]
feature_names = ['Sum', 'Omnivariance', 'Eigenentropy', 'Linearity',
                 'Planarity', 'Sphericity', 'Anisotropy']
plot_distributions(features, labels, feature_names)

# calculate within and between scatter matrices

Sw, Sb = compute_scatter_matrices(features, labels)

# calculate trace ratio
trace_ratio = compute_trace_ratio(Sw, Sb)
print(trace_ratio)
