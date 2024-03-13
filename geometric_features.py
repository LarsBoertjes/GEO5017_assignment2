import numpy as np
import math
from read_data import get_pointclouds, read_point_clouds
import matplotlib.pyplot as plt
from plotting_features import plot_distributions


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


def geometric_features(eigenvalues):
    """
    takes in eigenvalues and returns in following order geometric features:
    sum, omnivariance, eigenentropy, anisotropy, planarity, linearity, surface_variation, sphericity
    """
    e1, e2, e3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]

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

    anisotropy = (e1 - e3) / e1

    planarity = (e2 - e3) / e1

    linearity = (e1 - e2) / e1

    surface_variation = e3 / (e1 + e2 + e3)

    sphericity = e3 / e1

    return omnivariance, eigenentropy, anisotropy, planarity, linearity, surface_variation, sphericity


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
omnivariance, eigenentropy, anisotropy, planarity, linearity, surface_variation, sphericity = [], [], [], [], [], [], []

for eigenvalues in eigenvalues_pointclouds:
    geometricFeatures = geometric_features(eigenvalues)
    omnivariance.append(geometricFeatures[0])
    eigenentropy.append(geometricFeatures[1])
    anisotropy.append(geometricFeatures[2])
    planarity.append(geometricFeatures[3])
    linearity.append(geometricFeatures[4])
    surface_variation.append(geometricFeatures[5])
    sphericity.append(geometricFeatures[6])


# plot the feature distributions for each label
features = [omnivariance, eigenentropy, anisotropy, planarity, linearity, surface_variation, sphericity]
feature_names = ['Omnivariance', 'Eigenentropy', 'Anisotropy', 'Planarity', 'Linearity', 'Surface Variation', 'Sphericity']
plot_distributions(features, labels, feature_names)