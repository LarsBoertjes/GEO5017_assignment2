import numpy as np


# compute overlap matrix
def overlap_matrix(features):
    features_building = features[0:100]
    features_car = features[100:200]
    features_fence = features[200:300]
    features_pole = features[300:400]
    features_tree = features[400:500]

    classes = [features_building, features_car, features_fence, features_pole, features_tree]
    n_classes = len(classes)
    n_features = features.shape[1]

    # Initialize the overlap matrix
    overlap_matrix = np.zeros((n_classes, n_classes))

    # Calculate overlap for each pair of classes for each feature
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            total_overlap = 0
            for feature_index in range(n_features):
                feature_i = classes[i][:, feature_index]
                feature_j = classes[j][:, feature_index]

                # Calculate bin range for histogram
                bin_range = np.linspace(min(np.min(feature_i), np.min(feature_j)),
                                        max(np.max(feature_i), np.max(feature_j)), 20)

                # Calculate histograms for both distributions
                hist_i, _ = np.histogram(feature_i, bins=bin_range, density=True)
                hist_j, _ = np.histogram(feature_j, bins=bin_range, density=True)

                # Find the minimum of the two histograms to determine the overlap
                minima = np.minimum(hist_i, hist_j)

                # Calculate the sum of the minima to find the overlap
                overlap = np.sum(minima)

                # Accumulate the overlap for this feature
                total_overlap += overlap

            # Store the sum of overlaps for all features between classes i and j
            overlap_matrix[i, j] = total_overlap
            overlap_matrix[j, i] = total_overlap  # Symmetric matrix

    np.fill_diagonal(overlap_matrix, 1)

    return overlap_matrix
