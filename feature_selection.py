import numpy as np


def compute_within_class_scatter_matrix(features, labels):
    class_labels = np.unique(labels)
    n_features = features.shape[1]

    # Initialize the within-class scatter matrix to zeros
    Sw = np.zeros((n_features, n_features))

    # Iterate over each class to compute the within-class scatter matrix
    for k in class_labels:
        # Extract the features for class k
        class_features = features[labels == k]

        Sk = np.cov(class_features, rowvar=False)

        weight_k = float(class_features.shape[0] / features.shape[0])

        Sw += weight_k * Sk

    return Sw


def compute_between_class_scatter_matrix(features, labels):
    class_labels = np.unique(labels)
    overall_mean = np.mean(features, axis=0)
    n_features = features.shape[1]
    Sb = np.zeros((n_features, n_features))
    N = features.shape[0]

    for k in class_labels:
        class_features = features[labels == k]
        mean_k = np.mean(class_features, axis=0)
        Nk = class_features.shape[0]
        weight_k = Nk / N
        mean_diff = (mean_k - overall_mean).reshape(n_features, 1)
        Sb += weight_k * (mean_diff @ mean_diff.T)

    return Sb

def compute_trace_ratio(Sw, Sb):
    trace_Sw = np.trace(Sw)
    trace_Sb = np.trace(Sb)
    return trace_Sb / trace_Sw


def forward_search(feature_names, features, labels, d):
    """
    Based on lecture notes 08, p7
    """
    current_set_indices = []
    current_set_names = []
    current_features = None

    # fill lists above until specified number of items
    while len(current_set_indices) < d:
        best_feature_index = None
        best_feature_name = None
        best_trace_ratio = -float('inf')

        # iterate over the range of features
        for i in range(features.shape[1]):
            if i in current_set_indices:
                continue

            # if index is not yet used, candidate matrix is a column of the values belonging to feature i
            # if the index is used, we look to this one and the next
            if not current_set_indices:
                candidate_feature_matrix = features[:, [i]]
            else:
                candidate_feature_matrix = features[:, current_set_indices + [i]]

            # compute scatter matrices between candidate features
            # compute trace ratio
            Sw = compute_within_class_scatter_matrix(candidate_feature_matrix, labels)
            Sb = compute_between_class_scatter_matrix(candidate_feature_matrix, labels)
            trace_ratio = compute_trace_ratio(Sw, Sb)

            # adjust trace ratio
            if trace_ratio > best_trace_ratio:
                best_trace_ratio = trace_ratio
                best_feature_index = i
                best_feature_name = feature_names[i]

        # add best feature to the list
        if best_feature_index is not None:
            current_set_indices.append(best_feature_index)
            current_set_names.append(best_feature_name)

    current_features = features[:, current_set_indices]

    return current_set_names, current_features


def backward_search(feature_names, features, labels, d):
    """
    based on lecture notes 08, p7
    """
    current_set_indices = list(range(features.shape[1]))  # Start with all features
    current_set_names = feature_names.copy()
    current_features = features.copy()

    while len(current_set_indices) > d:
        worst_feature_index = None
        worst_feature_name = None
        best_trace_ratio = -float('inf')

        for i in current_set_indices:
            # Create a test feature matrix excluding the current feature
            test_set_indices = [idx for idx in current_set_indices if idx != i]
            test_set_features = features[:, test_set_indices]

            # Compute scatter matrices and the trace ratio for this test set
            Sw = compute_within_class_scatter_matrix(test_set_features, labels)
            Sb = compute_between_class_scatter_matrix(test_set_features, labels)
            trace_ratio = compute_trace_ratio(Sw, Sb)

            if trace_ratio > best_trace_ratio:
                best_trace_ratio = trace_ratio
                worst_feature_index = i
                worst_feature_name = feature_names[i]

        if worst_feature_index is not None:
            current_set_indices.remove(worst_feature_index)
            current_set_names.remove(worst_feature_name)

    current_features = features[:, current_set_indices]

    return current_set_names, current_features
