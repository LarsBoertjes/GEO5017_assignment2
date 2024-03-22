import numpy as np


def compute_scatter_matrices(features, labels):
    """
    Computes the within-class scatter matrix (Sw) and the between-class scatter matrix (Sb)
    for a given set of features and labels
    """

    # create matrix with each row a sample and each column a feature
    feature_matrix = np.array(features)
    num_features = feature_matrix.shape[1]

    # get unique classes and their indices
    classes, label_indices = np.unique(labels, return_inverse=True)
    num_classes = len(classes)

    # initialize the scatter matrices
    Sw = np.zeros((num_features, num_features))
    Sb = np.zeros((num_features, num_features))

    overall_mean = np.mean(feature_matrix, axis=0)

    # fill the scatter matrices
    for i, class_label in enumerate(classes):
        class_indices = (label_indices == i)
        class_feature_matrix = feature_matrix[class_indices, :]
        class_mean = np.mean(class_feature_matrix, axis=0)
        class_cov = np.cov(class_feature_matrix, rowvar=False)
        num_samples_in_class = class_feature_matrix.shape[0]

        # within-class scatter matrix
        Sw += class_cov * num_samples_in_class

        # between-class scatter matrix
        mean_diff = (class_mean - overall_mean).reshape(num_features, 1)
        Sb += num_samples_in_class * (mean_diff @ mean_diff.T)

    Sw /= len(labels)

    return Sw, Sb


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
            Sw, Sb = compute_scatter_matrices(candidate_feature_matrix, labels)
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
            Sw, Sb = compute_scatter_matrices(test_set_features, labels)
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
