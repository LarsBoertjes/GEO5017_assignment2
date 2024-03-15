import numpy as np

def compute_scatter_matrices(features, labels):
    """
    Computes the within-class scatter matrix (Sw) and the between-class scatter matrix (Sb)
    for a given set of features and labels
    """

    # create matrix with each row a sample and each column a feature
    feature_matrix = np.column_stack(features)
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

