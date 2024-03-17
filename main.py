from plotting_features import plot_distributions, plot_scatter_matrices
from feature_selection import compute_scatter_matrices, compute_trace_ratio, forward_search, backward_search
from geometric_features import compute_eigen, compute_geometric_features, extract_geometric_features
from noteigen_features import extract_geometric_features_not_eigen

# get all the feature and label arrays
features = extract_geometric_features()[0]
features.extend(extract_geometric_features_not_eigen())
labels = extract_geometric_features()[1]
feature_names = ['Sum', 'Omnivariance', 'Eigenentropy', 'Linearity',
                 'Planarity', 'Sphericity', 'Anisotropy', 'Relative Height', 'Area Bounding Box',
                 'Area Oriented Bounding Box', 'Aspect Ratio', 'Density m^2', 'Density m^3', 'Area Convex Hull',
                 'Ratio Convex Hull - Oriented Bounding Box', 'Distance Z', 'Distance XY']

# plot the distributions of the features
plot_distributions(features, labels, feature_names)

# compute within and between scatter matrices
Sw, Sb = compute_scatter_matrices(features, labels)

# plot the within and between scatter matrices
plot_scatter_matrices(feature_names, Sw, Sb)

# get 4 best features based on forward search
forward_features_names = forward_search(feature_names, features, labels, 4)
print(forward_features_names)

# get 4 best features based on backward search
backward_features_names = backward_search(feature_names, features, labels, 4)
print(backward_features_names)

