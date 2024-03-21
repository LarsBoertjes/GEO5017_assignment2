from plotting_features import plot_distributions, plot_scatter_matrices
from feature_selection import compute_scatter_matrices, compute_trace_ratio, forward_search, backward_search
from extracting_features import feature_extraction, data_loading

# get all the feature and label arrays
feature_extraction('data')

feature_names = ['Sum', 'Omnivariance', 'Eigenentropy', 'Linearity',
                 'Planarity', 'Sphericity', 'Anisotropy', 'Relative Height', 'Area Bounding Box',
                 'Area Oriented Bounding Box', 'Aspect Ratio', 'Density m^2', 'Density m^3', 'Area Convex Hull',
                 'Shape Index', 'Distance Z', 'Distance XY']

ID, X, y = data_loading()
print(X)

# plot the distributions of the features
# plot_distributions(X, y, feature_names)

# compute within and between scatter matrices
Sw, Sb = compute_scatter_matrices(X, y)

# plot the within and between scatter matrices
plot_scatter_matrices(feature_names, Sw, Sb)

# get 4 best features based on forward search
forward_features_names = forward_search(feature_names, X, y, 4)
print(forward_features_names)

# get 4 best features based on backward search
backward_features_names = backward_search(feature_names, X, y, 4)
print(backward_features_names)

# use the best features as input for models

