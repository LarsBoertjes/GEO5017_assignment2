from plotting_features import plot_distributions, plot_scatter_matrices
from feature_selection import compute_scatter_matrices, compute_trace_ratio, forward_search, backward_search
from extracting_features import feature_extraction

# get all the feature and label arrays
feature_extraction('data')

feature_names = ['Sum', 'Omnivariance', 'Eigenentropy', 'Linearity',
                 'Planarity', 'Sphericity', 'Anisotropy', 'Relative Height', 'Area Bounding Box',
                 'Area Oriented Bounding Box', 'Aspect Ratio', 'Density m^2', 'Density m^3', 'Area Convex Hull',
                 'Shape Index', 'Distance Z', 'Distance XY']

# plot the distributions of the features
# plot_distributions(features, labels, feature_names)
#
# # save the features to a txt file for further processing
#
# # normalize the features
#
# # compute within and between scatter matrices
# Sw, Sb = compute_scatter_matrices(features, labels)
#
# # plot the within and between scatter matrices
# plot_scatter_matrices(feature_names, Sw, Sb)
#
# # get 4 best features based on forward search
# forward_features_names = forward_search(feature_names, features, labels, 4)
# print(forward_features_names)
#
# # get 4 best features based on backward search
# backward_features_names = backward_search(feature_names, features, labels, 4)
# print(backward_features_names)
#
