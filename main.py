from plotting_features import plot_distributions, plot_scatter_matrices, plot_normalized_confusion_matrix
from feature_selection import compute_scatter_matrices, compute_trace_ratio, forward_search, backward_search
from extracting_features import feature_extraction, data_loading
from read_data import read_hyperparameters_from_file
import sklearn.model_selection as model_selection
from sklearn.ensemble import RandomForestClassifier
from learning_curve import plot_learning_curve, learning_curve
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

# Get all the feature and label arrays
feature_extraction('data')

feature_names = ['Sum', 'Omnivariance', 'Eigenentropy', 'Linearity',
                 'Planarity', 'Sphericity', 'Anisotropy', 'Relative Height', 'Area Bounding Box',
                 'Area Oriented Bounding Box', 'Aspect Ratio', 'Density m^2', 'Density m^3', 'Area Convex Hull',
                 'Shape Index', 'Distance Z', 'Distance XY']

ID, X, y = data_loading()

# Compute within and between scatter matrices
Sw, Sb = compute_scatter_matrices(X, y)

# Get 4 best features based on forward search
forward_features_names, forward_features = forward_search(feature_names, X, y, 4)
print(f"Best features using forward search: ", forward_features_names)

# Get 4 best features based on backward search
backward_features_names, backward_features = backward_search(feature_names, X, y, 4)
print(f"Best features using backward search: ", backward_features_names)

# Split the data from the best features
# We use forward here, but backward has the same features
X_train, X_test, y_train, y_test = model_selection.train_test_split(forward_features, y, train_size=0.6, random_state=0)

# Use the best features as input for models
RF_params = read_hyperparameters_from_file('RF_params')
RF_model = RandomForestClassifier(**RF_params)
RF_model.fit(X_train, y_train)
RF_predictions = RF_model.predict(X_test)

# Plot the learning curves
# RF learning curve
training_sizes, RF_apparent_errors, RF_true_errors = learning_curve(forward_features, y, RF_model)
plot_learning_curve('Random Forest', training_sizes, RF_apparent_errors, RF_true_errors)

# Use the best two models to compare
RF_confusion_matrix = confusion_matrix(y_test, RF_predictions)
plot_normalized_confusion_matrix(RF_confusion_matrix)

# Plot distributions for best models
# This can be useful to explain misclassification between classes
plot_distributions(forward_features, y, forward_features_names)

