from plotting_features import (plot_distributions, plot_feature_space, plot_feature_space_2d,
                               plot_normalized_confusion_matrix, plot_overlap_matrix)
from sklearn import svm
from feature_selection import (forward_search, backward_search, compute_within_class_scatter_matrix,
                               compute_between_class_scatter_matrix)
from extracting_features import feature_extraction, data_loading
from SVM import read_svm_hyperparameters_from_file
from DF import read_rf_hyperparameters_from_file
import sklearn.model_selection as model_selection
from sklearn.ensemble import RandomForestClassifier
from learning_curve import plot_learning_curve, learning_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
from evaluation import overlap_matrix
from sklearn.preprocessing import StandardScaler


# Get all the feature and label arrays
feature_extraction('data.txt')

feature_names = ['Sum', 'Omnivariance', 'Eigenentropy', 'Linearity',
                 'Planarity', 'Sphericity', 'Anisotropy', 'Relative Height', 'Area Bounding Box',
                 'Area Oriented Bounding Box', 'Aspect Ratio', 'Density m^2', 'Density m^3', 'Area Convex Hull',
                 'Shape Index', 'Distance Z', 'Distance XY']

ID, X, y = data_loading()

# Compute within and between scatter matrices
Sw = compute_within_class_scatter_matrix(X, y)
Sb = compute_between_class_scatter_matrix(X, y)

# Get 4 best features based on forward search
forward_features_names, forward_features = forward_search(feature_names, X, y, 4)
print(f"Best features using forward search: ", forward_features_names)

# Get 4 best features based on backward search
backward_features_names, backward_features = backward_search(feature_names, X, y, 4)
print(f"Best features using backward search: ", backward_features_names)

# Split the data from the best features
# We use forward here, but backward has the same features
X_train, X_test, y_train, y_test = model_selection.train_test_split(forward_features, y,
                                                                    train_size=0.6, random_state=101)

# Use the best features as input for models
# SVM
SVM_params = read_svm_hyperparameters_from_file('svm', X_train, X_test, y_train, y_test)
SVM_model = svm.SVC(**SVM_params)

# Random Forest
RF_params = read_rf_hyperparameters_from_file('rf', X_train, X_test, y_train, y_test)
RF_model = RandomForestClassifier(**RF_params)
print("reading is done")

# Plot the learning curves
# SVM
SVM_training_sizes, SVM_apparent_errors, SVM_true_errors = learning_curve(forward_features, y, SVM_model, scaled=True)
plot_learning_curve('SVM', SVM_training_sizes, SVM_apparent_errors, SVM_true_errors)

# RF learning curve
RF_training_sizes, RF_apparent_errors, RF_true_errors = learning_curve(forward_features, y, RF_model)
plot_learning_curve('Random Forest', RF_training_sizes, RF_apparent_errors, RF_true_errors)

# Use the best two models to compare
# SVM
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

SVM_model.fit(X_train_std, y_train)
SVM_predictions = SVM_model.predict(X_test_std)

SVM_confusion_matrix = confusion_matrix(y_test, SVM_predictions)
plot_normalized_confusion_matrix(SVM_confusion_matrix, 'SVM')

# RF
RF_model.fit(X_train, y_train)
RF_predictions = RF_model.predict(X_test)

RF_confusion_matrix = confusion_matrix(y_test, RF_predictions)
plot_normalized_confusion_matrix(RF_confusion_matrix, 'Random Forest')

# Plot distributions for best models
# This can be useful to explain misclassification between classes
plot_distributions(forward_features, y, forward_features_names)
plot_feature_space(forward_features[:, :3], y, forward_features_names[:3])
plot_feature_space(forward_features[:, 1:], y, forward_features_names[1:])
# plot_feature_space_2d(forward_features[:, 2:], y, forward_features_names[2:])
# plot_feature_space_2d(forward_features[:, 1:3], y, forward_features_names[1:3])

# Overlap matrix to discuss confusion matrix results
overlap = overlap_matrix(backward_features)
plot_overlap_matrix(overlap, ['building', 'car', 'fence', 'pole', 'tree'])
