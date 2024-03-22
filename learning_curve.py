# check_interval = 0.1 (can also be smaller or larger)
#   for i in range(1/ check_interval -1):
#   train test split ratio = (i+1)* check_interval
#   split the data accordingly train and test model on the corresponding sets (multiple times)
#   and record the (averaged) error rates
#   Plot the performances as curves

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from extracting_features import feature_extraction, data_loading
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit


def learning_curve(X, y, classifier, check_interval=0.05, num_experiments=10):
    # Range of training set sizes
    training_ratios = np.arange(check_interval, 1.0, check_interval)

    # Lists to store error rates
    training_examples = []
    apparent_errors = []
    true_errors = []

    # Iterating over different training sizes
    for train_size in training_ratios:
        test_size = 1.0 - train_size
        training_examples.append(int(len(X) * train_size))

        # Lists to store errors for each experiment
        apparent_errors_exp = []
        true_errors_exp = []


        for _ in range(num_experiments):
            # Data splitting
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size)

            # Training & predicting
            classifier.fit(X_train, y_train)
            train_pred = classifier.predict(X_train)
            test_pred = classifier.predict(X_test)

            # Apperent error calculation
            apparent_error = 1 - accuracy_score(y_train, train_pred)
            apparent_errors_exp.append(apparent_error)

            # True error calculation SVM
            true_error = 1 - accuracy_score(y_test, test_pred)
            true_errors_exp.append(true_error)

        # Calculate average errors for this training set size
        avg_apparent_error = np.mean(apparent_errors_exp)
        avg_true_error = np.mean(true_errors_exp)

        # Store average errors
        svm_apparent_errors.append(avg_apparent_error)
        svm_true_errors.append(avg_true_error)

    return training_examples, apparent_errors, true_errors

def plot_learning_curve(classifier, training_examples, apparent_errors, true_errors):
    plt.figure(figsize=(10, 6))
    plt.plot(training_examples, svm_apparent_errors, label='Apparent Error', marker='o')
    plt.plot(training_examples, svm_true_errors, label='True Error', marker='o')
    plt.xlabel('Training Set Size')
    plt.ylabel('Classification Error')
    plt.title(f'{classifier} Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


# Plotting function for Random Forest learning curve

ID, X_all, y = data_loading()
X = X_all[:, [2, 5, 6, 7]]

lin = svm.SVC(kernel='linear', C=1)
lin2 = svm.SVC(kernel='linear', C=10)

training_sizes, svm_apparent_errors, svm_true_errors = learning_curve(X, y, lin)
plot_learning_curve('SVM', training_sizes, svm_apparent_errors, svm_true_errors)




fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

common_params = {
    "X": X,
    "y": y,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}

for ax_idx, estimator in enumerate([lin, lin2]):
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
    handles, label = ax[ax_idx].get_legend_handles_labels()
    ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
    ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")

plt.show()
print(y)
