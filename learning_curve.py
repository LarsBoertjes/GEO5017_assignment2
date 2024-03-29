import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def learning_curve(X, y, classifier, check_interval=0.05, num_experiments=40, scaled=False):
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
        scaler = StandardScaler()

        # Lists to store errors for each experiment
        apparent_errors_exp = []
        true_errors_exp = []

        for _ in range(num_experiments):
            # Data splitting
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size)
            if scaled:
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Training & predicting
            classifier.fit(X_train, y_train)
            train_pred = classifier.predict(X_train)
            test_pred = classifier.predict(X_test)

            # Apparent error calculation
            apparent_error = 1 - accuracy_score(y_train, train_pred)
            apparent_errors_exp.append(apparent_error)

            # True error calculation
            true_error = 1 - accuracy_score(y_test, test_pred)
            true_errors_exp.append(true_error)

        # Store errors for this training set size
        apparent_errors.append(apparent_errors_exp)
        true_errors.append(true_errors_exp)

    return training_examples, apparent_errors, true_errors


def plot_learning_curve(classifier_name, training_examples, apparent_errors, true_errors):
    """ Plots the learning curve of a classifier.
            classifier_name: Name of the classifier (string)
            training_examples: Amount of training examples for each tested interval
            apparent_errors: training set errors of all experiments
            true_errors: approximated true error rate from the testing set"""
    plt.figure(figsize=(10, 6))

    # Calculate min and max errors
    min_apparent_errors = [np.min(errors) for errors in apparent_errors]
    max_apparent_errors = [np.max(errors) for errors in apparent_errors]
    min_true_errors = [np.min(errors) for errors in true_errors]
    max_true_errors = [np.max(errors) for errors in true_errors]

    # Plotting the filled areas
    plt.fill_between(training_examples, min_apparent_errors, max_apparent_errors, color='orange', alpha=0.1,
                     label='Apparent Error Range')
    plt.fill_between(training_examples, min_true_errors, max_true_errors, color='blue', alpha=0.1,
                     label='True Error Range')

    # Plotting average values
    avg_apparent_errors = [np.mean(errors) for errors in apparent_errors]
    avg_true_errors = [np.mean(errors) for errors in true_errors]
    plt.plot(training_examples, avg_apparent_errors, label='Average Apparent Error', color='orange', marker='o')
    plt.plot(training_examples, avg_true_errors, label='Average True Error', color='blue', marker='o')

    plt.xlabel('Training Set Size')
    plt.ylabel('Error Score')
    plt.title(f'{classifier_name} Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 0.45)
    plt.show()
