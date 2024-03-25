from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sklearn.model_selection as model_selection
from extracting_features import data_loading
from writing_hyperparameters import write_hyperparameters_to_file
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ID, X_all, y = data_loading()
X = X_all[:, [2, 5, 6, 7]]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.60,
                                                                    test_size=0.40, random_state=101)

def hyper_parameter_tuning(X_train, X_test, y_train, y_test):
    # hyperparameters ranges
    n_estimators = [1, 2, 5, 10, 25, 50, 75, 100, 125, 150]
    criterion = ["gini", "entropy", "log_loss"]
    max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bootstrap = [True, False]
    max_samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_samples_split = [2, 5, 10, 20, 50, 100]

    # get the best hyperparameters for first 4
    best_score_set1 = 0
    best_params_set1 = {}

    for n in n_estimators:
        for crit in criterion:
            for max_feat in max_features:
                for boot in bootstrap:
                    RF = RandomForestClassifier(n_estimators=n, criterion=crit, max_features=max_feat,
                                                bootstrap=boot, random_state=0)
                    RF.fit(X_train, y_train)
                    predictions = RF.predict(X_test)
                    score = accuracy_score(y_test, predictions)

                    if score > best_score_set1:
                        best_score_set1 = score
                        best_params_set1 = {'n_estimators': n, 'criterion': crit, 'max_features': max_feat,
                                       'bootstrap': boot}

    print("Best score:", best_score_set1)
    print("Best parameters:", best_params_set1)

    # take the first 4 parameters and tune the other three as well
    best_score_set2 = 0
    best_params_set2 = {}

    for max_sample in max_samples:
        for depth in max_depth:
            for min_samples in min_samples_split:
                RF = RandomForestClassifier(n_estimators=best_params_set1['n_estimators'], criterion=best_params_set1['criterion'],
                                            max_features=best_params_set1['max_features'], bootstrap=best_params_set1['bootstrap'],
                                            max_samples=max_sample, max_depth=depth, min_samples_split=min_samples,
                                            random_state=0)
                RF.fit(X_train, y_train)
                predictions = RF.predict(X_test)
                score = accuracy_score(y_test, predictions)

                if score > best_score_set2:
                    best_score_set2 = score
                    best_params_set2 = {'max_samples': max_sample, 'max_depth': depth,
                                        'min_samples_split': min_samples}

    print("Best score: ", best_score_set2)
    print("Best parameters:", best_params_set2)

    best_hyperparameters = {**best_params_set1, **best_params_set2}

    return best_hyperparameters

def max_depth_max_samples(X_train, X_test, y_train, y_test):
    # this function shows how max_depth and max_samples operate
    # the parameters used
    max_depth = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    max_samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Initialize matrices to store scores
    accuracy_matrix = np.zeros((len(max_depth), len(max_samples)))
    f1_matrix = np.zeros((len(max_depth), len(max_samples)))

    for i, depth in enumerate(max_depth):
        for j, samples in enumerate(max_samples):
            RF = RandomForestClassifier(n_estimators=50, criterion='gini', max_features=0.1,
                                        bootstrap=True, max_samples=samples, max_depth=depth,
                                        min_samples_split=5)
            RF.fit(X_train, y_train)
            predictions = RF.predict(X_test)

            accuracy_matrix[i, j] = accuracy_score(y_test, predictions)
            f1_matrix[i, j] = f1_score(y_test, predictions, average='weighted')

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy Heatmap
    sns.heatmap(accuracy_matrix, ax=axs[0], annot=True, fmt=".2f", cmap='viridis',
                xticklabels=max_samples, yticklabels=max_depth)
    axs[0].set_title('Accuracy Score')
    axs[0].set_xlabel('max_samples')
    axs[0].set_ylabel('max_depth')

    # F1 Score Heatmap
    sns.heatmap(f1_matrix, ax=axs[1], annot=True, fmt=".2f", cmap='viridis',
                xticklabels=max_samples, yticklabels=max_depth)
    axs[1].set_title('F1 Score')
    axs[1].set_xlabel('max_samples')
    axs[1].set_ylabel('max_depth')

    plt.tight_layout()
    plt.show()


def RF(X_train, X_test, y_train, y_test):
    parameters = hyper_parameter_tuning(X_train, X_test, y_train, y_test)
    write_hyperparameters_to_file(parameters, 'RF_params')


def read_rf_hyperparameters_from_file(model, X_train, X_test, y_train, y_test):
    hyperparameters = {}

    if model == 'rf':
        if not exists('RF_params'):
            RF(X_train, X_test, y_train, y_test)
        file_path = 'Rf_params'


    with open(file_path, "r") as file:
        for line in file:
            key, value = line.strip().split(": ", 1)
            if key.lower() not in ['accuracy', 'f1']:
                try:
                    hyperparameters[key] = int(value)
                except ValueError:
                    try:
                        hyperparameters[key] = float(value)
                    except ValueError:
                        # Attempt to interpret the value as a boolean if it matches 'True' or 'False'
                        if value == 'True':
                            hyperparameters[key] = True
                        elif value == 'False':
                            hyperparameters[key] = False
                        else:
                            hyperparameters[key] = value
    return hyperparameters