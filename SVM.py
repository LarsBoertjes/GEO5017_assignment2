from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from writing_hyperparameters import write_hyperparameters_to_file
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists


def hyper_parameter_tuning(X_train, X_test, y_train, y_test):
    best_lin_accuracy = 0.8
    best_poly_accuracy = 0.8
    best_rbf_accuracy = 0.8

    gamma_values = [0.02, 0.05, 1, 1.5, 2, 2.5]
    C_values_lin = [93, 95, 98, 100, 102]
    C_values = [10, 25, 50, 75, 100]
    coef0_values = [3, 3.5, 4, 4.5, 5]
    degree_values = [2, 3, 4, 5, 6]
    max_iter = [50000, 75000, 100000]

    for C in C_values_lin:
        for iter_val in max_iter:
            lin = svm.SVC(kernel='linear', max_iter=iter_val, C=C).fit(X_train, y_train)
            lin_pred = lin.predict(X_test)
            lin_accuracy = accuracy_score(y_test, lin_pred)
            lin_f1 = f1_score(y_test, lin_pred, average='weighted')

            if lin_accuracy > best_lin_accuracy:
                best_lin_accuracy = lin_accuracy
                best_hyper_parameters_lin = {'kernel': 'linear', 'C': C, 'max_iter': iter_val,
                                             'accuracy': lin_accuracy, 'f1': lin_f1}

    for C in C_values:
        for iter_val in max_iter:
            for gamma in gamma_values:
                rbf = svm.SVC(kernel='rbf', gamma=gamma, C=C, max_iter=iter_val).fit(X_train, y_train)
                rbf_pred = rbf.predict(X_test)
                rbf_accuracy = accuracy_score(y_test, rbf_pred)
                rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')

                if rbf_accuracy > best_rbf_accuracy:
                    best_rbf_accuracy = rbf_accuracy
                    best_hyper_parameters_rbf = {'kernel': 'rbf', 'C': C, 'gamma': gamma, 'max_iter': iter_val,
                                                 'accuracy': rbf_accuracy, 'f1': rbf_f1}

                for coef0 in coef0_values:
                    for degree in degree_values:
                        poly = svm.SVC(kernel='poly', degree=degree, gamma=gamma, coef0=coef0, C=C,
                                       max_iter=iter_val).fit(X_train, y_train)
                        poly_pred = poly.predict(X_test)
                        poly_accuracy = accuracy_score(y_test, poly_pred)
                        poly_f1 = f1_score(y_test, poly_pred, average='weighted')

                        if poly_accuracy > best_poly_accuracy:
                            best_poly_accuracy = poly_accuracy
                            best_hyper_parameters_poly = {'kernel': 'poly', 'C': C, 'gamma': gamma, 'coef0': coef0,
                                                          'degree': degree, 'max_iter': iter_val,
                                                          'accuracy': poly_accuracy, 'f1': poly_f1}

    return best_hyper_parameters_lin, best_hyper_parameters_rbf, best_hyper_parameters_poly


def create_heatmap(data_matrix, ax,  cmap, C_values, gamma_values, title=None, vmin=0.5, vmax=1):
    sns.heatmap(data_matrix, ax=ax, cmap=cmap, xticklabels=C_values, yticklabels=gamma_values, annot=True, fmt=".2f",
                vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=20, fontweight='bold')


def initial_test(X_train, X_test, y_train, y_test):
    # Initializing hyperparameters
    gamma_values = [0.001, 0.005, 0.01, 0.1, 1.0]
    C_values = [0.5, 1.0, 10.0, 50.0, 100.0]

    # Lists for outcomes
    rbf_accuracy_values = []
    rbf_f1_values = []
    poly_accuracy_values = []
    poly_f1_values = []
    lin_accuracy_values = []
    lin_f1_values = []
    sig_accuracy_values = []
    sig_f1_values = []

    # Initial grid search
    for gamma in gamma_values:
        for C in C_values:
            rbf = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(X_train, y_train)
            poly = svm.SVC(kernel='poly', degree=2, gamma=gamma, coef0=1, C=C).fit(X_train, y_train)
            lin = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
            sig = svm.SVC(kernel='sigmoid', gamma=gamma, coef0=2, C=C).fit(X_train, y_train)

            rbf_pred = rbf.predict(X_test)
            poly_pred = poly.predict(X_test)
            lin_pred = lin.predict(X_test)
            sig_pred = sig.predict(X_test)

            rbf_accuracy = accuracy_score(y_test, rbf_pred)
            rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
            poly_accuracy = accuracy_score(y_test, poly_pred)
            poly_f1 = f1_score(y_test, poly_pred, average='weighted')
            lin_accuracy = accuracy_score(y_test, lin_pred)
            lin_f1 = f1_score(y_test, lin_pred, average='weighted')
            sig_accuracy = accuracy_score(y_test, sig_pred)
            sig_f1 = f1_score(y_test, sig_pred, average='weighted')

            rbf_accuracy_values.append(rbf_accuracy)
            rbf_f1_values.append(rbf_f1)
            poly_accuracy_values.append(poly_accuracy)
            poly_f1_values.append(poly_f1)
            lin_accuracy_values.append(lin_accuracy)
            lin_f1_values.append(lin_f1)
            sig_accuracy_values.append(sig_accuracy)
            sig_f1_values.append(sig_f1)

    # Reshape accuracy and F1 score values to match the gamma_values and C_values dimensions
    rbf_accuracy_matrix = np.array(rbf_accuracy_values).reshape(len(gamma_values), len(C_values))
    polynomial_accuracy_matrix = np.array(poly_accuracy_values).reshape(len(gamma_values), len(C_values))
    linear_accuracy_matrix = np.array(lin_accuracy_values).reshape(len(gamma_values), len(C_values))
    sigmoid_accuracy_matrix = np.array(sig_accuracy_values).reshape(len(gamma_values), len(C_values))

    rbf_f1_matrix = np.array(rbf_f1_values).reshape(len(gamma_values), len(C_values))
    polynomial_f1_matrix = np.array(poly_f1_values).reshape(len(gamma_values), len(C_values))
    linear_f1_matrix = np.array(lin_f1_values).reshape(len(gamma_values), len(C_values))
    sigmoid_f1_matrix = np.array(sig_f1_values).reshape(len(gamma_values), len(C_values))

    # Plotting all heatmaps
    fig, axs = plt.subplots(2, 4, figsize=(18, 10))

    # Iterate over kernel types and plot the heatmaps
    for i, (kernel, acc_matrix, f1_matrix) in enumerate(zip(['RBF', 'Polynomial', 'Linear', 'Sigmoid'],
                                                            [rbf_accuracy_matrix, polynomial_accuracy_matrix,
                                                            linear_accuracy_matrix, sigmoid_accuracy_matrix],
                                                            [rbf_f1_matrix, polynomial_f1_matrix, linear_f1_matrix,
                                                            sigmoid_f1_matrix])):
        create_heatmap(acc_matrix, axs[0, i],  'viridis', C_values, gamma_values, f'{kernel} Kernel',)
        create_heatmap(f1_matrix, axs[1, i],  'viridis', C_values, gamma_values)

    # Set labels on the left side
    axs[0, 0].set_ylabel('Accuracy', fontsize=20, fontweight='bold')
    axs[1, 0].set_ylabel('F1', fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.show()


def SVM(X_train, X_test, y_train, y_test, standardized=True):

    # If standardized is true, use the standardized features
    if standardized:
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)
        initial_test(X_train, X_test, y_train, y_test)
        initial_test(X_train_std, X_test_std, y_train, y_test)
        parameters = hyper_parameter_tuning(X_train_std, X_test_std, y_train, y_test)[1]
        write_hyperparameters_to_file(parameters, 'svm_params')

    else:
        parameters = hyper_parameter_tuning(X_train, X_test, y_train, y_test)[1]
        write_hyperparameters_to_file(parameters, 'svm_params')


def read_svm_hyperparameters_from_file(model, X_train, X_test, y_train, y_test):
    hyperparameters = {}
    if model == 'svm':
        if not exists('svm_params'):
            SVM(X_train, X_test, y_train, y_test, True)
        file_path = 'svm_params'


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