from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from extracting_features import feature_extraction, data_loading
import seaborn as sns

def create_heatmap(data_matrix, ax, title, cmap, C_values, gamma_values, vmin=0.5, vmax=1):
    sns.heatmap(data_matrix, ax=ax, cmap=cmap, xticklabels=C_values, yticklabels=gamma_values, annot=True, fmt=".2f",
                vmin=vmin, vmax=vmax)
    ax.set_title(title)

# Loading the data
ID, X_all, y = data_loading()
X = X_all[:, [2, 5, 6, 7]]

# Splitting the dataset for hyperparameter finetuning
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.60,
                                                                    test_size=0.40, random_state=101)
# Optional for standardized features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


def hyper_parameter_tuning(X_train, X_test, y_train, y_test):
    best_hyper_parameters = {'kernel': None, 'C': None, 'gamma': None, 'coef0': None, 'degree': None, 'max_iter': None,
                             'decision_function_shape': None, 'accuracy':None}
    best_accuracy = 0.0
    best_f1_score = 0.0

    gamma_values = [80, 90, 100, 110, 120]
    C_values = [0.005, 0.01, 0.5, 1.0, 1.5]
    coeff0_values = [1, 2, 3, 4, 5]
    degree_values = [1, 2, 3, 4, 5]
    dec_function = ['ovr', 'ovo']
    max_iter = [50000, 75000, 100000, -1]

    for C in C_values:
        for iter_val in max_iter:
            for dec_func in dec_function:
                lin = svm.SVC(kernel='linear', max_iter=iter_val, decision_function_shape=dec_func, C=C).fit(X_train,
                                                                                                             y_train)
                lin_pred = lin.predict(X_test)
                lin_accuracy = accuracy_score(y_test, lin_pred)
                lin_f1 = f1_score(y_test, lin_pred, average='weighted')

                if lin_accuracy > best_accuracy:
                    best_accuracy = lin_accuracy
                    best_hyper_parameters = {'kernel': 'linear', 'C': C, 'max_iter': iter_val,
                                             'decision_function_shape': dec_func, 'accuracy': lin_accuracy}
                if lin_f1 > best_f1_score:
                    best_f1_score = lin_f1

                for gamma in gamma_values:
                    rbf = svm.SVC(kernel='rbf', gamma=gamma, C=C, max_iter=iter_val,
                                  decision_function_shape=dec_func).fit(X_train, y_train)
                    rbf_pred = rbf.predict(X_test)
                    rbf_accuracy = accuracy_score(y_test, rbf_pred)
                    rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')

                    if rbf_accuracy > best_accuracy:
                        best_accuracy = rbf_accuracy
                        best_hyper_parameters = {'kernel': 'rbf', 'C': C, 'gamma': gamma, 'max_iter': iter_val,
                                                 'decision_function_shape': dec_func, 'accuracy': rbf_accuracy}
                    if rbf_f1 > best_f1_score:
                        best_f1_score = rbf_f1

                    for coef0 in coeff0_values:
                        for degree in degree_values:
                            poly = svm.SVC(kernel='poly', degree=degree, gamma=gamma, coef0=coef0, C=C,
                                           max_iter=iter_val, decision_function_shape=dec_func).fit(X_train, y_train)
                            poly_pred = poly.predict(X_test)
                            poly_accuracy = accuracy_score(y_test, poly_pred)
                            poly_f1 = f1_score(y_test, poly_pred, average='weighted')

                            if poly_accuracy > best_accuracy:
                                best_accuracy = poly_accuracy
                                best_hyper_parameters = {'kernel': 'poly', 'C': C, 'gamma': gamma, 'coef0': coef0,
                                                         'degree': degree, 'max_iter': iter_val,
                                                         'decision_function_shape': dec_func, 'accuracy': poly_accuracy}
                            if poly_f1 > best_f1_score:
                                best_f1_score = poly_f1

    return best_hyper_parameters


def initial_test():
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
    param_combinations = []

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

    # Iterate over kernel types and metrics to plot all heatmaps
    kernel_types = ['RBF', 'Polynomial', 'Linear', 'Sigmoid']
    metric_names = ['Accuracy', 'F1']

    for i, kernel in enumerate(kernel_types):
        for j, metric in enumerate(metric_names):
            if metric == 'Accuracy':
                data_matrix = globals()[f'{kernel.lower()}_accuracy_matrix']
            else:
                data_matrix = globals()[f'{kernel.lower()}_f1_matrix']

            create_heatmap(data_matrix, axs[j, i], f'{kernel} Kernel - {metric}', 'viridis', C_values, gamma_values)

    plt.tight_layout()
    plt.show()

# print(hyper_parameter_tuning(X_train, X_test, y_train, y_test))
rbf = svm.SVC(kernel='rbf').fit(X_train, y_train)
poly = svm.SVC(kernel='poly').fit(X_train, y_train)
lin = svm.SVC(kernel='linear').fit(X_train, y_train)
sig = svm.SVC(kernel='sigmoid').fit(X_train, y_train)

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
