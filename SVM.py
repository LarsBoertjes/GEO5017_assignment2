from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from extracting_features import feature_extraction, data_loading
import seaborn as sns

ID, X_all, y = data_loading()
X = X_all[:, [2, 5, 6, 7]]
print(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.60,
                                                                    test_size=0.40, random_state=30)
scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

rbf = svm.SVC(kernel='rbf', gamma=0.1, C=1.0).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=2, gamma=0.1, coef0=1, C=1).fit(X_train, y_train)
lin = svm.SVC(kernel='linear', C=1.0).fit(X_train, y_train)
sig = svm.SVC(kernel='sigmoid', gamma=0.1, coef0=2, C=1.0).fit(X_train, y_train)

poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
lin_pred = lin.predict(X_test)
sig_pred = sig.predict(X_test)

# rbf_accuracy = accuracy_score(y_test, rbf_pred)
# rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
# print('Accuracy (RBF Kernel):', '%.2f' % (rbf_accuracy * 100))
# print('F1 (RBF Kernel):', '%.2f' % (rbf_f1 * 100))
#
# poly_accuracy = accuracy_score(y_test, poly_pred)
# poly_f1 = f1_score(y_test, poly_pred, average='weighted')
# print('Accuracy (Polynomial Kernel):', '%.2f' % (poly_accuracy * 100))
# print('F1 (Polynomial Kernel):', '%.2f' % (poly_f1 * 100))
#
# lin_accuracy = accuracy_score(y_test, lin_pred)
# lin_f1 = f1_score(y_test, lin_pred, average='weighted')
# print('Accuracy (Linear Kernel):', '%.2f' % (lin_accuracy * 100))
# print('F1 (Linear Kernel):', '%.2f' % (lin_f1 * 100))
#
# sig_accuracy = accuracy_score(y_test, sig_pred)
# sig_f1 = f1_score(y_test, sig_pred, average='weighted')
# print('Accuracy (SIG):', '%.2f' % (sig_accuracy * 100))
# print('F1 (SIG):', '%.2f' % (sig_f1 * 100))

gamma_values = [0.001, 0.005, 0.01, 0.5, 0.1, 0.5, 1.0, 5]
C_values = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 150]
deg_values = [1, 2, 3, 4]
coef0_values = [1, 2, 3, 4]

rbf_accuracy_values = []
rbf_f1_values = []
poly_accuracy_values = []
poly_f1_values = []
lin_accuracy_values = []
lin_f1_values = []
sig_accuracy_values = []
sig_f1_values = []
param_combinations = []

for gamma in gamma_values:
    for C in C_values:
        param_combinations.append((gamma, C))
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


# Plotting

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# RBF Kernel
axs[0, 0].plot(np.arange(len(rbf_accuracy_values)), rbf_accuracy_values, label='Accuracy')
axs[0, 0].plot(np.arange(len(rbf_f1_values)), rbf_f1_values, label='F1 Score')
axs[0, 0].set_title('RBF Kernel')
axs[0, 0].legend()

# Polynomial Kernel
axs[0, 1].plot(np.arange(len(poly_accuracy_values)), poly_accuracy_values, label='Accuracy')
axs[0, 1].plot(np.arange(len(poly_f1_values)), poly_f1_values, label='F1 Score')
axs[0, 1].set_title('Polynomial Kernel')
axs[0, 1].legend()

# Linear Kernel
axs[1, 0].plot(np.arange(len(lin_accuracy_values)), lin_accuracy_values, label='Accuracy')
axs[1, 0].plot(np.arange(len(lin_f1_values)), lin_f1_values, label='F1 Score')
axs[1, 0].set_title('Linear Kernel')
axs[1, 0].legend()

# Sigmoid Kernel
axs[1, 1].plot(np.arange(len(sig_accuracy_values)), sig_accuracy_values, label='Accuracy')
axs[1, 1].plot(np.arange(len(sig_f1_values)), sig_f1_values, label='F1 Score')
axs[1, 1].set_title('Sigmoid Kernel')
axs[1, 1].legend()

for ax in axs.flat:
    ax.set_xlabel('Parameter Combination')
    ax.set_ylabel('Performance')
    ax.set_xticks(np.arange(len(param_combinations)))
    ax.set_xticklabels([f'({gamma}, {C})' for gamma, C in param_combinations], rotation=45, ha='right')

for ax in axs.flat:
    ax.set_xlabel('Parameter Combination')
    ax.set_ylabel('Performance')
    ax.set_xticks(np.arange(len(param_combinations)))
    ax.set_xticklabels([f'({gamma}, {C})' for gamma, C in param_combinations], rotation=45, ha='right')


plt.tight_layout()
plt.show()

# Reshape the accuracy values to match the gamma_values and C_values dimensions
rbf_accuracy_matrix = np.array(rbf_accuracy_values).reshape(len(gamma_values), len(C_values))
poly_accuracy_matrix = np.array(poly_accuracy_values).reshape(len(gamma_values), len(C_values))
lin_accuracy_matrix = np.array(lin_accuracy_values).reshape(len(gamma_values), len(C_values))
sig_accuracy_matrix = np.array(sig_accuracy_values).reshape(len(gamma_values), len(C_values))

# Plotting using Seaborn
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# RBF Kernel
sns.heatmap(rbf_accuracy_matrix, ax=axs[0, 0], cmap='viridis', xticklabels=C_values, yticklabels=gamma_values, annot=True, fmt=".2f")
axs[0, 0].set_title('RBF Kernel - Accuracy')

# Polynomial Kernel
sns.heatmap(poly_accuracy_matrix, ax=axs[0, 1], cmap='viridis', xticklabels=C_values, yticklabels=gamma_values, annot=True, fmt=".2f")
axs[0, 1].set_title('Polynomial Kernel - Accuracy')

# Linear Kernel
sns.heatmap(lin_accuracy_matrix, ax=axs[1, 0], cmap='viridis', xticklabels=C_values, yticklabels=gamma_values, annot=True, fmt=".2f")
axs[1, 0].set_title('Linear Kernel - Accuracy')

# Sigmoid Kernel
sns.heatmap(sig_accuracy_matrix, ax=axs[1, 1], cmap='viridis', xticklabels=C_values, yticklabels=gamma_values, annot=True, fmt=".2f")
axs[1, 1].set_title('Sigmoid Kernel - Accuracy')

plt.tight_layout()