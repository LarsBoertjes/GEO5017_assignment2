from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import sklearn.model_selection as model_selection
from geometric_features import extract_geometric_features
from read_data import point_clouds_for_classification

def dataset():
    pc = point_clouds_for_classification('data')

    dataset = np.zeros((len(pc[1]), 4))

    all_features = extract_geometric_features()[0]

    relative_heights = []
    for height in pc[2]:
        relative_heights.append(max(height) - min(height))

    dataset[:, 0] = all_features[2]
    dataset[:, 1] = all_features[5]
    dataset[:, 2] = all_features[6]
    dataset[:, 3] = relative_heights

    labels = pc[1]

    return dataset, labels

# read dataset
X, y = dataset()


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.60,
                                                                    test_size=0.40, random_state=101)

rbf = svm.SVC(kernel='rbf', gamma=0.1, C=1.0).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=2, coef0=1, C=1).fit(X_train, y_train)
lin = svm.SVC(kernel='linear', C=1.0).fit(X_train, y_train)
sig = svm.SVC(kernel='sigmoid', coef0=2, C=1.0, gamma=0.1).fit(X_train, y_train)

poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
lin_pred = lin.predict(X_test)
sig_pred = sig.predict(X_test)

rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel):', '%.2f' % (rbf_accuracy * 100))
print('F1 (RBF Kernel):', '%.2f' % (rbf_f1 * 100))

poly_accuracy = accuracy_score(y_test, poly_pred)
poly_f1 = f1_score(y_test, poly_pred, average='weighted')
print('Accuracy (Polynomial Kernel):', '%.2f' % (poly_accuracy * 100))
print('F1 (Polynomial Kernel):', '%.2f' % (poly_f1 * 100))

lin_accuracy = accuracy_score(y_test, lin_pred)
lin_f1 = f1_score(y_test, lin_pred, average='weighted')
print('Accuracy (Linear Kernel):', '%.2f' % (lin_accuracy * 100))
print('F1 (Linear Kernel):', '%.2f' % (lin_f1 * 100))

sig_accuracy = accuracy_score(y_test, sig_pred)
sig_f1 = f1_score(y_test, sig_pred, average='weighted')
print('Accuracy (SIG):', '%.2f' % (sig_accuracy * 100))
print('F1 (SIG):', '%.2f' % (sig_f1 * 100))