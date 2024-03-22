from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sklearn.model_selection as model_selection
import matplotlib.pyplot as plt
import numpy as np
from extracting_features import data_loading

ID, X_all, y = data_loading()
X = X_all[:, [2, 5, 6, 7]]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.60,
                                                                    test_size=0.40, random_state=101)

# Hyperparameters ranges
n_estimators = [1, 2, 5, 10, 25, 50, 75, 100, 125, 150]
criterion = ["gini", "entropy", "log_loss"]
max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bootstrap = [True, False]
max_samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
min_samples_split = [2, 5, 10, 20, 50, 100]

best_score = 0
best_params = 0

for n in n_estimators:
    for crit in criterion:
        for max_feat in max_features:
            for boot in bootstrap:
                    for depth in max_depth:
                        RF = RandomForestClassifier(n_estimators=n, criterion=crit, max_features=max_feat,
                                                        bootstrap=boot, max_depth=depth, random_state=0)
                        RF.fit(X_train, y_train)
                        predictions = RF.predict(X_test)
                        score = accuracy_score(y_test, predictions)

                        if score > best_score:
                            best_score = score
                            best_params = {'n_estimators': n, 'criterion': crit, 'max_features': max_feat,
                                               'bootstrap': boot, 'max_depth': depth,}

print("Best score:", best_score)
print("Best parameters:", best_params)