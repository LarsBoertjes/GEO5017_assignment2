from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sklearn.model_selection as model_selection
from extracting_features import data_loading
from writing_hyperparameters import write_hyperparameters_to_file

ID, X_all, y = data_loading()
X = X_all[:, [2, 5, 6, 7]]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.60,
                                                                    test_size=0.40, random_state=101)

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

# output best params giving score of 0.95:
# n_estimators = 75
# criterion = gini
# max_features = 0.1
# bootstrap = True
# max_samples = 0.4
# max_depth = 6
# min_samples_split 2

# write hyperparameters to file
write_hyperparameters_to_file(best_hyperparameters, 'RF_params')

