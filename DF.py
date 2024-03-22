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

RF = RandomForestClassifier(max_depth=5, random_state=0)
RF.fit(X_train, y_train)

predictions = RF.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')
cm = confusion_matrix(y_test, predictions)

print(accuracy)
print(f1)
print(cm)

# Tweaking the hyperparameters
def tweak_train_test_ratio(X, y, train_size):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=train_size, random_state=101)
    RF = RandomForestClassifier(max_depth=5, random_state=0).fit(X_train, y_train)
    predictions = RF.predict(X_test)
    return f1_score(y_test, predictions, average='weighted')

def tweak_max_depth(X, y, max_depth):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, random_state=101)
    RF = RandomForestClassifier(max_depth=max_depth, random_state=0).fit(X_train, y_train)
    predictions = RF.predict(X_test)
    return f1_score(y_test, predictions, average='weighted')

def tweak_n_estimators(X, y, n_estimators):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.7, random_state=101)
    RF = RandomForestClassifier(max_depth=5, n_estimators=n_estimators, random_state=0).fit(X_train, y_train)
    predictions = RF.predict(X_test)
    return f1_score(y_test, predictions, average='weighted')

ratios = [(i/20) for i in range(1, 20)]
f1_scores_train = []

for ratio in ratios:
    f1 = tweak_train_test_ratio(X, y, ratio)
    f1_scores_train.append(f1)

depths = range(1, 15)
f1_scores_depth = []

for depth in depths:
    f1 = tweak_max_depth(X, y, depth)
    f1_scores_depth.append(f1)

# Generate a range of n_estimators values
estimators = range(10, 201, 10)
f1_scores_estimators = []

for estimator in estimators:
    f1 = tweak_n_estimators(X, y, estimator)
    f1_scores_estimators.append(f1)

plt.figure(figsize=(18, 6))

# Plot for Train Ratio
plt.subplot(1, 3, 1)
plt.plot(ratios, f1_scores_train, marker='o', linestyle='-', color='b')
plt.title('F1 Score vs. Train Ratio')
plt.xlabel('Train Ratio')
plt.ylabel('F1 Score')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(True)

# Plot for Max Depth
plt.subplot(1, 3, 2)
plt.plot(depths, f1_scores_depth, marker='o', linestyle='-', color='g')
plt.title('F1 Score vs. Max Depth')
plt.xlabel('Max Depth')
plt.xticks(depths)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(True)

# New plot for n_estimators
plt.subplot(1, 3, 3)
plt.plot(estimators, f1_scores_estimators, marker='o', linestyle='-', color='r')
plt.title('F1 Score vs. n_estimators')
plt.xlabel('n_estimators')
plt.xticks(estimators, rotation='vertical')  # Adjust rotation for better label visibility
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(True)

plt.tight_layout()
plt.show()

print("We're done")