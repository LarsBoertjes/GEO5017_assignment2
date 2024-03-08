from sklearn import datasets, svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -- Iris dataset
# 3 different types of irises
# rows being the samples
# columns being: Sepal Length, Sepal Width, Petal Length and Petal Width
iris = datasets.load_iris()

# Scatter plot of the Iris dataset
_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c = iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc='lower right', title="Classes"
)

# plot shows clear Pattern for Setosa type
# Versicolor & Virginica types in these dimensions still overlapping
plt.show()

X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    train_size=0.6, test_size=0.4, random_state=101)

# construct the SVC classifiers on the trainingset
# rbf (gaussian)
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
# poly (polynomial)
poly = svm.SVC(kernel='poly', degree=3, C=1).fit(X_train, y_train)

# Perform predictions on the test set
poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)

