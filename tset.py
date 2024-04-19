from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.mixture import GaussianMixture

iris = load_iris()

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.

X_train = iris.data[train_index]
y_train = iris.target[train_index]
X_test = iris.data[test_index]
y_test = iris.target[test_index]
