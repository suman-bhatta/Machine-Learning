"""
Generating Decision Trees from Bagging Classifier
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# Load the Iris dataset
data = datasets.load_iris()

# Convert X to a pandas DataFrame to add column names
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

# Train the Bagging Classifier
clf = BaggingClassifier(n_estimators=12, oob_score=True, random_state=22)
clf.fit(X_train, y_train)

# Plot a tree from one of the base estimators
plt.figure(figsize=(12, 8))
plot_tree(clf.estimators_[0], feature_names=X.columns, class_names=data.target_names, filled=True)
plt.show()
