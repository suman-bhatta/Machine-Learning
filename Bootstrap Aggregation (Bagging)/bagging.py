"Import the necessary data and evaluate base classifier performance."

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

data = datasets.load_wine(as_frame=True)

x = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

y_pred = dtree.predict(X_test)

print("Train data accuracy: ", accuracy_score(y_train, dtree.predict(X_train)))
print("Test data accuracy: ", accuracy_score(y_test, y_pred))