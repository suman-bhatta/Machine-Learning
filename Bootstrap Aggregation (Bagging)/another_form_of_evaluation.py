"Create a model with out-of-bag metric."

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier

data = datasets.load_wine(as_frame = True)

x = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

oob_model = BaggingClassifier(oob_score = True, random_state = 42)
oob_score = oob_model.fit(X_train, y_train).oob_score_ 

oob_model.fit(X_train, y_train)

print("Out-of-bag score: ", oob_score)