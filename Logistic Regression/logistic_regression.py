import numpy
from sklearn import linear_model

x = numpy.array([4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
y = numpy.array([21, 19, 18, 17, 15, 16, 14, 13, 12])


logr = linear_model.LogisticRegression()
logr.fit(x, y)

predicated = logr.predict(numpy.array([4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1))
print(predicated)