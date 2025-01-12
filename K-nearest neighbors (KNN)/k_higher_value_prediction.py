# Assuming data, classes, new_point, x, y, new_x, and new_y are defined elsewhere in your code
# Example:

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


data = [[1, 2], [2, 3], [3, 4], [4, 5]]
classes = [0, 1, 0, 1]
new_point = [[2.5, 3.5]]
x = [1, 2, 3, 4]
y = [2, 3, 4, 5]
new_x = 2.5
new_y = 3.5

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(data, classes)

prediction = knn.predict(new_point)

plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
plt.show()