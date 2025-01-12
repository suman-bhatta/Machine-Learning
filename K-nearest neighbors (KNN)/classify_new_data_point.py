import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Sample data
x = [1, 2, 3, 4, 5, 6, 7]
y = [10, 20, 30, 40, 50, 60, 70]
classes = [0, 1, 0, 1, 0, 1, 0]

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(list(zip(x, y)), classes)

new_x = 8
new_y = 22
new_point = [[new_x, new_y]]

prediction = knn.predict(new_point)

plt.scatter(x, y, c=classes)

plt.scatter(new_x, new_y, c='red', marker='x', s=100)
plt.text(new_x, new_y, f"Prediction: {prediction[0]}", fontsize=12)
plt.show()