import numpy
import matplotlib.pyplot as plt

x = numpy.random.uniform(0.0, 5.0, 100000)
y = numpy.random.uniform(0.0, 5.0, 100000)

plt.scatter(x, y)
plt.show()