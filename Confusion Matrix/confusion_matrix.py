import matplotlib.pyplot as plt
import numpy
from sklearn import metrics


actual = numpy.random.binomial(1,0.9,size=100)
predicted = numpy.random.binomial(1,0.5,size=100)


confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=[0,1]).plot()
# confusion_matrix, diaplay_labels=[0,1]

cm_display.plot()
plt.show()


"Accuracy"

import numpy
from sklearn import metrics

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

Accuracy = metrics.accuracy_score(actual, predicted)

print(Accuracy)




"Precision"

import numpy
from sklearn import metrics

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

Precision = metrics.precision_score(actual, predicted)

print(Precision)



"Sensitivity (Recall)"

import numpy 
from sklearn import metrics


actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

sensitivity = metrics.recall_score(actual, predicted)

print(sensitivity)


"Specificity"


import numpy
from sklearn import metrics

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

specificity = metrics.recall_score(actual, predicted, pos_label=0)

print(specificity)