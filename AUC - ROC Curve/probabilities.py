import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from matplotlib import pyplot as plt

n = 10000
y = np.array([0] * n + [1] * n)
# 
y_prob_1 = np.array(
    np.random.uniform(.25, .5, n//2).tolist() + 
    np.random.uniform(.3, .7, n).tolist() + 
    np.random.uniform(.5, .75, n//2).tolist()
)
y_prob_2 = np.array(
    np.random.uniform(0, .4, n//2).tolist() + 
    np.random.uniform(.3, .7, n).tolist() + 
    np.random.uniform(.6, 1, n//2).tolist()
)

print(f'model 1 accuracy score: {accuracy_score(y, y_prob_1>.5)}')
def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_true, y_prob):.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()

plot_roc_curve(y, y_prob_1, 'Model 1')
print(f'model 2 accuracy score: {accuracy_score(y, y_prob_2>.5)}')
plot_roc_curve(y, y_prob_2, 'Model 2')
plt.show()

print(f'model 1 AUC score: {roc_auc_score(y, y_prob_1)}')
print(f'model 2 AUC score: {roc_auc_score(y, y_prob_2)}')
