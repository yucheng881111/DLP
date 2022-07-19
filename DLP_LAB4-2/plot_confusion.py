import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score


with open('pred.txt', 'r') as f:
	lines_p = f.readlines()

p = lines_p[0].split()
pre = [int(i) for i in p]


with open('lab.txt', 'r') as f:
	lines_l = f.readlines()

l = lines_l[0].split()
lab = [int(i) for i in l]

print(len(pre))
print(len(lab))
ConfusionMatrixDisplay.from_predictions(lab, pre, normalize = 'true')
#ConfusionMatrixDisplay.from_predictions(lab, pre)
print(accuracy_score(lab, pre))
plt.show()
