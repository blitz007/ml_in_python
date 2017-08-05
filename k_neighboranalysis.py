from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import neighbors  
cancer= load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer['data'],cancer['target'],stratify=cancer['target'],random_state=66)

training_accuracy=[]
test_accuracy=[]
#trying neighbors from n=1 to 10
neighbors_settings=range(1,11)

for n_neighbors in neighbors_settings:
	clf=neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
	clf.fit(X_train,y_train)
	training_accuracy.append(clf.score(X_train,y_train))
	test_accuracy.append(clf.score(X_test,y_test))

plt.plot(neighbors_settings,training_accuracy,label='Training Accuracy')
plt.plot(neighbors_settings,test_accuracy,label='Test Accuracy')
plt.legend()
plt.show()



