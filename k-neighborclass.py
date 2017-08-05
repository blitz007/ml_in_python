from mglearn import datasets
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
X,y = datasets.make_forge()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',metric_params=None,n_jobs=1,n_neighbors=3,p=2,weights='uniform')

print (clf.predict(X_train))
print (clf.score(X_test,y_test))




