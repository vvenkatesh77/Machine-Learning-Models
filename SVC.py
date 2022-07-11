import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()


df=pd.DataFrame(iris.data) # 4 cols
df['target']= iris.target#adding a col

print(df.head())

df0=df[:50]
df1=df[50:100]
df2=df[100:]

import matplotlib.pyplot as plt

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0[0],df[1],color="green",marker='+')
plt.scatter(df0[0],df[1],color="blue",marker='*')
plt.show()

#Train using support Vector Machine

X=df.iloc[:,:4]
Y=df.iloc[:,4]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.svm import SVC
model=SVC()
model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)

#making the confusion matrix
cm=metrics.confusion_matrix(Y_test,Y_pred)
print("Confusion matrix:\n",cm)

#finding the accuracy of the predicted value(Y_pred) by comparing wit the actual resulta
print('accuracy',metrics.accuracy_score(Y_test,Y_pred))
#print("prediction:",model.predict([[4.8,3.0,1.5,0.3]]))

#tune parameters
#**1 Regularization (c)**

model_C= SVC(C=1)
model_C.fit(X_train,Y_train)
print("Model using c=1 as reg",model_C.score(X_test,Y_test))

model_C=SVC(C=10)
model_C.fit(X_train,Y_train)
print("Model using c=10 as reg",model_C.score(X_test,Y_test))



#**2.Gamma**
model_g=SVC(gamma=10)
model_g.fit(X_train,Y_train)
print("Model score gamma",model_C.score(X_test,Y_test))


#  **Kernal**
model_linear_kernal=SVC(kernal='linear')
model_linear_kernal.fit(X_train,Y_train)
print("Model score Kernal",model_linear_kernal.score(X_test,Y_test))











