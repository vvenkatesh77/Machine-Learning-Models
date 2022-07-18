import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
dataset=pd.read_csv('bill_authentication.csv')
print(dataset.head())

x=dataset.iloc[:,0:4]
y=dataset.iloc[:,4]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


#Feature Scaling
#print("Before applying Standard Scaler:")
print(x_train)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train =sc.fit_transform(x_train)
x_test=sc.transform(x_test)
#print("After applying Standard Scaler:")
print(x_train)

from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

rc=BaggingClassifier(DecisionTreeClassifier(),n_estimators=5)
#rc=AdaBoostClassifier(n_estimators=22,learning_rate=1)
#rc=GradientBoostingClassifier(max_depth=2,n_estimators=24,learning_rate=1.0)
#rc=RandomForestClassifier(n_estimators=5)

rc.fit(x_train,y_train)
y_pred=rc.predict(x_test)
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred)*100)

#Finding the number of n_estimators
'''
for n in range(1,50):
    rc=AdaBoostClassifier(n_estimators=n,learning_rate=1)
    
    #rc=GradientBoostingClassifier(max_depth=2,n_estimators=n,learning_rate=1.0)
    rc.fit(x_train,y_train)
    y_pred=rc.predict(x_test)
    print("Accuraacy for %s estimators:%s" %(n,accoracy_score(y_test,y_pred)))

'''

#Applying Multiple Algorithms at same time
bc=BaggingClassifier(DecisionTreeClassifier(),n_estimators=5,random_state=0)
abc=AdaBoostClassifier(n_estimators=22,learning_rate=1)
gb=GradientBoostingClassifier(max_depth=2,n_estimators=24,learning_rate=1.0)
rfc=RandomForestClassifier(n_estimators=5)
d1={}
for rc in bc,abc,gb,rfc:
      rc.fit(x_train, y_train)
      y_pred = rc.predict(x_test)
      #print(confusion_matrix(y_test,y_pred))
      #print(classification_report(y_test,y_pred))
      #print("Accuracy for",rc,accuracy_score(y_test, y_pred))
      d1[rc]=accuracy_score(y_test, y_pred)*100
      
for k,v in d1.items():
      print(k,":",v)





















