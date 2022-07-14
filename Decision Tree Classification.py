import numpy as np
import pandas as pd
from sklearn import metrics, tree

#importing data
dt=pd.read_csv('weather.csv')
print(dt.head())


#factorised
# converting the test data to Num data
dt['play'],_=pd.factorize(dt['play'])
dt['outlook'],_=pd.factorize(dt['outlook'])
dt['temperature'],_=pd.factorize(dt['temperature'])
dt['humidity'],_=pd.factorize(dt['humidity'])
dt['windy'],_=pd.factorize(dt['windy'])
print(dt.head())


#defining the dataset

X=dt.iloc[:,0:-1]
Y=dt.iloc[:,-1]

#print(X.head())
#print(Y.head())

#split the data into 80% training and 20% testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#train te decision tree
#dtree=dtree.DecisonTreeClassificatier(criterion = 'entropy',max_depth=3)
dtree=tree.DecisionTreeClassifier(criterion='gini',max_depth=3)
dtree.fit(X_train,Y_train)

#use the model to make the prediction with the test data
Y_pred=dtree.predict(X_test)
#print("Prediction values:",Y_pred)
'''
from sklearn.metrics import accuracy_score,confusion_matrix
conf_mat= confusion_matrix(Y_test,Y_pred)
print("The confusion matrix is --\n",conf_mat)
'''
#now caluclating that how much accurate
print("Accuracy==",metrics.accuracy_score(Y_test,Y_pred))

##create some feature values for this single row
'''
new=pd.DataFrame()

outlook=input("enter outlook(sunny/overcast/rainy):")
if(outlook=="sunny"):
    outlook=0
elif(outlook=="overcast"):
    outlook=1
elif(outlook=="rainy"):
    outlook=2

temp=input("enter temp(hot/mild/cool):")
if(temp=="hot"):
    temp=0
elif(temp=="mild"):
    temp=1
elif(temp=="rainy"):
    temp=2    



hum=input("enter Humidity(high/normal):")
if(hum=="high"):
    hum=0
elif(hum=="normal"):
    hum=1
 
     
wind=input("enter windy(False/True):")

if(wind=="false"):
    wind=0
elif(wind=="true"):
    wind=1
 
new['outlook']=[outlook]
new['temperature']=[temp]
new['humidity']=[hum]
new['windy']=[wind]

print(new)

Y_pred=model.predict(new)
if(Y_pred==1):
    print("play possible")
else:
    print("play not possible")
    
'''












      














