import pandas as pd
import numpy as np
#from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


df=pd.read_csv('Admission_Prediction.csv')
print(df.head(10))


#import matplotlib.pyplot as plt
'''
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0[0],df[1],color="green",marker='+')
plt.scatter(df0[0],df[1],color="blue",marker='*')
plt.show()
'''

print("NaN Values List before Cleaning",df.isna().sum())

#As we can see ,there are some column with missing values. we need to impute those missing values
df['GRE Score'].fillna(df['GRE Score'].mean(),inplace=True)
df['TOEFL Score'].fillna(df['TOEFL Score'].mean(),inplace=True)
df['University Rating'].fillna(df['University Rating'].mode()[0],inplace=True)

#seeeing that after imputation no colulmn has missing values
print("NaN Values list after cleaning",df.isna().sum())

#print(df['University Rating'].head(10))

X=df.drop(['Chance of Admit','Serial No.'],axis=1)
Y=df['Chance of Admit']


#Train using support Vector Machine
#splitting the dataset into training set and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#importing the model
from sklearn.svm import SVR
svr=SVR(C=10)
svr.fit(X_train,Y_train)

Y_pred=svr.predict(X_test)

#Accuracy test
from sklearn.metrics import r2_score
from sklearn import metrics
rmse=np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
print("RMSE Score:",rmse)
score=metrics.r2_score(Y_test,Y_pred)
print("R2 Score:",score)

