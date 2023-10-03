
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Prediction 
#Simple Linear Regression data preparation

veriler = pd.read_csv('datasources/sales.csv')

print(veriler)

months=veriler[['Aylar']]
print(months)

sales=veriler[['Satislar']]
print(sales)

sales2=veriler.iloc[:,:1].values
print(sales2)


#splitting data for training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(months,sales,test_size=0.33, random_state=0)

'''
#attribute scaling, making different values ​​close to each other

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)
'''
#model build(linear regression)
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)

pr=lr.predict(x_test)

x_train=x_train.sort_index()
y_train=y_train.sort_index()
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("sales by months")
plt.xlabel("Months")
plt.ylabel("Sales")