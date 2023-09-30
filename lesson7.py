
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Prediction 
#Simple Linear Regression data preparation

veriler = pd.read_csv('sales.csv')

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


#attribute scaling, making different values ​​close to each other

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)




