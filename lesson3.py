
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('eksikveriler.csv')

print(veriler)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

ulke=veriler.iloc[:,0:1].values
print(ulke)

# the process of numbering categorical data

from sklearn import preprocessing

le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)
