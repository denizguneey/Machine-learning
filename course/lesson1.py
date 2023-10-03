import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yukleme
#/Users/denizguney/Desktop/MachineeLearning/ aynı dosya içine kaydetmezsen

veriler= pd.read_csv('datasources/veriler.csv') 

#veri onisleme
boy=veriler[['boy']]
print(boy)

boykilo=veriler[['boy','kilo']]
print(boykilo)