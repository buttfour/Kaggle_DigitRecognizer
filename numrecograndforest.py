# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 13:39:23 2016

@author: buttfive
"""

#Import the necessary packages.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#Import the data
df_train = pd.read_csv("C:/train.csv")
df_test = pd.read_csv("C:/test.csv")

#Preprocess the data.
np_train = np.array(df_train)
np_test = np.array(df_test)
y=np_train[:,0]
X=np_train[:,1:]

#Use a random forest classifier to fit the training data and classify the test data.
model= RandomForestClassifier(n_estimators=1000,n_jobs=-1)
model.fit(X, y)
value = model.predict(np_test)

#Postprocess and export the data in the desired format.
value = pd.DataFrame(value, index = np.arange(value.shape[0]+1)[1:])
value.to_csv('value.csv',sep=',',header=['Label'], index_label='ImageId')