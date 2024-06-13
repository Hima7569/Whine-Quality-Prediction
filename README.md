# Whine-Quality-Prediction
# Based upon the features the model predicts the quality of the winr 
# here the machine learning model was SVM
import pandas as pd
import numpy as np
Import Data
df=pd.read_csv(r'/content/WhiteWineQuality (2).csv',sep=';')
Describe Data
df.describe()
df.info()
df.head()
# Data Visualization
from matplotlib import pyplot as plt
df.plot(kind='scatter', x='fixed acidity', y='volatile acidity', s=32, alpha=.8)
df['volatile acidity'].plot(kind='line', figsize=(8, 4), title='volatile acidity')
# Data Preprocessing
df.shape
df.columns
df.quality.value_counts()
df.isnull().sum()
df.groupby('quality').mean()
# Define Target Variable (y) and Feature Variables (X)
y=df['quality']
print(y.shape)
x = df.drop(['quality'], axis=1)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x = ss.fit_transform(x)
x
Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
x_train.shape,x_test.shape,y_train.shape,y_test.shape
# Modeling
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
pred=svc.predict(x_test)
# Model Evaluation
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
# Predictions
pred
