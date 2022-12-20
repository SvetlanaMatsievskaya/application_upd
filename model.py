import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, LabelEncoder, MinMaxScaler, StandardScaler
df = pd.read_excel('df_clean.xlsx')

df.drop(df.columns[df.columns.str.contains('unnamed', case=False)],
           axis=1,
           inplace=True)
X = df.drop(columns=['Прочность при растяжении, МПа'],
            axis=1)
y = df['Прочность при растяжении, МПа']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
lr = make_pipeline(StandardScaler(),LinearRegression())
lr.fit(X_train, y_train)
pickle.dump(lr, open('model.pkl','wb'))