# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:50:11 2020

@author: benva
"""

#New method for encoding categorial data

import pandas as pd

df = pd.read_csv('http://bit.ly/kaggletrain')
#this below over-writes the existing df to this
df = df.loc[df.Embarked.notna(), ['Survived', 'Pclass', 'Sex', 'Embarked']]

X = df.loc[:, ['Pclass']]
y = df.Survived

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = 'lbfgs')
logreg.fit(X, y)



from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit_transform(df[['Embarked']])

X = df.drop('Survived', axis = 'columns')

from sklearn.compose import make_column_transformer
column_trans = make_column_transformer((OneHotEncoder(), ['Sex', 'Embarked']), remainder = 'passthrough')
column_trans.fit_transform(X)

#for chaining things together
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(column_trans, logreg)


X_new = X.sample(5, random_state = 99)
pipe.fit(X, y)
pipe.predict(X_new)