# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:50:11 2020

@author: benva
"""

#New method for encoding categorial data

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

df = pd.read_csv('http://bit.ly/kaggletrain')
#this below over-writes the existing df to this
df = df.loc[df.Embarked.notna(), ['Survived', 'Pclass', 'Sex', 'Embarked']]
X = df.drop('Survived', axis = 'columns')
y = df.Survived

#make column transformer made of ohe and passing through all the columns
column_trans = make_column_transformer((OneHotEncoder(), ['Sex', 'Embarked']), remainder = 'passthrough')
column_trans.fit_transform(X)
logreg = LogisticRegression(solver = 'lbfgs')

#pipeline that a column transforms and a model
pipe = make_pipeline(column_trans, logreg)

#cross validation of the pipeling
cross_val_score(pipe, X, y, cv = 5, scoring = 'accuracy').mean()


#building Xs new dataframe
X_new = X.sample(5, random_state = 99)

#making predictions on new data
pipe.fit(X, y)
pipe.predict(X_new)


#this way is better than using dummy encoding in pandas because
# 1. dont need to create a huge dataframe, dataframe stays the same throughout
# 2. when new data comes in you dont have to use dummys on it
# 3. you can do a grid search with both model and preprocessing parameters
# 4. in some cases, preprocessing outside of scikit learn can make cross validation scores less reliable