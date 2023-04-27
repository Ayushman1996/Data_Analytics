# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:42:15 2023

@author: user
"""

import pandas as pd
import seaborn as sns
df=pd.read_csv("C:/Users/user/Desktop/ML_Model/car_evaluation.csv")

df.head()
df.tail()
df.shape

#Renaming of columns
col_name=["buying","Maint","doors","persons","lug_boot","safety","class"]
df.columns=col_name

df.info()

for x in col_name:
    print(df[x].value_counts())
    
x=df.drop(["class"],axis=1)
y=df["class"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=(4))


## One_hot Encoding...
x_train=pd.get_dummies(x_train,columns=["buying","Maint","doors","persons","lug_boot","safety"])
x_test=pd.get_dummies(x_test,columns=["buying","Maint","doors","persons","lug_boot","safety"])



from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100,random_state=(4))

rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

feature_score=pd.Series(rfc.feature_importances_,index=x_train.columns)
sns.barplot(x=feature_score, y=feature_score.index)

from sklearn.metrics import classification_report
class_1=classification_report(y_test, y_pred)