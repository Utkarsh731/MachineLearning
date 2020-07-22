import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

train=pd.read_csv("titanic_train.csv")
train.head()
train.describe()
#print(train.isnull())
#sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#ns.set_style("whitegrid")
#sns.countplot(x="Survived",hue="Sex",data=train)
#sns.boxplot(x="Pclass",y="Age",data=train)
#Data cleaning
def cleanAge(col):
    age=col[0]
    pclass=col[1]
    if pd.isnull(age):
        if pclass==1:
            return 37
        elif pclass==2:
            return 29
        else:
            return 24
    return age

train["Age"]=train[["Age","Pclass"]].apply(cleanAge,axis=1)
train.drop("Cabin",axis=1,inplace=True)
train.dropna(inplace=True)
#Data cleaning done
sex=pd.get_dummies(train["Sex"],drop_first=True)
embark=pd.get_dummies(train["Embarked"],drop_first=True)
train=pd.concat([train,sex,embark],axis=1)
train.drop(["Sex","Embarked","Name","Ticket"],axis=1,inplace=True)
train.drop("PassengerId",axis=1,inplace=True)
x=train.drop("Survived",axis=1)
y=train["Survived"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)
prediction=logmodel.predict(x_test)
print(classification_report(y_test,prediction))
