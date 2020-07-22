import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv('USA_Housing.csv')
#df.describe()
#df.info()
#print(df.columns)
#sns.pairplot(df)
#sns.distplot(df["Price"])
#plt.show()
x=df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y=df['Price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=1001)
lm=LinearRegression()
lm.fit(x_train,y_train)

#print(lm.intercept_)
#print(lm.coef_)
cdf=pd.DataFrame(lm.coef_,x.columns,columns=["Coeff"])
predict=lm.predict(x_test)
#print(list(x_test))
plt.scatter(y_test,predict)
sns.distplot(y_test-predict)
print(metrics.mean_absolute_error(y_train,predict))
