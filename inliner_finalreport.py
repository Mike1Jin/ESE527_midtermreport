#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import KernelPCA, PCA

from factor_analyzer import FactorAnalyzer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


from sklearn import preprocessing  
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV


# In[72]:


df = pd.read_excel('inliner.xlsx')
X_input = df.dropna()
scaler = MinMaxScaler()
X_input = scaler.fit_transform(X_input)
X_input = pd.DataFrame(X_input)
X_input.columns = ["Frequency","Angle of attack","Chord length","Free-stream velocity","Suction side displacement thickness","label","Scaled sound pressure level"] #添加自定义的columns的名字


# In[73]:


X_input


# In[74]:


X = X_input.iloc[:,0:6]

y = X_input.iloc[:,6]


# In[75]:


X


# In[76]:


y


# In[77]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

x = sm.add_constant(X) 
model = sm.OLS(y, x).fit() 
print(model.summary())


# In[78]:


X_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=10) #


# In[80]:


mean_squared_error(y_test,model.predict(x_test))


# In[81]:


from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[82]:



X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10) #
ridge=Ridge(alpha=0.4)
ridge.fit(X_train,y_train)
print("Training dataset score："+str(r2_score(y_train,ridge.predict(X_train))))
print("Testing dataset score："+str(r2_score(y_test,ridge.predict(x_test))))
print("Training dataset MSE："+str(mean_squared_error(y_train,ridge.predict(X_train))))
print("Testing dataset MSE："+str(mean_squared_error(y_test,ridge.predict(x_test))))


# In[83]:


from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split


# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# perform Lasso regression with cross-validation
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train, y_train)

# evaluate the model on the test set
train_score=lasso_cv.score(X_train, y_train)
test_score = lasso_cv.score(X_test, y_test)
print("Training dataset score："+str(train_score))
print("Testing dataset score："+str(test_score))
print("Training dataset MSE："+str(mean_squared_error(y_train,lasso_cv.predict(X_train))))
print("Testing dataset MSE："+str(mean_squared_error(y_test,lasso_cv.predict(X_test))))


# In[ ]:





# In[84]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[85]:


X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10) 
from sklearn.preprocessing import MinMaxScaler
# Training dataset MinMaxSscaler
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(X_train)
X_train = min_max_scaler.transform(X_train)
y_train=y_train.values.reshape(-1,1)
min_max_scaler.fit(y_train)
y_train = min_max_scaler.transform(y_train)
# Testing dataset MinMaxSscaler
min_max_scaler.fit(x_test)
x_test = min_max_scaler.transform(x_test)
y_test=y_test.values.reshape(-1,1)
min_max_scaler.fit(y_test)
y_test = min_max_scaler.transform(y_test)


# In[86]:


#Decision Tree
DTR = DecisionTreeRegressor(max_depth=30)
DTR.fit(X_train,y_train)
y_pre_DTR = DTR.predict(x_test)
DTR_score=r2_score(y_test,y_pre_DTR)
print(DTR_score)
print(mean_squared_error(y_test,y_pre_DTR))


# In[87]:


#SVR
SUPPOT = SVR()
SUPPOT.fit(X_train,y_train)
y_pre_SVR = SUPPOT.predict(x_test)
SVR_score=r2_score(y_test,y_pre_SVR)
print(SVR_score)
print(mean_squared_error(y_test,y_pre_SVR))


# In[88]:


#Randomforest
rf = RandomForestRegressor(n_estimators=30,max_depth=20)
rf.fit(X_train,y_train)
y_pre_rf = rf.predict(x_test)
rf_score=r2_score(y_test,y_pre_rf)
print(rf_score)
print(mean_squared_error(y_test,y_pre_rf))


# In[89]:


#AdaBoostRegression
adb = AdaBoostRegressor()
adb.fit(X_train,y_train)
y_pre_adb = adb.predict(x_test)
adb_score=r2_score(y_test,y_pre_adb)
print(adb_score)
print(mean_squared_error(y_test,y_pre_adb))


# In[90]:


#GradientBoostingRegression
gbtr = GradientBoostingRegressor()
gbtr.fit(X_train,y_train)
y_pre_gbtr = gbtr.predict(x_test)
gbtr_score=r2_score(y_test,y_pre_gbtr)
print(gbtr_score)
print(mean_squared_error(y_test,y_pre_gbtr))


# In[ ]:





# In[91]:


from sklearn.neural_network import MLPRegressor


# In[92]:


train_score = []
test_score = []
layers = list(range(5,300,5)) 
for i in layers:
    MLP = MLPRegressor(activation = 'relu', 
                       solver = 'sgd', 
                       hidden_layer_sizes = (i,), 
                       alpha=1e-2, 
                       max_iter = 400,
                       learning_rate_init = 0.1)
    MLP.fit(X_train,y_train)
    y_pre=MLP.predict(x_test)
    test_score.append(mean_squared_error(y_test,y_pre))


# In[93]:


# Plot scores
plt.plot(layers,test_score,'-',label = 'test set')
plt.xlabel('layers')
plt.ylabel('score')
plt.legend()
plt.xscale("log")


# In[94]:


min(test_score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




