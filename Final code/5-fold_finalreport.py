#!/usr/bin/env python
# coding: utf-8

# In[50]:


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


# In[51]:


from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[65]:


from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


# In[53]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[66]:


df = pd.read_excel('airfoil_self_noise.xlsx')
df.head()

df.shape   
df.info() 
X_input = df.dropna()
scaler = MinMaxScaler()

X_input = scaler.fit_transform(X_input)
X_input = pd.DataFrame(X_input)
X_input.columns = ["Frequency","Angle of attack","Chord length","Free-stream velocity","Suction side displacement thickness","label","Scaled sound pressure level"] #添加自定义的columns的名字


# In[67]:


X = X_input.iloc[:,0:6]
y = X_input.iloc[:,6]


# In[68]:


X


# In[69]:


y


# In[70]:


list1=[[0,330],[331,645],[646,934],[935,1225],[1226,1503]]
a=[]
for i in list1:
    a.append(i[0])
print(a)


# In[73]:


ols_score=[]
ridge_Training_score=[]
ridge_Testing_score=[]
lasso_Training_score=[]
lasso_Testing_score=[]
DTRscore=[]
SVRscore=[]
RFscore=[]
adbscore=[]
gbtrscore=[]
MLPscore=[]
for i in list1:
    X_test=X.iloc[i[0]:i[1]]
    y_test=y.iloc[i[0]:i[1]]
    X_train= X.drop(X.index[i[0]:i[1]])
    y_train= y.drop(y.index[i[0]:i[1]])
    x = sm.add_constant(X_train) 
    model = sm.OLS(y_train, x).fit() 
    ols_score.append(mean_squared_error(y_test,model.predict(sm.add_constant(X_test))))
    
    ridge=Ridge(alpha=0.4)
    ridge.fit(X_train,y_train)
    ridge_Training_score.append(mean_squared_error(y_train,ridge.predict(X_train)))
    ridge_Testing_score.append(mean_squared_error(y_test,ridge.predict(X_test)))

    lasso_cv = LassoCV(cv=5)
    lasso_cv.fit(X_train, y_train)
    train_score=lasso_cv.score(X_train, y_train)
    test_score = lasso_cv.score(X_test, y_test)
    lasso_Training_score.append(mean_squared_error(y_train,lasso_cv.predict(X_train)))
    lasso_Testing_score.append(mean_squared_error(y_test,lasso_cv.predict(X_test)))
    
    DTR = DecisionTreeRegressor(max_depth=30)
    DTR.fit(X_train,y_train)
    y_pre_DTR = DTR.predict(X_test)
   # DTR_score=r2_score(y_test,y_pre_DTR)
    DTRscore.append((mean_squared_error(y_test,y_pre_DTR)))
    
    SUPPOT = SVR()
    SUPPOT.fit(X_train,y_train)
    y_pre_SVR = SUPPOT.predict(X_test)
    SVRscore.append((mean_squared_error(y_test,y_pre_SVR)))

    
    rf = RandomForestRegressor(n_estimators=30,max_depth=20)
    rf.fit(X_train,y_train)
    y_pre_rf = rf.predict(X_test)
    #rf_score=r2_score(y_test,y_pre_rf)
    RFscore.append((mean_squared_error(y_test,y_pre_rf)))
    
    adb = AdaBoostRegressor()
    adb.fit(X_train,y_train)
    y_pre_adb = adb.predict(X_test)
    #adb_score=r2_score(y_test,y_pre_adb)
    adbscore.append((mean_squared_error(y_test,y_pre_adb)))
    
    gbtr = GradientBoostingRegressor()
    gbtr.fit(X_train,y_train)
    y_pre_gbtr = gbtr.predict(X_test)
    #gbtr_score=r2_score(y_test,y_pre_gbtr)
    gbtrscore.append((mean_squared_error(y_test,y_pre_gbtr)))
    
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
        y_pre=MLP.predict(X_test)
        test_score.append(mean_squared_error(y_test,y_pre))
    MLPscore.append(min(test_score))


# In[74]:


print(np.mean(ols_score))
print(np.mean(ridge_Training_score))
print(np.mean(ridge_Testing_score))
print(np.mean(lasso_Training_score))
print(np.mean(lasso_Testing_score))


# In[75]:


print(ols_score)
print((ridge_Training_score))
print((ridge_Testing_score))
print((lasso_Training_score))
print((lasso_Testing_score))


# In[62]:


print((DTRscore))
print((SVRscore))
print((RFscore))
print((adbscore))
print((gbtrscore))
print((MLPscore))


# In[63]:


print(np.mean(DTRscore))
print(np.mean(SVRscore))
print(np.mean(RFscore))
print(np.mean(adbscore))
print(np.mean(gbtrscore))
print(np.mean(MLPscore))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




