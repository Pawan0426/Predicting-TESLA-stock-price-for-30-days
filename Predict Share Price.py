#!/usr/bin/env python
# coding: utf-8

# In[16]:


pip install quandl


# # Predicting stock price using machine learning

# In[17]:


import quandl
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# ## Get stock data for Tesla

# df = quandl.get('WIKI/TSLA')

# In[18]:


df.tail()


# ## Getting the dajusted close price for tesla

# In[19]:


df = df[["Adj. Close"]]


# In[20]:


print(df)


# ### Variable predicting 'n' days out in the future

# In[25]:


forecast_out = 30
# creating dependent and target variables this will be shifted 'n' units up
df['prediction'] = df[["Adj. Close"]].shift(-forecast_out)
print(df.tail())


# In[29]:


# Creating independent data set called X and converting data set to numpy array 
X = np.array(df.drop(['prediction'],1))
# Remove the last 'n' row

X = X[:-forecast_out]
print(X)


# In[31]:


# creating dependent data set we call it y
y = np.array(df['prediction'])

# getting all the y valuse except last'n' rows

y = y[:-forecast_out]
print(y)


# ### Splitting data into 80% training and 20% testing

# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[33]:


# creat and train the support vector regressor
svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)
svr_rbf.fit(X_train, y_train)


# ### Testing model: Score returns the cofficient of determination R**2 of the prediction. The best possible score is 1.0

# In[36]:


svm_confidence = svr_rbf.score(X_test,y_test)
print("SVM confidence: ", svm_confidence)


# ### Creat and train linear regression

# In[37]:


lr = LinearRegression()
lr.fit(X_train,y_train)


# ### Testing model: Score returns the cofficient of determination R**2 of the prediction. The best possible score is 1.0

# In[38]:


lr_confidence = lr.score(X_test,y_test)
print("lr confidence: ", lr_confidence)


# ### Set X_forecast is equal to last 30 rows of the original data stes from Adj. Close column.

# In[42]:


X_forecast = np.array(df.drop(['prediction'],1))[-forecast_out:]
print(X_forecast)


# ### Print linear regresson model prediction for 'n' days.

# In[44]:


lr_prediction = lr.predict(X_forecast)
print(lr_prediction)


# ### Print support vector regressor model prediction for 'n' days.

# In[45]:


svm_prediction = svr_rbf.predict(X_forecast)
print(svm_prediction)


# In[49]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


predictions = lr.predict(X_test)


# In[50]:


plt.scatter(y_test,predictions)


# In[51]:


sns.distplot((y_test-predictions),bins=50);


# In[52]:


from sklearn import metrics


# Regression Evaluation Metrics
# Here are three common evaluation metrics for regression problems:
# 
# Mean Absolute Error (MAE) is the mean of the absolute value of the errors.
#  
# Mean Squared Error (MSE) is the mean of the squared errors.
#  
# Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors.
#  
# Comparing these metrics:
# 
# MAE is the easiest to understand, because it's the average error.
# 
# MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# 
# RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.
# 
# All of these are loss functions, because we want to minimize them.

# In[53]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




