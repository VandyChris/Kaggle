
# coding: utf-8

# In[60]:

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[61]:

dataset = pd.read_csv('train.csv')


# In[62]:

dataset.head(n = 5)


# In[63]:

y = dataset.iloc[:, 0].values
x = dataset.iloc[:, 1:].values


# In[68]:

n_sample = y.shape[0]
n_train = int(n_sample * 0.7)


# In[69]:

x_train = x[:n_train, :]
y_train = y[:n_train]
x_vali  = x[n_train:, :]
y_vali  = y[n_train:]


# In[70]:

dataset2 = pd.read_csv('test.csv')


# In[71]:

x_test = dataset2.values


# In[72]:

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)


# In[73]:

rf.score(x_vali, y_vali)


# In[74]:

y_predict = rf.predict(x_test)


# In[75]:

n_test = x_test.shape[0]


# In[86]:

submission = pd.DataFrame({'ImageId':np.arange(n_test)+1, 'label':y_predict})


# In[87]:

submission.to_csv('submission_RandomForest.csv', index = False)


# In[ ]:



