
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import preprocessing


# In[2]:

train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')

train_df.head(n = 200)
id_test = test_df['ID'].values


# In[3]:

train_df.info()
print("##################################")
test_df.info()


# In[4]:

train_df.target.value_counts().plot(kind = 'bar')


# In[5]:

a_df, tmp_indexer = pd.factorize(train_df.v3)


# In[6]:

train_df.v3.value_counts()


# In[7]:

# fill in missing values
for feature in test_df.columns:
    # for float feature, fill in the missing values with the mean
    if train_df[feature].dtype == 'float64':
        train_df[feature].loc[pd.isnull(train_df[feature])] = train_df[feature].median()
        test_df[feature].loc[pd.isnull(test_df[feature])]   = test_df[feature].median()
        
    # for string feature, fill in the missing value with most occurred value
    if train_df[feature].dtype == 'object':
        train_df[feature], tmp_indexer = pd.factorize(train_df[feature])
        test_df[feature] = tmp_indexer.get_indexer(test_df[feature])
    # for int feature, fill in the missing value with the most occured value
    if train_df[feature].dtype == 'int64':
        train_df[feature].loc[pd.isnull(train_df[feature])] = train_df[feature].value_counts().index[0]
        test_df[feature].loc[pd.isnull(test_df[feature])]   =  test_df[feature].value_counts().index[0]


# In[8]:

train_df.head(n = 200)


# In[9]:

test_df.head(n = 100)


# In[10]:

x = train_df.drop(['ID', 'target'], axis = 1)
y = train_df['target']

n = int(x.shape[0] * 0.95)

x_train = x.iloc[1:n, :]
y_train = y.iloc[1:n]

x_vali  = x.iloc[n:, :]
y_vali  = y.iloc[n:]

x_test  = test_df.drop('ID', axis = 1)


# In[11]:

from sklearn.metrics import log_loss


# In[56]:

from sklearn.ensemble import ExtraTreesClassifier


# In[57]:

etc = ExtraTreesClassifier(n_estimators=1500,criterion= 'entropy', max_depth = 50)
etc.fit(x_train, y_train)


# In[58]:

etc.score(x_vali, y_vali)


# In[59]:

etc.score(x_train, y_train)


# In[60]:

log_loss(y_vali, etc.predict_proba(x_vali)[:, 1])


# In[61]:

y_pred = etc.predict_proba(x_test)
#print y_pred

pd.DataFrame({"ID": id_test, "PredictedProb": y_pred[:,1]}).to_csv('BNP_ExtraTree.csv',index=False)


# In[ ]:



