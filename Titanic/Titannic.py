
# coding: utf-8

# In[77]:

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn.ensemble import RandomForestClassifier


# In[78]:

train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')
# preview data
train_df.head()


# In[79]:

train_df.info()
print('-------------------')
test_df.info()


# In[80]:

# drop PassengerId, Name, Ticket; they are not useful
# drop Cabin also due to too many missing data
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
test_df  = test_df.drop(['Name', 'Ticket', 'Cabin'], axis = 1)


# In[81]:

# In the training data, "Embarked" has two missing values, so we use the most frequent vlaue to fill in them
train_df['Embarked'].value_counts()


# In[82]:

# fill in the two missing values with the most frequent value 'S'
train_df['Embarked'] = train_df['Embarked'].fillna("S")


# In[83]:

# in the test data, "Fare" has one missing value. Fill in it with the median
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)
# convert from float to int, since we will use random forest later
#titanic_df['Fare'] = titanic_df['Fare'].astype(int)
#test_df['Fare']    = test_df['Fare'].astype(int)


# In[84]:

# in the training and test data, "age" has a lot of missing vlaues
# we fill in these missing values by random integers

# mean and std of training and test data
age_mean_train = train_df['Age'].mean()
age_std_train  = train_df['Age'].std()
n_nan_age_train = train_df['Age'].isnull().sum()

age_mean_test = test_df['Age'].mean()
age_std_test  = test_df['Age'].std()
n_nan_age_test = test_df['Age'].isnull().sum()

# random numbers between mean-std and mean+std
rand_age_train = np.random.randint(low = age_mean_train - age_std_train, high = age_mean_train + age_std_train, size = n_nan_age_train)
rand_age_test  = np.random.randint(low = age_mean_test  - age_std_test , high = age_mean_test  + age_std_test , size = n_nan_age_test )

# fill in the missing values
train_df['Age'][train_df['Age'].isnull()] = rand_age_train
test_df['Age'][test_df['Age'].isnull()]   = rand_age_test


# In[85]:

# we assume siblings, spouse, parents, children are equally important
# so we merge "SibSp" and "Parch" into a single feature 'Family', which means whether a passenger has familily member aboard
train_df['Family'] = train_df['SibSp'] + train_df['Parch']
test_df['Family'] = test_df['SibSp'] + test_df['Parch']
train_df['Family'].loc[train_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] > 0] = 1

# drop Parch and SibSp
train_df = train_df.drop(['SibSp', 'Parch'], axis = 1)
test_df  = test_df.drop(['SibSp', 'Parch'], axis = 1)


# In[86]:

# convert Sex to integer, male --> 1, female --> 0
train_df['Sex'].loc[train_df['Sex'] == 'male'] = 1
train_df['Sex'].loc[train_df['Sex'] == 'female'] = 0
test_df['Sex'].loc[test_df['Sex'] == 'male'] = 1
test_df['Sex'].loc[test_df['Sex'] == 'female'] = 0


# In[87]:

# convert Pclass to integer, S-->0, C-->1, Q-->2
train_df['Embarked'].loc[train_df['Embarked'] == 'S'] = 0
train_df['Embarked'].loc[train_df['Embarked'] == 'C'] = 1
train_df['Embarked'].loc[train_df['Embarked'] == 'Q'] = 2

test_df['Embarked'].loc[test_df['Embarked'] == 'S'] = 0
test_df['Embarked'].loc[test_df['Embarked'] == 'C'] = 1
test_df['Embarked'].loc[test_df['Embarked'] == 'Q'] = 2


# In[88]:

# define traing and test data
X_train = train_df.drop('Survived', axis = 1)
Y_train = train_df['Survived']
X_test  = test_df.drop('PassengerId', axis = 1)


# In[89]:

# build random forest model
model = RandomForestClassifier(n_estimators = 100)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
model.score(X_train, Y_train)


# In[90]:

submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],
                          'Survived': Y_pred})
submission.to_csv('prediction.csv', index = False)


# In[ ]:



