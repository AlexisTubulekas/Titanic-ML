#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import train and test data

# In[149]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[6]:


train.describe()


# In[110]:


train.isnull().sum()


# In[68]:


train.head()
train['Died'] = 1 - train['Survived']


# # Analysis

# In[20]:


#Women VS Men
male_data = train[train['Sex']=='male']
female_data = train[train['Sex']=='female']

fig, ax =plt.subplots(1,2)

sns.countplot(x='Sex',hue='Survived',data=male_data,ax=ax[0])
sns.countplot(x='Sex',hue='Survived',data=female_data,ax=ax[1])


# In[50]:


#train['Age'].hist(bins=20)

def custom_round(x, base=10):
    return int(base * round(float(x)/base))

sns.countplot(train['Age'].dropna().apply(lambda x: custom_round(x, base=10)),hue=train['Survived'])


# In[54]:


sns.countplot(train['Pclass'],hue=train['Survived'])


# In[59]:


#Women VS Men
fig, ax =plt.subplots(1,2)

sns.countplot(x='Pclass',hue='Survived',data=male_data,ax=ax[0]).set_title("Men")
sns.countplot(x='Pclass',hue='Survived',data=female_data,ax=ax[1]).set_title("Women")


# In[64]:


sns.countplot(x='SibSp',hue='Survived',data=train)


# In[69]:


train.groupby('SibSp').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()


# In[95]:


train.groupby('Parch').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()


# # Clean Age

# In[150]:


mean_age_1 = train[train['Pclass']==1]['Age'].mean()
mean_age_2 = train[train['Pclass']==2]['Age'].mean()
mean_age_3 = train[train['Pclass']==3]['Age'].mean()

def impute_age(x):
    
    if pd.isnull(x[0]):
    
        if x[1]==1:
            return mean_age_1

        if x[1]==2:
            return mean_age_2

        else:
            return mean_age_3
        
    else: 
        return x[0]

#train
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

#test
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
    


# In[ ]:





# # Dummy data of features

# In[151]:


features = ['PassengerId','Age','Sex','Parch','SibSp']
train_X = train[features]
train_y = train['Survived']

#test
test = test[features]


# In[152]:


train_X.head()


# In[153]:


sex_dum = pd.get_dummies(train_X['Sex'],drop_first=True)
train_X.drop('Sex',axis=1,inplace=True)
train_X = pd.concat([train_X,sex_dum],axis=1)

#test
sex_dum = pd.get_dummies(test['Sex'],drop_first=True)
test.drop('Sex',axis=1,inplace=True)
test = pd.concat([test,sex_dum],axis=1)


# In[160]:


from sklearn.ensemble import RandomForestClassifier


# In[161]:


rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)


# In[162]:


rfc.fit(train_X,train_y)


# In[163]:


y_test = rfc.predict(test)


# In[164]:


test.head()


# In[165]:


output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_test})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




