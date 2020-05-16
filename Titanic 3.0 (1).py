#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import train and test data

# In[231]:


#Read train and test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[230]:


print(train.info()
     ,train.describe())
train.head()


# # Explorative Analysis

# In[7]:


#Women VS Men
male_data = train[train['Sex']=='male']
female_data = train[train['Sex']=='female']

fig, ax =plt.subplots(1,2)

sns.countplot(x='Sex',hue='Survived',data=male_data,ax=ax[0])
sns.countplot(x='Sex',hue='Survived',data=female_data,ax=ax[1])


# In[8]:


#Age groups

def custom_round(x, base=10):
    return int(base * round(float(x)/base))

sns.countplot(train['Age'].dropna().apply(lambda x: custom_round(x, base=10)),hue=train['Survived'])


# In[9]:


#Classes
sns.countplot(train['Pclass'],hue=train['Survived'])


# In[10]:


#Women VS Men
fig, ax =plt.subplots(1,2)

sns.countplot(x='Pclass',hue='Survived',data=male_data,ax=ax[0]).set_title("Men")
sns.countplot(x='Pclass',hue='Survived',data=female_data,ax=ax[1]).set_title("Women")


# In[11]:


#Siblings/Spouse
sns.countplot(x='SibSp',hue='Survived',data=train)


# In[12]:


#Create Died column
train['Died'] = 1 - train['Survived']

#Relative Sibling/Spouse
train.groupby('SibSp').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()


# In[13]:


#Relative Parch
train.groupby('Parch').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()


# In[14]:


#Fare
plt.figure(figsize=[8,5])
plt.hist([train[train['Survived'] == 1]['Fare'], train[train['Survived'] == 0]['Fare']], bins = 30, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Nr of passengers')
plt.legend()
plt.show()


# # Clean Missing Age Values

# In[232]:


#Create means for different Pclass to handle missing values
mean_age_1 = train[train['Pclass']==1]['Age'].mean()
mean_age_2 = train[train['Pclass']==2]['Age'].mean()
mean_age_3 = train[train['Pclass']==3]['Age'].mean()

#If age is missing asign a mean
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

#Replace missing age values in train
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

#Replace missing age values in test
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
    


# ## Clean Fare

# In[233]:


#Check for null values
test[test['Fare'].isnull()]

#Calculate avg fare for Pclass = 3
mean_fare = train[train['Pclass']==3]['Fare'].mean()

#Fill in avg fare for missing fares
test['Fare'].fillna(mean_fare,inplace=True)


# ## Create Title
# 

# In[234]:


#Extract title from name
def get_title(string):
    
    x = string.split(',')
    title_split = x[1].split('.')
    return title_split[0]

#Create new column
train['Title'] = train['Name'].apply(get_title)
test['Title'] = test['Name'].apply(get_title)

#Remove whitespace
train['Title'] = train['Title'].apply(lambda x:x.lstrip())
test['Title'] = test['Title'].apply(lambda x:x.lstrip())


# In[236]:


#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
   title=x['Title']
   if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir','Dona']:
       return 'Noble'
   elif title in ['Countess', 'Mme','the Countess']:
       return 'Mrs'
   elif title in ['Mlle', 'Ms','Lady']:
       return 'Miss'
   elif title =='Dr':
       if x['Sex']=='Male':
           return 'Mr'
       else:
           return 'Mrs'
   else:
       return title

#Apply replace_titles to test and train
train['Title']=train.apply(replace_titles, axis=1)
test['Title']=test.apply(replace_titles, axis=1)


# In[249]:


#Countplot of new titles
sns.countplot(test['Title'])


# # Setting features and creating dummy data

# In[254]:


#Decide on features for input to model
features = ['Age','Sex','Pclass','Parch','SibSp','Fare','Title','Embarked']
train_X = train[features]
train_y = train['Survived']
test_X = test[features]


# In[266]:


train_X.head()


# In[256]:


#Create dummy variables for Sex column
sex_dum = pd.get_dummies(train_X['Sex'],drop_first=True)
train_X.drop('Sex',axis=1,inplace=True)
train_X = pd.concat([train_X,sex_dum],axis=1)


sex_dum = pd.get_dummies(test['Sex'],drop_first=True)
test_X.drop('Sex',axis=1,inplace=True)
test_X = pd.concat([test_X,sex_dum],axis=1)


# In[257]:


#Create dummy variables for Title column
title_dum = pd.get_dummies(train_X['Title'],drop_first=True)
train_X.drop('Title',axis=1,inplace=True)
train_X = pd.concat([train_X,title_dum],axis=1)


title_dum = pd.get_dummies(test_X['Title'],drop_first=True)
test_X.drop('Title',axis=1,inplace=True)
test_X = pd.concat([test_X,title_dum],axis=1)


# In[265]:


#Create dummy variables for Embarked column
embark_dum = pd.get_dummies(train['Embarked'],drop_first=True)
train_X.drop('Embarked',axis=1,inplace=True)
train_X = pd.concat([train_X,embark_dum],axis=1)


embark_dum = pd.get_dummies(test_X['Embarked'],drop_first=True)
test_X.drop('Embarked',axis=1,inplace=True)
test_X = pd.concat([test_X,embark_dum],axis=1)


# In[261]:


from sklearn.ensemble import RandomForestClassifier


# In[267]:


rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)


# In[268]:


#Fit the model
rfc.fit(train_X,train_y)


# In[270]:


#Create prediction for test data based on model
y_test = rfc.predict(test_X)


# In[271]:


#Output submission
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_test})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




