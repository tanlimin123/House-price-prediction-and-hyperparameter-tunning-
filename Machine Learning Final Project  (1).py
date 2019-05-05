#!/usr/bin/env python
# coding: utf-8

# 

# # Predicting house sale prices for King County, which includes Seattle.
# 

# In[ ]:





# In[150]:


#Using linear regression to predict the house pricing.At the same time, the price is divided into cheap, general, and expensive. The price tag is classified by machine learning.


# In[204]:



##import the data and package
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
import seaborn as sns
from scipy.stats import pearsonr

data = pd.read_csv('kc_house_data.csv')


# In[7]:


#overview the data to check if there are outlier 
data.describe()


# In[9]:


#check the missing value
data.isnull().sum()
#if there are some missing value 
# df = df.dropna(how='any')


# In[5]:


str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in data.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = data.columns.difference(str_list) 
# Create Dataframe containing only numerical features
data_num = data[num_list]
f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation of features')
# Draw the heatmap using seaborn
#sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="PuBuGn", linecolor='k', annot=True)
sns.heatmap(data_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=True)


# In[10]:


#caculate the correlation value between the feature and the target and do the feature selection for linear model
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)


# In[21]:


from sklearn import linear_model


regr = linear_model.LinearRegression()
new_data = data[['sqft_living','grade', 'sqft_above', 'sqft_living15','bathrooms','view','sqft_basement','lat','bedrooms']]


# In[22]:


X = new_data.values
y = data.price.values



# In[224]:


from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#Load boston housing dataset as an example

# X = boston["data"]
# Y = boston["target"]
# names = boston["feature_names"]
# rf = RandomForestRegressor()
# rf.fit(X, Y)
# print "Features sorted by their score:"
# print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
#              reverse=True)


# In[24]:


regr.fit(X_train, y_train)
print(regr.predict(X_test))


# In[17]:


regr.score(X_test, y_test)


# In[18]:


# Calculate the Root Mean Squared Error
print("RMSE: %.2f"
      % math.sqrt(np.mean((regr.predict(X_test) - y_test) ** 2)))


# In[162]:


df=pd.read_csv('kc_house_data.csv',nrows=1000)


# In[163]:


#encoding the price and divide it into three lable cheap, general, expensive
df['price_level']=pd.cut((df['price'].values),3, labels=[" cheap", "general", "expensive"])


# In[164]:


target = 'price_level'
feature='sqft_living','grade', 'sqft_above', 'sqft_living15','bathrooms','view','sqft_basement','lat','bedrooms'


# In[165]:


#drop the column that is meaningless

X = df.drop(columns=[target,'price','id','date','long','zipcode','yr_renovated','sqft_lot'])
y = df[target]


# In[166]:


#check the categorical variable
for j in range(X.shape[1]):
    print(X.columns[j] + ':')
    print(X.iloc[:, j].value_counts(), end='\n\n')


# In[167]:


import pandas as pd

# Implement me
X = pd.get_dummies(X, columns=X.columns)
X.head()


# In[168]:


y.value_counts()


# In[169]:


from sklearn.preprocessing import LabelEncoder

# Implement me
le = LabelEncoder()
y = le.fit_transform(y)

pd.DataFrame(data=y, columns=[target])[target].value_counts()


# In[170]:


from imblearn.over_sampling import RandomOverSampler

# RandomOverSampler (with random_state=0)
# Implement me
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_sample(X, y)

pd.DataFrame(data=y, columns=[target])[target].value_counts()


# In[171]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

clfs = {'lr': LogisticRegression(random_state=0),
        'dt': DecisionTreeClassifier(random_state=0),
        'rf': RandomForestClassifier(random_state=0),
        'knn': KNeighborsClassifier(),
        'gnb': GaussianNB()}


# In[172]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe_clfs = {}

for name, clf in clfs.items():
    # Implement me
    pipe_clfs[name] = Pipeline([('StandardScaler', StandardScaler()), ('clf', clf)])
    


# In[173]:


param_grids = {}


# In[174]:


C_range = [10 ** i for i in range(-4, 5)]

param_grid = [{'clf__multi_class': ['ovr'], 
               'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
               'clf__C': C_range},
              
              {'clf__multi_class': ['multinomial'],
               'clf__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
               'clf__C': C_range}]

# Implement me
param_grids['lr'] = param_grid


# In[ ]:





# In[175]:


param_grid = [{'clf__min_samples_split': [2, 10, 30],
               'clf__min_samples_leaf': [1, 10, 30]}]

# Implement me
param_grids['dt'] = param_grid


# In[176]:


param_grid = [{'clf__n_estimators': [2, 10, 30],
               'clf__min_samples_split': [2, 10, 30],
               'clf__min_samples_leaf': [1, 10, 30]}]

# Implement me
param_grids['rf'] = param_grid


# In[ ]:





# In[177]:


param_grid = [{'clf__n_neighbors': list(range(1, 11))}]

# Implement me
param_grids['knn'] = param_grid


# In[178]:


param_grid = [{'clf__var_smoothing': [10 ** i for i in range(-10, -7)]}]

# Implement me
param_grids['gnb'] = param_grid


# In[179]:


import warnings
warnings.filterwarnings('ignore')


# In[180]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# The list of [best_score_, best_params_, best_estimator_]
best_score_param_estimators = []

# For each classifier
for name in pipe_clfs.keys():
    # GridSearchCV
    # Implement me
    gs = GridSearchCV(estimator=pipe_clfs[name],
                      param_grid=param_grids[name],
                      scoring='accuracy',
                      n_jobs=-1,
                      cv=StratifiedKFold(n_splits=5,
                                         shuffle=True,
                                         random_state=0))
    # Fit the pipeline
    # Implement me
    gs = gs.fit(X, y)
    
    # Update best_score_param_estimators
    best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])


# In[182]:


# Sort best_score_param_estimators in descending order of the best_score_
# Implement me
best_score_param_estimators = sorted(best_score_param_estimators, key=lambda x : x[0], reverse=True)

# For each [best_score_, best_params_, best_estimator_]
for best_score_param_estimator in best_score_param_estimators:
    # Print out [best_score_, best_params_, best_estimator_], where best_estimator_ is a pipeline
    # Since we only print out the type of classifier of the pipeline
    print([best_score_param_estimator[0], best_score_param_estimator[1], type(best_score_param_estimator[2].named_steps['clf'])], end='\n\n')


# In[ ]:





# 

# In[ ]:




