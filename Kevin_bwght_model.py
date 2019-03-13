# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:14:19 2019

@author: Kevin
"""

# Loading Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # regression modeling

from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import sklearn.metrics # more metrics for model performance evaluation

# Loading excel dataset

file = 'birthweight_feature_set.xlsx'

health = pd.read_excel(file)
   
# Impute missing data with median
birth_weight = pd.DataFrame.copy(health)

for col in birth_weight:
    
    """ Impute missing values using the median of each column """
    
    if birth_weight[col].isnull().any():
        
        col_median = birth_weight[col].median()
        
        birth_weight[col] = birth_weight[col].fillna(col_median).round(3)


birth_weight.corr().round(2)

#creating new variables
# Creating binary variable 'drinker'
birth_weight['drinker'] = (birth_weight.drink > 0).astype('int')

# Creating binary variable 'smoker'
birth_weight['smoker'] = (birth_weight.cigs > 0).astype('int')

birth_weight['trasher'] = birth_weight.drinker+birth_weight.smoker

birth_weight['over_54'] = (birth_weight.mage > 54).astype('int')

birth_weight['f_over_49'] = (birth_weight.fage > 49).astype('int')

birth_weight['risky'] = (birth_weight.npvis > 14).astype('int')

#Steph's Variables with Age Ranges
counter = 0
birth_weight['cage'] = 0

for value in birth_weight['mage']:
    if value < 30:
        if birth_weight.loc[counter, 'fage'] < 65:
            birth_weight.loc[counter,'cage'] = 1
        elif birth_weight.loc[counter,'fage'] >= 65:
            birth_weight.loc[counter,'cage'] = 2      
    elif value < 35:
        if birth_weight.loc[counter, 'fage'] < 40:
            birth_weight.loc[counter,'cage'] = 1
        elif birth_weight.loc[counter,'fage'] < 65:
            birth_weight.loc[counter,'cage'] = 2 
        elif birth_weight.loc[counter,'fage'] >= 65:
            birth_weight.loc[counter,'cage'] = 3
    elif value < 45:
        if birth_weight.loc[counter,'fage'] < 40:
            birth_weight.loc[counter,'cage'] = 2 
        elif birth_weight.loc[counter,'fage'] >= 40:
            birth_weight.loc[counter,'cage'] = 3
    else:
        birth_weight.loc[counter,'cage'] = 3
    counter+=1

#getting the log value of father's education

birth_weight['log_feduc']= np.log(birth_weight['feduc'])

#checking correlation with new variables
birth_weight.corr()['bwght'].sort_values().round(2)

#creating new df with low and normal birthweights
birth_weight2 = birth_weight[birth_weight.bwght < 4500]

birth_weight2.corr()['bwght'].sort_values().round(2)


#modeling

health_data = birth_weight.drop(['bwght','omaps','fmaps'],
                              axis = 1)

health_target = birth_weight.loc[: ,'bwght']

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(health_data,
                                                    health_target,
                                                    test_size = 0.1,
                                                    random_state = 508)

health_train = pd.concat([X_train, y_train], axis = 1)

#ols model with statsmodel
ols_1 = smf.ols(formula = """bwght ~ cigs +
                                     drink +
                                     fage+
                                     over_54+
                                     log_feduc                                      
                                     """,
                                     data = health_train)

results_ols_1 = ols_1.fit()

print(results_ols_1.summary())


#model with sci kit learn 

health_data = birth_weight.loc[:,['cigs',
                                'drink',
                                'over_54',
                                'fage',
                                'log_feduc',]]

health_target = birth_weight.loc[: ,'bwght']

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(health_data,
                                                    health_target,
                                                    test_size = 0.1,
                                                    random_state = 508)

from sklearn.linear_model import LinearRegression

# Instantiate
lr = LinearRegression()

# Fit
lr_fit = lr.fit(X_train, y_train)

# Predict[]
lr_pred = lr_fit.predict(X_test)

# Score
y_score_ols = lr_fit.score(X_test, y_test)

print(y_score_ols) 
print('Training Score', lr_fit.score(X_train, y_train).round(4))
print('Testing Score:', lr_fit.score(X_test, y_test).round(4))
#score is 0.61258
#K-Nearest Neighbors

#Determining the best number of neighbors

training_accuracy = []
test_accuracy = []


# Building a visualization to check to see  1 to 50
neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


# Plotting the visualization
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()



########################
# What is the optimal number of neighbors?
########################

print(test_accuracy.index(max(test_accuracy)))


# Instantiate
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 13)

# Fit
knn_reg.fit(X_train, y_train)

# Predict
y_pred = knn_reg.predict(X_test)

# Score
y_score_knn = knn_reg.score(X_test, y_test)

print(y_score_knn)#score is 0.4838036631622624

#decision tree model

from sklearn.tree import DecisionTreeRegressor # Regression trees

# Instantiate
tree_reg = DecisionTreeRegressor(criterion = 'mse',
                                 min_samples_leaf = 20,
                                 random_state = 508)

# Fit
tree_reg.fit(X_train, y_train)

# Predict
y_pred = tree_reg.predict(X_test)

# Score
y_score_tree = tree_reg.score(X_test, y_test)

print(y_score_tree) #score is 0.5153973414990176

#Lasso Regression Model
# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4,normalize=True)

# Fit the regressor to the data
lasso.fit(X_train,y_train)

#Predict
y_pred_lasso = lasso.predict(X_test)

#Score
y_score_lasso = lasso.score(X_test,y_test)

print(y_score_lasso)

#Ridge Regression Model

from sklearn.linear_model import Ridge

# Instantiate a lasso regressor: lasso
ridge = Ridge(alpha=0.4,normalize=True)

# Fit the regressor to the data
ridge.fit(X_train,y_train)

#Predict
y_pred_ridge = ridge.predict(X_test)

#Score
y_score_ridge = ridge.score(X_test,y_test)

print(y_score_ridge)

