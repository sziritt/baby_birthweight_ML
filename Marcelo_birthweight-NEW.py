# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:51:20 2019
@author: Stephy Zirit
Working Directory:
/Users/Stephy Zirit/Documents/HULT/Module B/Machine Learning
Purpose:
    Assigment 1 
    Certain factors contribute to the health of a newborn
    baby. One such health measure is birth weight.
    Countless studies have identified factors, both
    preventative and hereditary, that lead to low birth
    weight.
    Your team has been hired as public health
    consultants to analyze and model an infant’s birth
    weight based on such characteristics.
"""

###############################################################################
##### DATA DICTIONARY
###############################################################################
"""
|----------|---------|---------------------------------|
| variable | label   | description                     |
|----------|---------|---------------------------------|
| 1        | mage    | mother's age                    |
| 2        | meduc   | mother's educ                   |
| 3        | monpre  | month prenatal care began       |
| 4        | npvis   | total number of prenatal visits |
| 5        | fage    | father's age, years             |
| 6        | feduc   | father's educ, years            |
| 7        | omaps   | one minute apgar score          |
| 8        | fmaps   | five minute apgar score         |
| 9        | cigs    | avg cigarettes per day          |
| 10       | drink   | avg drinks per week             |
| 11       | male    | 1 if baby male                  |
| 12       | mwhte   | 1 if mother white               |
| 13       | mblck   | 1 if mother black               |
| 14       | moth    | 1 if mother is other            |
| 15       | fwhte   | 1 if father white               |
| 16       | fblck   | 1 if father black               |
| 17       | foth    | 1 if father is other            |
| 18       | bwght   | birthweight, grams              |
|----------|---------|---------------------------------|
"""



###############################################################################
##### LIBRARIES AND SET UP OF FILE 
###############################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file = 'birthweight_feature_set.xlsx'
birth_weight = pd.read_excel(file)

birth_weight = birth_weight.drop(['omaps','fmaps'],axis=1)

###############################################################################
##### MISSING VALUES
###############################################################################
print(birth_weight.isnull().sum()) 


# Flagging missing values
for col in birth_weight:
    if birth_weight[col].isnull().astype(int).sum() > 0:
        birth_weight['m_'+col] = birth_weight[col].isnull().astype(int)


# Filling NAs in 'npvis' , 'meduc' and 'feduc' with their MEDIANs

birth_weight.npvis = birth_weight.npvis.fillna(birth_weight.npvis.median())
birth_weight.meduc = birth_weight.meduc.fillna(birth_weight.meduc.median())
birth_weight.feduc = birth_weight.feduc.fillna(birth_weight.feduc.median())

# Rechecking NAs:
print(birth_weight.isnull().sum()) 



#########################################
# NEW VARIABLES - FEATURE ENGINEERING:

# Creating binary variable 'drinker'
birth_weight['drinker'] = (birth_weight.drink > 0).astype('int')

# Creating binary variable 'smoker'
birth_weight['smoker'] = (birth_weight.cigs > 0).astype('int')

birth_weight['trasher'] = birth_weight.drinker+birth_weight.smoker

birth_weight.loc[birth_weight.trasher == 2,'trasher'] = 4

#outliers:

# Creating binary variable 'out_drink' # original=12
birth_weight['out_drink'] = (birth_weight.drink > 8).astype('int')

# Creating binary variable 'out_cigs' # original - no outliers?
birth_weight['out_cigs'] = (birth_weight.cigs > 12).astype('int')

# Creating binary variable 'lo_out_npvis' # original=7
birth_weight['lo_out_npvis'] = (birth_weight.npvis <=10).astype('int')

# Creating binary variable 'hi_out_npvis' # original=15
birth_weight['hi_out_npvis'] = (birth_weight.npvis > 15).astype('int')

# Creating binary variable 'out_mage' # original=60 (2nd = 54)
birth_weight['out_mage'] = (birth_weight.mage > 54).astype('int')

# INCLUDE MOTHER´S AGE > 65
birth_weight['big_out_mage'] = (birth_weight.mage >64).astype('int')*3 

# Creating binary variable 'out_fage' # original=55
birth_weight['out_fage'] = (birth_weight.fage > 46).astype('int')

# Creating binary variable 'mcollege' - mothers who went to college
birth_weight['mcollege'] = (birth_weight.meduc >= 14).astype('int')

# Creating binary variable 'fcollege' - fathers who went to college
birth_weight['fcollege'] = (birth_weight.feduc >= 14).astype('int')


# Creating binary variable 'out_feduc' # original=7
birth_weight['out_feduc'] = (birth_weight.feduc < 7).astype('int')

# Creating binary variable 'out_monpre' # original=4
birth_weight['out_monpre'] = (birth_weight.monpre > 4).astype('int')

# New variable 'visgap' = diff between # prenatal visits & prenatal month of start
birth_weight['visgap'] = birth_weight.npvis-birth_weight.monpre

# New variable 'mage_medu' = ratio mother's age and education
birth_weight['mage_medu'] = birth_weight.mage/birth_weight.meduc

# New variable 'fage_fedu' = ratio father's age and education
birth_weight['fage_fedu'] = birth_weight.fage/birth_weight.feduc



df = birth_weight

#####################
# Adding normalized columns for ['mage','meduc','monpre','npvis','fage','feduc','cigs','drink']

normalize = lambda var : (birth_weight[var]-min(birth_weight[var]))/(max(birth_weight[var])-min(birth_weight[var]))

standardize = lambda var : (birth_weight[var]-birth_weight[var].mean())/birth_weight[var].std()


cont_vars = ['mage','meduc','monpre','npvis','fage','feduc']#,'cigs','drink']

for col in cont_vars:
    birth_weight['norm_'+col] = normalize(col)
    birth_weight['std_'+col] = standardize(col)
    birth_weight['log_'+col] = np.log(birth_weight[col])
    birth_weight['sq_'+col] = birth_weight[col]**2
    birth_weight['cub_'+col] = birth_weight[col]**3

df = birth_weight
#########################################

#########
## K-means clusters

# Clusters for risk in normal weight, low weight and hi weight

from sklearn.cluster import KMeans

data_for_cluster = birth_weight.drop(['bwght'],axis=1)

kmeans = KMeans(n_clusters=3, random_state=0).fit(data_for_cluster) #orig=6

# Check clusters
# kmeans.labels_

# assign new column:
clusters = pd.get_dummies(kmeans.labels_,drop_first=False)
clusters.columns = ['group1','group2','group3']#,'group4','group5']
df = pd.concat([df,clusters],axis=1)

##########################################
# Factor Analysis

#Factor 1 - mother and father's age
# Factor 2 - Prenatal care

from sklearn.decomposition import FactorAnalysis

df_factor = df.drop('bwght',axis=1)

transformer = FactorAnalysis(n_components=2, random_state=0)

X_transformed = transformer.fit_transform(df_factor)

factornames = ['factor1','factor2']

factors = pd.DataFrame(X_transformed,columns=factornames)

df = pd.concat([df,factors],axis=1)

##############################################
# Classes of weights

#(REMOVE FROM MODEL TRAIN-TEST!):

# low, normal, high weight

df['wclass'] = 'norm_weight'
df.loc[df.bwght < 2500,'wclass'] = 'lo_weight'
df.loc[df.bwght > 4000,'wclass'] = 'hi_weight'

weights = pd.get_dummies(df['wclass'],drop_first=False)
df = pd.concat([df,weights],axis=1)

df = df.drop('wclass',axis=1)



###############################################################################
##### EXPLORATORY ANALYSIS
###############################################################################



# Column names
birth_weight.columns

# Displaying the first rows of the DataFrame
print(birth_weight.head())

# Dimensions of the DataFrame
birth_weight.shape

# Information about each variable
birth_weight.info()

# Descriptive statistics
birth_weight.describe().round(2)


# Normal weight (between 2500 and 4000)
nweight = birth_weight[(birth_weight.bwght <= 4000) & (birth_weight.bwght >= 2500)]

# LARGE FOR GESTATIONAL AGE (LGA)
# A.K.A. "giant babies"
hiweight = birth_weight[birth_weight.bwght > 4000]
hiweight


#230 LGAs (12.55%)

# SMALL FOR GESTATIONAL AGE (SGA)
# A.K.A. "small rats"
lowweight = birth_weight[birth_weight.bwght < 2500]
# 92 SGAs (5.02%)



# Very low weight
vlow = birth_weight[birth_weight.bwght < 1500]

print('normal')
nweight.describe()

print('hi')
hiweight.describe()

print('low')
lowweight.describe()

print('very low')
vlow.describe()




##Class of mothers with more than 14 prenatal visits:
high_risk = birth_weight[birth_weight.npvis > 14]
high_risk.describe()
plt.hist(high_risk.bwght)
plt.show()
plt.boxplot(high_risk.bwght)
plt.show()

# Tracking mothers with high risk:
high_risk = birth_weight[birth_weight.npvis > 14]
high_risk.describe()


## Variable distributions:
for col in df.columns:
    x = df[col]
    plt.title("Variable: "+col)
    plt.hist(x)
    plt.show()

for col in df.columns:
    x = df[col]
    plt.title("Variable: "+col)
    plt.boxplot(x,vert=False)
    plt.show()
    
## Correlation between variables:


# adding jitter to better visualize data:
def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

for col in df.columns:
    x = df[col]
    y = df['bwght']
    #print("#### x VARIABLE:",col)
    #print("#### y VARIABLE: bwght")
    #sns.stripplot(x,y,jitter=True)
    plt.scatter(rand_jitter(x), y)
    plt.xlabel(col)
    plt.ylabel("bwght")
    plt.axhline(2500,color='blue')
    plt.axhline(4000,color='red')
    plt.show()

# Correlations:
df.corr()
df.corr()['bwght'].sort_values()

# Paiwise relationship:

for col1 in range(0,len(df.columns)):
    x = df.columns[col1]
    for col2 in range(0,len(df.columns)):
        y = df.columns[col2]
        if x != 'wclass':
            if y != 'wclass':
                sns.lmplot(x,y,data=df,hue='wclass',fit_reg=False)
                plt.savefig(col1+'_'+col2+'.png')
                plt.show()

    

#########################################
# MODEL ONE - LINEAR REGRESSION 

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 


# Parameters X and y:

X = df.drop(['omaps', 'fmaps','bwght','omaps', 'fmaps','m_meduc','m_npvis','m_feduc','group3'],axis=1)
y = df.bwght


#################################

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=508)

# Create the regressor: reg_all
reg_all = LinearRegression(normalize=False)

# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test , y_pred))
print("Root Mean Squared Error: {}".format(rmse))

#########################################
# MODEL 2 - KNN 

from sklearn.neighbors import KNeighborsRegressor

# Creating a regressor object
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 3)



# Checking the type of this new object
type(knn_reg)


# Teaching (fitting) the algorithm based on the training data
knn_reg.fit(X_train, y_train)

# Calling the score method, which compares the predicted values to the actual values

y_score = knn_reg.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)



###################################################
# APPROACH USING STATSMODELS - LINEAR REGRESSION

import statsmodels.formula.api as smf 

# Code to help building the variables to fit:
# for x in X.columns: print("df['"+x+"'] +") 

### OPTION 1 
# Building a Regression Base
lm_babyweight = smf.ols(formula = """bwght ~ df['mage'] +
df['meduc'] +
df['monpre'] +
df['npvis'] +
df['fage'] +
df['feduc'] +
df['cigs'] +
df['drink'] +
df['male'] +
df['mwhte'] +
df['mblck'] +
df['moth'] +
df['fwhte'] +
df['fblck'] +
df['foth'] +
df['out_drink'] +
df['lo_out_npvis'] +
df['hi_out_npvis'] +
df['out_mage'] +
df['out_fage'] +
df['out_feduc'] +
df['out_monpre'] """,
                        data = df)

##### OPTION 2
# Building a Regression Base
lm_babyweight = smf.ols(formula = """bwght ~ df['mage'] +
df['meduc'] +
df['monpre'] +
df['npvis'] +
df['fage'] +
df['feduc'] +
df['cigs'] +
df['drink'] +
df['male'] +
df['mwhte'] +
df['mblck'] +
df['moth'] +
df['fwhte'] +
df['fblck'] +
df['foth'] +
df['m_meduc'] +
df['m_npvis'] +
df['m_feduc'] +
df['drinker'] +
df['smoker'] +
df['trasher'] +
df['out_drink'] +
df['out_cigs'] +
df['lo_out_npvis'] +
df['hi_out_npvis'] +
df['out_mage'] +
df['big_out_mage'] +
df['out_fage'] +
df['mcollege'] +
df['fcollege'] +
df['out_feduc'] +
df['out_monpre'] +
df['visgap'] +
df['norm_mage'] +
df['std_mage'] +
df['log_mage'] +
df['sq_mage'] +
df['cub_mage'] +
df['norm_meduc'] +
df['std_meduc'] +
df['log_meduc'] +
df['sq_meduc'] +
df['cub_meduc'] +
df['norm_monpre'] +
df['std_monpre'] +
df['log_monpre'] +
df['sq_monpre'] +
df['cub_monpre'] +
df['norm_npvis'] +
df['std_npvis'] +
df['log_npvis'] +
df['sq_npvis'] +
df['cub_npvis'] +
df['norm_fage'] +
df['std_fage'] +
df['log_fage'] +
df['sq_fage'] +
df['cub_fage'] +
df['norm_feduc'] +
df['std_feduc'] +
df['log_feduc'] +
df['sq_feduc'] +
df['cub_feduc'] +
df['group1'] +
df['group2'] +
df['group3'] +
df['factor1'] +
df['factor2']  """,
                        data = df)



# Fitting Results
results = lm_babyweight.fit()


# Printing Summary Statistics
print(results.summary())

# monpre,drink,mblck,out_cigs,log_npvis,group3

# LINEAR REGRESSION - CLEANER MODEL V.2:
lm_babyweight2 = smf.ols(formula = """bwght ~ df['mage'] +
df['cigs'] +
df['drink'] +
df['mwhte'] +
df['mblck'] +
df['moth'] +
df['fwhte'] +
df['fblck'] +
df['foth']
 """,
                        data = df)

results2 = lm_babyweight2.fit()


# Printing Summary Statistics
print(results2.summary())

# R-squared reduced...




####################################################
# LASSO MODEL

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso # original lasso w.o. normalized data 0.005
lasso = Lasso(alpha=0.855,normalize=True,max_iter=100000)

# Fit the regressor to the data
lasso.fit(X_train, y_train)

# Calling the score method, which compares the predicted values to the actual values

y_score = lasso.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)

# Predict on the test data: y_pred
y_pred = lasso.predict(X_test)
# Compute and print R^2 and RMSE
print("R^2: {}".format(lasso.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test , y_pred))
print("Root Mean Squared Error: {}".format(rmse))


#####
# Compute and print the coefficients
lasso_coef = lasso.coef_
# Variables + coef table:
var_coef = pd.DataFrame({'var':X.columns,'coef':lasso_coef})


print(lasso_coef)



# Plot the coefficients:
# 
plt.plot(range(len(X.columns)), lasso_coef)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=60)
plt.margins(0.02)
plt.show()


######################################################
# RIDGE REGRESSION MODEL

# Model finetuning (Run code in Lasso model first)
drop_vars = []
for val in var_coef.loc[var_coef.coef ==0,'var'].values: drop_vars.append(val)

# Parameters X and y:

X = X.drop(drop_vars,axis=1)
y = df.bwght


#################################

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=508)


# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score



# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

ridge.alpha = 0.75

ridge.fit(X_train, y_train)

# Calling the score method, which compares the predicted values to the actual values

y_score = ridge.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)

# Predict on the test data: y_pred
y_pred = ridge.predict(X_test)
# Compute and print R^2 and RMSE
print("R^2: {}".format(ridge.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test , y_pred))
print("Root Mean Squared Error: {}".format(rmse))


###################
# ALPHA TESTING

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


display_plot(ridge_scores, ridge_scores_std)


##################################################
# POLYNOMIAL REGRESSION

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

poly = PolynomialFeatures(degree=4)
X_ = poly.fit_transform(X)
X_test_ = poly.fit_transform(X_test)
X_train_ = poly.fit_transform(X_train)

# Instantiate
lg = LinearRegression()

# Fit
lg.fit(X_train_, y_train)

# Calling the score method, which compares the predicted values to the actual values

y_score = lg.score(X_test_, y_test)

# The score is directly comparable to R-Square
print(y_score)

###################################################
# XGBOOST

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 1, n_estimators = 10)

xg_reg.fit(X_train,y_train)

# Calling the score method, which compares the predicted values to the actual values

predicted = xg_reg.predict(X_test)
residuals = np.array(y_test) - predicted

y_test_mean = np.mean(y_test)
# Calculate total sum of squares
tss =  sum((y_test - y_test_mean)**2 )
# Calculate residual sum of squares
rss =  (residuals**2).sum()
# Calculate R-squared
rsq  =  1 - (rss/tss)
cat('The R-square of the test data is ', round(rsq,3), '\n')





##########################################################
# SVR

from sklearn.svm import SVR

svr = SVR(kernel='linear',C=1.179)
svr.fit(X_train, y_train) 

# Calling the score method, which compares the predicted values to the actual values

y_score = svr.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)

# Predict on the test data: y_pred
y_pred = svr.predict(X_test)
# Compute and print R^2 and RMSE
print("R^2: {}".format(svr.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test , y_pred))
print("Root Mean Squared Error: {}".format(rmse))


##########################################
# SGD

from sklearn import linear_model

clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)

clf.fit(X_train, y_train) 

# Calling the score method, which compares the predicted values to the actual values

y_score = clf.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)




##################################
# comparing results to evaluate model

lasso_predictions = pd.DataFrame(lasso.predict(X))

lasso_predictions.columns = ['lasso_results']

results = pd.concat([df,lasso_predictions],axis=1)

results['residuals'] = results['bwght'] - results['lasso_results']


results.to_excel('lasso_results2.xls')








#########################################
# ELASTIC NET MODEL

from sklearn.linear_model import ElasticNet

regr = ElasticNet(random_state=0)

regr.alpha = 1.9

regr.l1_ratio = 0.65

regr.fit(X_train,y_train)

# Calling the score method, which compares the predicted values to the actual values

y_score = regr.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)



#########
# Theil sen model


from sklearn.linear_model import TheilSenRegressor # Theil Sen Regressor Model

# Instantiate
ts_reg = TheilSenRegressor(random_state = 508)

# Fit
ts_reg.fit(X_train, y_train)

# Predict
y_pred = ts_reg.predict(X_test)

# Score
y_score_ts = ts_reg.score(X_test, y_test)

print(y_score_ts)

#############
# Regression tree

from sklearn.tree import DecisionTreeRegressor # Regression trees

# Instantiate
tree_reg = DecisionTreeRegressor(criterion = 'mse',
                                 min_samples_leaf = 14,
                                 random_state = 508)

# Fit
tree_reg.fit(X_train, y_train)

# Predict
y_pred = tree_reg.predict(X_test)

# Score
y_score_tree = tree_reg.score(X_test, y_test)

print(y_score_tree)



########
# Bayesian ridge

from sklearn import linear_model

bayes_reg = linear_model.BayesianRidge()

bayes_reg.fit(X_train, y_train)

# Predict
y_pred = bayes_reg.predict(X_test)

# Score
y_score_bayes_reg = bayes_reg.score(X_test, y_test)

print(y_score_bayes_reg)



##################################################################
## KILLER REGRESSION

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 



baby_data = df.loc[:,['log_mage',
                      'out_mage',
                      'std_fage',
                      'log_feduc',
                      'cigs',
                      'out_cigs',
                      'smoker',
                      'drink',
                      'drinker',
                      'trasher',
                      'male',
                      'lo_out_npvis',
                      'hi_out_npvis',
                      'norm_npvis',                  
                      'std_monpre',
                      #'mage_medu'
                      ]]
baby_target = df.loc[:,'bwght']

X1_train, X1_test, y1_train, y1_test = train_test_split(baby_data,baby_target, 
                                                        test_size = 0.1, random_state=508)
reg_all2 = LinearRegression()
reg_all2.fit(X1_train,y1_train)

# Compute and print R^2 and RMSE
y1_pred_reg2 = reg_all2.predict(X1_test)
print('R-Squared: ',reg_all2.score(X1_test,y1_test).round(3))
rmse = np.sqrt(mean_squared_error(y1_test , y1_pred_reg2))
print("Root Mean Squared Error: {}".format(rmse))

# store predictions and residuals
y_pred_reg_all2 = reg_all2.predict(X1_test)
reg_all2_predictions = pd.DataFrame(y_pred_reg_all2)
reg_all2_predictions.columns = ['reg_all2_results']
results = pd.concat([df,reg_all2_predictions],axis=1)
results['residuals'] = results['bwght'] - results['reg_all2_results']
results.to_excel('reg_all2_results.xls')

########################################
# KNN V2

baby_data = df.loc[:,['mage',
                      'cigs',
                      'drink',
                      'fage',
                      'hi_out_npvis',
                      'log_npvis'
                      ]]
baby_target = df.loc[:,'bwght']

X1_train, X1_test, y1_train, y1_test = train_test_split(baby_data,baby_target, 
                                                        test_size = 0.1, random_state=508)

knn_reg2 = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 10)

# Teaching (fitting) the algorithm based on the training data
knn_reg2.fit(X1_train, y1_train)

# Calling the score method, which compares the predicted values to the actual values

y_score_knn2 = knn_reg2.score(X1_test, y1_test)

# The score is directly comparable to R-Square
print(y_score_knn2)



###############################################################
# Regression trees and feature importance:

# Importing new libraries
from sklearn.tree import DecisionTreeRegressor # Regression trees
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects

# graphviz


# Importing other libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # train/test split
#from sklearn.model_selection import cross_val_score # k-folds cross validation


###############################################################################
# Train/Test Splits
###############################################################################

"""
Prof. Chase:
    These first few steps are starting to feel like the movie Groundhog Day
    starring Bill Murray (1993), then you're in good shape.
    
    If you don't know about the movie Groundhog Day, I suggest you check it
    out: https://www.imdb.com/title/tt0107048/mediaviewer/rm4118549504
"""


# Preparing our previous model
baby_data = df.loc[:,['mage',
'meduc',
'monpre',
'npvis',
'fage',
'feduc',
'cigs',
'drink',
'male',
'mwhte',
'mblck',
'moth',
'fwhte',
'fblck',
'foth',
'm_meduc',
'm_npvis',
'm_feduc',
'drinker',
'smoker',
'trasher',
'out_drink',
'out_cigs',
'lo_out_npvis',
'hi_out_npvis',
'out_mage',
'big_out_mage',
'out_fage',
'mcollege',
'fcollege',
'out_feduc',
'out_monpre',
'visgap',
'norm_mage',
'std_mage',
'log_mage',
'sq_mage',
'cub_mage',
'norm_meduc',
'std_meduc',
'log_meduc',
'sq_meduc',
'cub_meduc',
'norm_monpre',
'std_monpre',
'log_monpre',
'sq_monpre',
'cub_monpre',
'norm_npvis',
'std_npvis',
'log_npvis',
'sq_npvis',
'cub_npvis',
'norm_fage',
'std_fage',
'log_fage',
'sq_fage',
'cub_fage',
'norm_feduc',
'std_feduc',
'log_feduc',
'sq_feduc',
'cub_feduc',
'mage_medu',
                      'fage_fedu',
'group1',
'group2',
'group3',
'factor1',
'factor2' ]]
baby_target = df.loc[:,'bwght']

X1_train, X1_test, y1_train, y1_test = train_test_split(baby_data,baby_target, 
                                                        test_size = 0.1, random_state=508)




###############################################################################
# Decision Trees
###############################################################################

"""
Prof. Chase:
    Notice how all of our modeling techniques in scikit-learn follow the same
    steps:
        * create a model object
        * fit data to the model object
        * predict on new data
        * score
    
    If you can master these four steps, any modeling technique in scikit-learn
    is at your disposal.
"""


# Let's start by building a full tree.
tree_full = DecisionTreeRegressor(random_state = 508)
tree_full.fit(X1_train, y1_train)

print('Training Score', tree_full.score(X1_train, y1_train).round(4))
print('Testing Score:', tree_full.score(X1_test, y1_test).round(4))



"""
Prof. Chase:
    Notice that the training score is extremely high. This is because the tree
    kept growing until the response variable was sorted as much as possible. In
    the real world we do not want a tree this big because:
        * it will predict poorly on new data, as observed above
        * it is too detailed to gain meaningful insights
    
    Displaying this tree would be a HUGE mistake. Let's manage the size of our
    tree through pruning.
"""



# Creating a tree with only two levels.
tree_2 = DecisionTreeRegressor(max_depth = 2,
                               min_samples_leaf = 30,
                               random_state = 508)

tree_2_fit = tree_2.fit(X1_train, y1_train)


print('Training Score', tree_2.score(X1_train, y1_train).round(4))
print('Testing Score:', tree_2.score(X1_test, y1_test).round(4))


# Our tree is much less predictive, but it can be graphed in a way that is
# easier to interpret.

dot_data = StringIO()


export_graphviz(decision_tree = tree_2_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = baby_data.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png(),
      height = 500,
      width = 800)

def plot_feature_importances(model, train = X1_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X1_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')

plot_feature_importances(tree_full,
                         train = X1_train,
                         export = True)

features = pd.DataFrame({'var':baby_data.columns, 'coef':tree_full.feature_importances_})

