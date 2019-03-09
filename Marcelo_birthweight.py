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

###############################################################################
##### EXPLORATORY ANALYSIS
###############################################################################

# class of baby weight (REMOVE FROM MODEL TRAIN-TEST!):
# low, normal, high weight
birth_weight['wclass'] = 'normal'
birth_weight.loc[birth_weight.bwght < 2500,'wclass'] = 'low'
birth_weight.loc[birth_weight.bwght > 4000,'wclass'] = 'hi'

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
                sns.lmplot(x,y,data=df,hue='wclass')
                plt.show()

    
#########################################
# NEW VARIABLES - FEATURE ENGINEERING:

# Creating binary variable 'drinker'
birth_weight['drinker'] = (birth_weight.drink > 0).astype('int')

# Creating binary variable 'smoker'
birth_weight['smoker'] = (birth_weight.cigs > 0).astype('int')

birth_weight['trasher'] = birth_weight.drinker+birth_weight.smoker

birth_weight.loc[birth_weight.trasher == 2,'trasher'] = 4

#outliers:

# INCLUDE MOTHER´S AGE > 67

# Creating binary variable 'out_drink' # original=12
birth_weight['out_drink'] = (birth_weight.drink > 12).astype('int')

# Creating binary variable 'lo_out_npvis' # original=7
birth_weight['lo_out_npvis'] = (birth_weight.npvis < 7).astype('int')

# Creating binary variable 'hi_out_npvis' # original=15
birth_weight['hi_out_npvis'] = (birth_weight.npvis > 15).astype('int')

# Creating binary variable 'out_mage' # original=60
birth_weight['out_mage'] = (birth_weight.mage > 60).astype('int')

# Creating binary variable 'out_fage' # original=55
birth_weight['out_fage'] = (birth_weight.fage > 55).astype('int')

# Creating binary variable 'out_feduc' # original=7
birth_weight['out_feduc'] = (birth_weight.feduc < 7).astype('int')

# Creating binary variable 'out_monpre' # original=4
birth_weight['out_monpre'] = (birth_weight.monpre > 4).astype('int')

# New variable 'visgap' = diff between # prenatal visits & prenatal month of start
birth_weight['visgap'] = birth_weight.npvis-birth_weight.monpre

df = birth_weight

#####################
# Adding normalized columns for ['mage','meduc','monpre','npvis','fage','feduc','cigs','drink']

normalize = lambda var : (birth_weight[var]-min(birth_weight[var]))/(max(birth_weight[var])-min(birth_weight[var]))

standardize = lambda var : (birth_weight[var]-birth_weight[var].mean())/birth_weight[var].std()


cont_vars = ['mage','meduc','monpre','npvis','fage','feduc'] #,'cigs','drink']

for col in cont_vars:
    #birth_weight['norm_'+col] = normalize(col)
    #birth_weight['std_'+col] = standardize(col)
    birth_weight['log_'+col] = np.log(birth_weight[col])
    

df = birth_weight
#########################################

#########
## K-means clusters

from sklearn.cluster import KMeans

data_for_cluster = birth_weight.drop(['omaps','fmaps','bwght'],axis=1)

kmeans = KMeans(n_clusters=6, random_state=0).fit(data_for_cluster)

# Check clusters
# kmeans.labels_

# assign new column:
clusters = pd.get_dummies(kmeans.labels_,drop_first=True)
clusters.columns = ['group1','group2','group3','group4','group5']
df = pd.concat([df,clusters],axis=1)


#########################################
# MODEL ONE - LINEAR REGRESSION 

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 


# Parameters X and y:

X = df.drop(['omaps', 'fmaps','bwght','omaps', 'fmaps','m_meduc','m_npvis','m_feduc'],axis=1)
y = df.bwght

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=508)

# Create the regressor: reg_all
reg_all = LinearRegression()

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
                              n_neighbors = 20)



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



# Fitting Results
results = lm_babyweight.fit()


# Printing Summary Statistics
print(results.summary())



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
lasso.fit(X, y)

# Calling the score method, which compares the predicted values to the actual values

y_score = lasso.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)

#####
# Compute and print the coefficients
lasso_coef = lasso.coef_

print(lasso_coef)

# Plot the coefficients:
# 
plt.plot(range(len(X.columns)), lasso_coef)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=60)
plt.margins(0.02)
plt.show()


######################################################
# RIDGE REGRESSION MODEL

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

ridge.alpha = 5

ridge.fit(X, y)

# Calling the score method, which compares the predicted values to the actual values

y_score = ridge.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)

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

poly = PolynomialFeatures(degree=1)
X_ = poly.fit_transform(X)
X_test_ = poly.fit_transform(X_test)

# Instantiate
lg = LinearRegression()

# Fit
lg.fit(X_, y)

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

clf = SVR(kernel='linear',C=1.179)
clf.fit(X_train, y_train) 

# Calling the score method, which compares the predicted values to the actual values

y_score = clf.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)


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

results.to_excel('lasso_results.xls')








#########################################
# ELASTIC NET MODEL

from sklearn.linear_model import ElasticNet

regr = ElasticNet(random_state=0)

regr.alpha = 1.25

regr.l1_ratio = 0.65

regr.fit(X_train,y_train)

# Calling the score method, which compares the predicted values to the actual values

y_score = regr.score(X_test, y_test)

# The score is directly comparable to R-Square
print(y_score)
