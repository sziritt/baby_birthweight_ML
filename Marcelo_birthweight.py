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

""" 9 cols with missign values:
meduc 30
monpre 5
npvis 68
fage 6
feduc 47
omaps 3
fmaps 3
cigs 110
drink 115
"""
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

# temp DF - drop NAs (if any):
df = birth_weight.dropna()

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

# New variable 'visgap' = difference between # prenatal visits and prenatal month of start
birth_weight['visgap'] = birth_weight.npvis-birth_weight.monpre

# Normal weight (between 2500 and 4000)
nweight = birth_weight[(birth_weight.bwght <= 4000) & (birth_weight.bwght >= 2500)]

# LARGE FOR GESTATIONAL AGE (LGA)
# A.K.A. "giant babies"
hiweight = birth_weight[birth_weight.bwght > 4000]



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

#########################################
# NEW VARIABLES - FEATURE ENGINEERING:
#outliers:

# Creating binary variable 'out_drink'
birth_weight['out_drink'] = (birth_weight.drink > 12).astype('int')

# Creating binary variable 'lo_out_npvis'
birth_weight['lo_out_npvis'] = (birth_weight.npvis < 7).astype('int')

# Creating binary variable 'hi_out_npvis'
birth_weight['hi_out_npvis'] = (birth_weight.npvis > 15).astype('int')

# Creating binary variable 'out_mage'
birth_weight['out_mage'] = (birth_weight.mage > 60).astype('int')

# Creating binary variable 'out_fage'
birth_weight['out_fage'] = (birth_weight.fage > 55).astype('int')

# Creating binary variable 'out_educ'
birth_weight['out_feduc'] = (birth_weight.feduc < 7).astype('int')

# Creating binary variable 'out_monpre'
birth_weight['out_monpre'] = (birth_weight.monpre > 4).astype('int')

#########################################
# RESCALING VARIABLES:

# Function to normalize column data (rescale from 0 to 1) in df:
# Usage instructions - To normalize column 'mage' >>> normalize('mage')
normalize = lambda x : (df[x]-df[x].min())/df[x].max()



#########################################
# MODEL ONE - LINEAR REGRESSION 

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 

# temp DF - drop NAs (if any):
df = birth_weight.dropna()

# Parameters X and y:

X = df.drop(['omaps', 'fmaps','bwght','omaps', 'fmaps','m_meduc','m_npvis','m_feduc'],axis=1)
y = df.bwght

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

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