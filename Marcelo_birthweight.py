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

birth_weight['visgap'] = birth_weight.npvis-birth_weight.monpre
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


# temp DF - drop NAs:
df = birth_weight.dropna()


## Variable distributions:
for col in df.columns:
    x = df[col]
    plt.title("Variable: "+col)
    plt.hist(x)
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
df.corr()['bwght'].sort_values()



#########################################
# MODEL BUILDING



