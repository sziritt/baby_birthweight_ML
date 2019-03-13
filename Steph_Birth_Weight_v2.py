# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:53:39 2019

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

""" 3 cols with missign values:
mage      0
meduc     3
xnpvis     3
fage      0
feduc     7
omaps     0
fmaps     0
cigs      0
drink     0
male      0
mwhte     0
mblck     0
moth      0
fwhte     0
fblck     0
foth      0
bwght     0
"""
# Flagging missing values
for col in birth_weight:
    if birth_weight[col].isnull().astype(int).sum() > 0:
        birth_weight['m_'+col] = birth_weight[col].isnull().astype(int)

# Filling NAs in 'npvis' , 'meduc' and 'feduc' with the MEDIAN
birth_weight.npvis = birth_weight.npvis.fillna(birth_weight.npvis.median())
birth_weight.meduc = birth_weight.meduc.fillna(birth_weight.meduc.median())
birth_weight.feduc = birth_weight.feduc.fillna(birth_weight.feduc.median())

# Rechecking NAs:
print(birth_weight.isnull().sum()) 
###############################################################################
##### AGE RANGE
###############################################################################
""" 
mage (20-29) & fage (20-64) - standard 1
mage (30-34) & fage (20-39) - standard 1
mage (30-34) & fage (40-64) - high risk 2
mage (35-44) & fage (35-39) - high risk 2
mage (35-44) & fage (40-64) - highest risk 3
else - highest risk 3
"""

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
    counter += 1

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

x = birth_weight['cage']
y = birth_weight['bwght']
plt.scatter(rand_jitter(x), y)
plt.xlabel('Combine ages')
plt.ylabel("bwght")
plt.axhline(2500,color='blue')
plt.axhline(4000,color='red')
plt.show()

###############################################################################
##### PRENATAL CARE AND VISITS
###############################################################################
"""
Weeks 4 to 28: 1 prenatal visit a month 
    month 1-6 - 6 visits
Weeks 28 to 36: 1 prenatal visit every 2 weeks 
    month 6-8 - 4 visits
Weeks 36 to 40: 1 prenatal visit every week
    month >8 - 4 visits
Starting in month 1 = 14 visits

month 1 - 14 - (9,15)
month 2 - 13 - (8,14)
month 3 - 12 - (7,13)
month 4 - 11 - (6,12)
month 5 - 10 - (5,11)
month 6 - 8 - (4,9)
month 7 - 6 - (3,7)
month 8 - 4 - (2,5)
"""   
counter = 0
birth_weight['regular'] = 0

for value in birth_weight['npvis']:
    if birth_weight.loc[counter,'monpre'] == 1:
        if (value >= 8 and value <= 15):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 2:
        if (value >= 7 and value <= 14):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 3:
        if (value >= 6 and value <= 13):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 4:
        if (value >= 5 and value <= 12):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 5:
        if (value >= 4 and value <= 11):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 6:
        if (value >= 3 and value <= 9):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 7:
        if (value >= 2 and value <= 7):
            birth_weight.loc[counter,'regular'] = 1
    elif birth_weight.loc[counter,'monpre'] == 8:
        if (value >= 1 and value <= 5):
            birth_weight.loc[counter,'regular'] = 1 
    counter += 1

x = birth_weight['regular']
y = birth_weight['bwght']
plt.scatter(rand_jitter(x), y)
plt.xlabel('Regular visits')
plt.ylabel("bwght")
plt.axhline(2500,color='blue')
plt.axhline(4000,color='red')
plt.show()

birth_weight.corr()['bwght'].sort_values()

###############################################################################
##### 
###############################################################################

###################################################
# APPROACH USING STATSMODELS - LINEAR REGRESSION

import statsmodels.formula.api as smf 

# Code to help building the variables to fit:
# for x in X.columns: print("df['"+x+"'] +") 

### OPTION 1 
# Building a Regression Base
lm_babyweight = smf.ols(formula = """bwght ~ birth_weight['mage'] +
birth_weight['meduc'] +
birth_weight['fage'] +
birth_weight['feduc'] +
birth_weight['monpre'] +
birth_weight['npvis'] +
birth_weight['cigs'] +
birth_weight['drink'] +
birth_weight['male'] +
birth_weight['mwhte'] +
birth_weight['mblck'] +
birth_weight['moth'] +
birth_weight['fwhte'] +
birth_weight['fblck'] +
birth_weight['foth'] +
birth_weight['cage'] +
birth_weight['regular'] """,
                        data = birth_weight)

# Fitting Results
results = lm_babyweight.fit()

# Printing Summary Statistics
print(results.summary())

### OPTION 2
# Building a Regression Base with signiicant variables
lm_babyweight2 = smf.ols(formula = """bwght ~ 
birth_weight['cigs'] +
birth_weight['drink'] +
birth_weight['mwhte'] +
birth_weight['mblck'] +
birth_weight['moth'] +
birth_weight['fwhte'] +
birth_weight['fblck'] +
birth_weight['foth'] +
birth_weight['cage'] """,
                        data = birth_weight)
# Fitting Results
results2 = lm_babyweight2.fit()

# Printing Summary Statistics
print(results2.summary())

### OPTION 3
# Building a Regression Base
lm_babyweight3 = smf.ols(formula = """bwght ~ birth_weight['mage'] +
birth_weight['cigs'] +
birth_weight['drink'] """,
                        data = birth_weight)
# Fitting Results
results3 = lm_babyweight3.fit()

# Printing Summary Statistics
print(results3.summary())






