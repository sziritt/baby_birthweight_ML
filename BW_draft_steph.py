# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:36:03 2019

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
##### LIBRARIES AND FILE SET UP
###############################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt # check if using

file = 'birthweight_feature_set.xlsx'
birth_weight = pd.read_excel(file)

birth_weight = birth_weight.drop(['omaps','fmaps'], axis=1)

###############################################################################
##### DESCRIPTIVES STATISTICS
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
## Variable distributions:
for col in birth_weight.columns:
    x = birth_weight[col]
    plt.title("Variable: "+col)
    plt.hist(x)
    plt.show()

for col in birth_weight.columns:
    x = birth_weight[col]
    plt.title("Variable: "+col)
    plt.boxplot(x,vert=False)
    plt.show()

## Correlation between variables:
# adding jitter to better visualize data:
def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

for col in birth_weight.columns:
    x = birth_weight[col]
    y = birth_weight['bwght']
    plt.xlabel(col)
    plt.ylabel("bwght")
    plt.axhline(2500,color='blue')
    plt.axhline(4000,color='red')
    plt.show()

# Correlations:
birth_weight.corr()['bwght'].sort_values()
# Correlation matrix

# Classification: low, normal, high weight
df = pd.DataFrame.copy(birth_weight)

df['wclass'] = 'norm_weight'
df.loc[df.bwght < 2500,'wclass'] = 'lo_weight'
df.loc[df.bwght > 4000,'wclass'] = 'hi_weight'

weights = pd.get_dummies(df['wclass'],drop_first=False)
df = pd.concat([df,weights],axis=1)


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

################
## PIVOT TABLES

pivot_vals = ['mage',
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
              'bwght']

table_mean = pd.pivot_table(
        df, values=pivot_vals, 
        index='wclass',aggfunc=np.mean).round(2).iloc[[1,2,0],:]
table_mean

table_median = pd.pivot_table(
        df,values=pivot_vals, 
        index='wclass',aggfunc=np.median).round(2).iloc[[1,2,0],:]
table_median

###############################################################################
##### BUILDING NEW VARIABLES
###############################################################################
# Creating binary variable 'drinker'
birth_weight['drinker'] = (birth_weight.drink > 0).astype('int')

# Creating binary variable 'smoker'
birth_weight['smoker'] = (birth_weight.cigs > 0).astype('int')

birth_weight['trasher'] = birth_weight.drinker+birth_weight.smoker

birth_weight.loc[birth_weight.trasher == 2,'trasher'] = 4

# Binary outliers:
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

# New variable 'mage_medu' = ratio mother's age and education
birth_weight['mage_medu'] = birth_weight.mage/birth_weight.meduc

# New variable 'fage_fedu' = ratio father's age and education
birth_weight['fage_fedu'] = birth_weight.fage/birth_weight.feduc

# New variable 'cigs_mage' = force multiplier for drinking and age:
# This means that older women who drink too much will be more penalized
birth_weight['cigs_mage'] = birth_weight.cigs*birth_weight.mage

# New variable 'drink_mage' = force multiplier for drinking and age:
# This means that older women who drink too much will be more penalized
birth_weight['drink_mage'] = birth_weight.drink*birth_weight.mage

# New variable 'out_cigs_mage' = outlier of 'out_cigs_mage' (>590):
birth_weight['out_cigs_mage'] = (birth_weight.monpre > 590).astype('int')

# New variable 'out_drink_mage' = outlier of 'out_drink_mage' (>1250):
birth_weight['out_drink_mage'] = (birth_weight.monpre > 1250).astype('int')

# Combine ages of mather and father
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

""" Month prenatal care started - avg number of visits - regular range
month 1 - 14 - (8,15)
month 2 - 13 - (7,14)
month 3 - 12 - (6,13)
month 4 - 11 - (5,12)
month 5 - 10 - (4,11)
month 6 - 8 - (3,9)
month 7 - 6 - (2,7)
month 8 - 4 - (1,5)
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

# Getting the log value of father's education because of skewness 
birth_weight['log_feduc']= np.log(birth_weight['feduc'])

# Checking correlation with new variables
birth_weight.corr()['bwght'].sort_values().round(2)

###############################################################################
##### LINEAR REGRESSION USSING STATSMODELS
###############################################################################
# Libraries
import statsmodels.formula.api as smf 

df = pd.DataFrame.copy(birth_weight)
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

# Fitting Results
results = lm_babyweight.fit()

# Printing Summary Statistics
print(results.summary())

###############################################################################
##### 
###############################################################################

###############################################################################
##### 
###############################################################################








###############################################################################
##### TEST MODELS FUNCTION
###############################################################################
# Additional Libraries
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 

# Define function to test diferrent models
def test_regression(variables):
    values = str()
    for var in variables: 
        if var == variables[-1]:
            values = values + "df['"+var+"']"
        else:
            values = values + "df['"+var+"'] +"

    v4_baby = smf.ols(formula = "bwght ~ "+values,data=df)

    results_v4 = v4_baby.fit()

    print(results_v4.summary())
    
    print("\n####################################\n")
    
    print("## Testing variables in sklearn:\n")
    
    bb_data = df.loc[:,variables]
    
    bb_target = df.loc[:,'bwght']

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
                                                        bb_data,bb_target, 
                                                        test_size = 0.1, 
                                                        random_state=508)
    bb = LinearRegression()
    bb.fit(Xf_train, yf_train)

    # Compute and print R^2 and RMSE
    yf_pred_reg2 = bb.predict(Xf_test)
    print('R-Squared: ', bb.score(Xf_test,yf_test).round(3))
    rmse = np.sqrt(mean_squared_error(yf_test , yf_pred_reg2))
    print("Root Mean Squared Error: {}".format(rmse))

# Using testing function:
test_regression(['mage','cigs','drink','lo_out_npvis',
                 'hi_out_npvis','out_drink','drinker'])


