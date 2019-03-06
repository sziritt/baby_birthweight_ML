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
##### LIBRARIES AND SET UP OF FILE 
###############################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file = 'birthweight.xlsx'
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

birth_quantiles = birth_weight.loc[:, :].quantile([0.20,
                                                    0.40,
                                                    0.60,
                                                    0.80,
                                                    1.00])

print(birth_quantiles)
# Explore variables
for col in birth_weight:
    print(col)

"""
Assumed Continuous/Interval Variables - 
mage
meduc
monpre
npvis
fage
feduc


bwght

Assumed Categorical -
omaps # Could be binary less than 7 require medical attention
fmaps # Could be binary less than 7 require medical attention
cigs
drink


Binary Classifiers -
male
mwhte
mblck
moth
fwhte
fblck
foth

"""
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

###############################################################################
##### PLOTS - EXPLORATORY ANALYSIS
###############################################################################
##### Boxplots - for numerical variables

for col in birth_weight.iloc[:, :18]: # variables
   birth_weight.boxplot(column = col, vert = False)
   plt.title(f"{col}")
   plt.tight_layout()
   plt.show()
    
###### Histograms with distribution plots
# Doesn't work because of missing values
for col in birth_weight.iloc[:, :18]:
    sns.distplot(birth_weight[col], bins = 'fd')
    plt.tight_layout()
    plt.show()

###############################################################################
##### OUTLIER ANALYSIS
###############################################################################

# Define function to flag outliers

def low_out(col,lim):
    birth_weight['o_'+col] = 0
    for val in enumerate(birth_weight.loc[ : , col]):   
        if val[1] <= lim:
            birth_weight.loc[val[0], 'o_'+col] = 1

def up_out(col,lim):
    birth_weight['o_'+col] = 0
    for val in enumerate(birth_weight.loc[ : , col]):   
        if val[1] >= lim:
            birth_weight.loc[val[0], 'o_'+col] = 1  

##### FLAGGIN LOWER OUTLIERS

#low_out('', 89)


##### FLAGGIN UPPER OUTLIERS

#up_out('', 9)


            
###############################################################################
##### HISTOGRAMS
###############################################################################
