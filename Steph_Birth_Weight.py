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


