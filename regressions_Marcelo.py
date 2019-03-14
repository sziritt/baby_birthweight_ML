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
# birth_weight['visgap'] = birth_weight.npvis-birth_weight.monpre

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

birth_weight['wclass'] = 'norm_weight'
birth_weight.loc[df.bwght < 2500,'wclass'] = 'lo_weight'
birth_weight.loc[df.bwght > 4000,'wclass'] = 'hi_weight'

########
# Testing new variables:


birth_weight['risk'] = 0
for value in enumerate(birth_weight['risk']):
    # Cigs
    if birth_weight.loc[value[0],'cigs'] > 17:
        birth_weight.loc[value[0],'risk'] += 2
    elif birth_weight.loc[value[0],'cigs'] < 4:
        birth_weight.loc[value[0],'risk'] += 0
    else:
        birth_weight.loc[value[0],'risk'] += 1
    #Drink    
    if birth_weight.loc[value[0],'drink'] > 10:
        birth_weight.loc[value[0],'risk'] += 10
    elif birth_weight.loc[value[0],'drink'] < 1:
        birth_weight.loc[value[0],'risk'] += 0
    else:
        birth_weight.loc[value[0],'risk'] += 1
        
     #fage   
    if birth_weight.loc[value[0],'fage'] > 48:
        birth_weight.loc[value[0],'risk'] += 2
    elif birth_weight.loc[value[0],'fage'] < 36:
        birth_weight.loc[value[0],'risk'] += 0
    else:
        birth_weight.loc[value[0],'risk'] += 1
                
     #mage   
    if birth_weight.loc[value[0],'mage'] > 58:
        birth_weight.loc[value[0],'risk'] += 2
    elif birth_weight.loc[value[0],'mage'] < 36:
        birth_weight.loc[value[0],'risk'] += 0
    else:
        birth_weight.loc[value[0],'risk'] += 1
df['risk'] = birth_weight['risk']
        
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
    
df['regular'] = birth_weight['regular']



birth_weight['magegroup'] = ''
# age brackets: up to 30, 30-40, 40-55, more 55
for value in enumerate(birth_weight['magegroup']):
    if birth_weight.loc[value[0],'mage'] <= 30:
        birth_weight.loc[value[0],'magegroup'] = 'mage<30'
    elif 30< birth_weight.loc[value[0],'mage'] <= 40:
        birth_weight.loc[value[0],'magegroup'] = 'mage30-40'
    elif 40< birth_weight.loc[value[0],'mage'] < 55:
        birth_weight.loc[value[0],'magegroup'] = 'mage40-55'
    else:
        birth_weight.loc[value[0],'magegroup'] = 'mage55plus'
    
df['magegroup'] = birth_weight['magegroup']

magegroup = pd.get_dummies(df['magegroup'],drop_first=False)
df = pd.concat([df,magegroup],axis=1)

df = df.drop('magegroup',axis=1)

birth_weight['fagegroup'] = ''
# age brackets: up to 30, 30-40, 40-55, more 55
for value in enumerate(birth_weight['fagegroup']):
    if birth_weight.loc[value[0],'fage'] <= 30:
        birth_weight.loc[value[0],'fagegroup'] = 'fage<30'
    elif 30< birth_weight.loc[value[0],'fage'] <= 40:
        birth_weight.loc[value[0],'fagegroup'] = 'fage30-40'
    elif 40< birth_weight.loc[value[0],'fage'] < 50:
        birth_weight.loc[value[0],'fagegroup'] = 'fage40-50'
    else:
        birth_weight.loc[value[0],'fagegroup'] = 'fage50plus'
    
df['fagegroup'] = birth_weight['fagegroup']

magegroup = pd.get_dummies(df['fagegroup'],drop_first=False)
df = pd.concat([df,magegroup],axis=1)

df = df.drop('fagegroup',axis=1)





##########################################
# Assembling Regression Model

###################################################
# APPROACH USING STATSMODELS - LINEAR REGRESSION

import statsmodels.formula.api as smf 

# Code to help building the variables to fit:
# for x in X.columns: print("df['"+x+"'] +") 

### Regression Version 1 
# Building a Regression Base
v1_babyweight = smf.ols(formula = """bwght ~ df['mage'] +
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
results_v1 = v1_babyweight.fit()


# Printing Summary Statistics
print(results_v1.summary())



##### Linear model V2
# Building a Regression Base
v2_babyweight = smf.ols(formula = """bwght ~ df['mage'] +
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
results_v2 = v2_babyweight.fit()


# Printing Summary Statistics
print(results_v2.summary())


# LINEAR REGRESSION - CLEANER MODEL V.3:
v3_babyweight = smf.ols(formula = """bwght ~ df['mage'] +
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

results_v3 = v3_babyweight.fit()


# Printing Summary Statistics
print(results_v3.summary())

# R-squared reduced...



###########
# LAST MODEL MAR-12nd
# SCORE = 0.743

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 

baby_data = df.loc[:,['mage',
                      'out_mage',
                      'fage',
                      'log_feduc',
                      'cigs',
                      'smoker',
                      'drink',
                      'drinker',
                      'lo_out_npvis',
                      'hi_out_npvis',
                      'cigs_mage', # effects cigs*age
                      'drink_mage',# effects drink*age
                      
                      
                      #'regular'
                      #'out_cigs',                                                   #'sq_npvis',                  
                      #'std_monpre',                                                 #'trasher',
                      #'male',
                      #'risk', # ????????????????


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

################ Model in statsmodels:

import statsmodels.formula.api as smf 

# Code to help building the variables to fit:
# for x in X.columns: print("df['"+x+"'] +") 

### V3 in statsmodels 
# Building a Regression Base
v3_babyweight = smf.ols(formula = """bwght ~ df['mage'] +
df['out_mage'] +
df['fage'] +
df['log_feduc'] +
df['cigs'] +
df['smoker'] +
df['drink'] +
df['drinker'] +
df['lo_out_npvis'] +
df['hi_out_npvis'] +
df['cigs_mage'] +
df['drink_mage'] """,
                        data = df)

# Fitting Results
results_v3 = v3_babyweight.fit()


# Printing Summary Statistics
print(results_v3.summary())



#######################################
# Storing model predictions

y_pred_reg_all2 = reg_all2.predict(X1_test)
reg_all2_predictions = pd.DataFrame(y_pred_reg_all2)
reg_all2_predictions.columns = ['reg_all2_results']
results = pd.concat([df,reg_all2_predictions],axis=1)
results['residuals'] = results['reg_all2_results'] - results['bwght']
results.to_excel('reg_all2_results.xls')


###########################
# TEST MODELS FUNCTION



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

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(bb_data,bb_target, 
                                                            test_size = 0.1, random_state=508)
    bb = LinearRegression()
    bb.fit(Xf_train,yf_train)

    # Compute and print R^2 and RMSE
    yf_pred_reg2 = bb.predict(Xf_test)
    print('R-Squared (train set): ',bb.score(Xf_train,yf_train).round(3))
    print('R-Squared  (test set): ',bb.score(Xf_test,yf_test).round(3))
    rmse = np.sqrt(mean_squared_error(yf_test , yf_pred_reg2))
    print("Root Mean Squared Error: {}".format(rmse))


# TESTING FUNCTION:
test_regression(['mage','cigs','drink','lo_out_npvis','hi_out_npvis','out_drink','drinker'])


##############################
#  TESTING KNN

from sklearn.neighbors import KNeighborsRegressor

def test_knn(variables,knn):
    bb_data = df.loc[:,variables]
    
    bb_target = df.loc[:,'bwght']

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(bb_data,bb_target, 
                                                            test_size = 0.1, random_state=508)
    for k in range(1,knn):
        print('number of neighbors :'+str(k))
        bb = KNeighborsRegressor(algorithm = 'auto',
                                  n_neighbors = k)
        bb.fit(Xf_train,yf_train)

        # Compute and print R^2 and RMSE
        yf_pred_reg2 = bb.predict(Xf_test)
        print('R-Squared (train set): ',bb.score(Xf_train,yf_train).round(3))
        print('R-Squared  (test set): ',bb.score(Xf_test,yf_test).round(3))
        rmse = np.sqrt(mean_squared_error(yf_test , yf_pred_reg2))
        print("Root Mean Squared Error: {}".format(rmse))




# KNN Model with Rˆ2 = 0.614 with k=7
test_knn(['mage','fage','cigs','drink','npvis','lo_out_npvis','out_drink'],15)