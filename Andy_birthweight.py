# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:22:01 2019

@author: Andy Chen
"""
###############################################################################
##### LIBRARIES AND SET UP OF FILE 
###############################################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

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
cigs
drink

Assumed Categorical - ordinal
omaps # Could be binary less than 7 require medical attention
fmaps # Could be binary less than 7 require medical attention

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

""" cols with missign values:
meduc 3
npvis 3
feduc 7

"""


# Filling NAs in 'npvis' , 'meduc' and 'feduc' with their MEDIANs
birth_weight.npvis = birth_weight.npvis.fillna(birth_weight.npvis.median())
birth_weight.meduc = birth_weight.meduc.fillna(birth_weight.meduc.median())
birth_weight.feduc = birth_weight.feduc.fillna(birth_weight.feduc.median())

# Rechecking NAs:
print(birth_weight.isnull().sum())

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

for col in birth_weight.iloc[:, :18]:
    sns.distplot(birth_weight[col], bins = 'fd')
    plt.title("Variable: "+col)
    plt.tight_layout()
    plt.show()


##### Jitter scatter plot and correlation

# adding jitter to better visualize data:
def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

for col in birth_weight.columns[0:18]:
    x = birth_weight[col]
    y = birth_weight['bwght']
    #print("#### x VARIABLE:",col)
    #print("#### y VARIABLE: bwght")
    #sns.stripplot(x,y,jitter=True)
    plt.scatter(rand_jitter(x), y) 
    plt.xlabel(col)
    plt.ylabel("bwght")
    plt.axhline(2500,color='blue')
    plt.axhline(4000,color='red')
    plt.show()

###############################################################################
##### CORRELATIONS
###############################################################################  
    
    
birth_weight.corr()

"""
mage & fage: 0.583608
meduc & feduc: 0.619529
"""


birth_weight.corr()['bwght'].sort_values(ascending = True)

""" top 4 correlated with bwght
drink    -0.743125
cigs     -0.572385
mage     -0.463811
fage     -0.395985
"""


monpre_corr = birth_weight.corr()['monpre'].sort_values(ascending = True)
print(monpre_corr[:3])

""" top 3 correlated with monpre
npvis   -0.342874
feduc   -0.286874
meduc   -0.232228
"""


npvis_corr = birth_weight.corr()['npvis'].sort_values()
print(npvis_corr[:3])

""" most correlated with npvis (other correlations are too low)
monpre   -0.342874
"""



meduc_corr = birth_weight.corr()['meduc'].sort_values()
print(meduc_corr[:5])

""" top 3 correlated with meduc
mwhte    -0.284530
fblck    -0.233591
monpre   -0.232228
"""


feduc_corr = birth_weight.corr()['feduc'].sort_values()
print(feduc_corr[:5])

""" top 3 correlated with feduc
mwhte    -0.414750
fwhte    -0.357027
monpre   -0.286874
"""

drink_corr = birth_weight.corr()['drink'].sort_values()
print(drink_corr[:5])

""" top 1 correlated with feduc
bwght   -0.743125
Other variables not highly correlated
"""

cigs_corr = birth_weight.corr()['cigs'].sort_values()
print(cigs_corr[:5])

""" top 1 correlated with feduc
bwght   -0.743125
Other variables not highly correlated
"""

# plot monpre and npvis for correlation 
plt.scatter('monpre', 'npvis', data = birth_weight)
plt.ylabel("Prenatal Visits")
plt.xlabel("Month Prenatal Care Starts")
plt.show()


###############################################################################
##### WEIGHT GROUPING
###############################################################################          
"""
nweight = birth_weight[(birth_weight.bwght <= 4000) & (birth_weight.bwght >= 2500)]
hiweight = birth_weight[birth_weight.bwght > 4000]
lowweight = birth_weight[birth_weight.bwght < 2500]
vlow = birth_weight[birth_weight.bwght < 1500]
"""

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

###############################################################################
##### PRENATAL CARE AND VISITS
###############################################################################
"""
Weeks 4 to 28: 1 prenatal visit a month 
    month 1-7 - 6 visits
Weeks 28 to 36: 1 prenatal visit every 2 weeks 
    month 7-9 - 4 visits
Weeks 36 to 40: 1 prenatal visit every week
    month 9-10 - 4 visits
Starting in month 1 = 14 visits
month 1 - 14 - npvis(9,15)
month 2 - 13 - npvis(8,14)
month 3 - 12 - npvis(7,13)
month 4 - 11 - npvis(6,12)
month 5 - 10 - npvis(5,11)
month 6 - 9  - npvis(4,9)
month 7 - 8  - npvis(3,7)
month 8 - 6  - npvis(2,5)
month 9 - 4
"""   

birth_weight.monpre.value_counts()
"""
month 2    93
month 1    45
month 3    33
month 4    13
month 5     4  # 2% outlier
month 8     3  # 1.5% outlier
month 7     3  # 1.5% outlier
month 6     2  # 1.0% outlier  # total = 6.1% outliers
"""

birth_weight.npvis.value_counts()
""" outliers value_counts
16.0     5  # 2.5% outliers
5.0      3  # 1.5% outlier
20.0     3
3.0      2  # 1.0% outlier
30.0     2
17.0     2
18.0     1  # 0.5% outliers
31.0     1
25.0     1
19.0     1
35.0     1
2.0      1   # 0.5% outliers
total    23  # 11.7% outliers
"""

birth_weight['npvis'].quantile([0.1, 0.9])
"""
0.1     7.5
0.9    15.0
"""

birth_weight['monpre'].quantile([0.1, 0.9])
"""
0.1    1.0
0.9    4.0
"""

# flagging npvis & monpre outliers
for val in range(0, 196):
    if birth_weight.loc[val, 'npvis'] > 15.0 or birth_weight.loc[val, 'npvis'] < 7.5:
        birth_weight.loc[val, 'out_npvis'] = 1
    else:
        birth_weight.loc[val, 'out_npvis'] = 0


for val in enumerate(birth_weight.loc[: , 'monpre']):
    if val[1] > 4.0:
        birth_weight.loc[val[0], 'out_monpre'] = 1
    elif val[1] < 1.0:
        birth_weight.loc[val[0], 'out_monpre'] = 1
    else:
        birth_weight.loc[val[0], 'out_monpre'] = 0

print(birth_weight.out_npvis.value_counts()) # 37 outliers
print(birth_weight.out_monpre.value_counts()) # 12 outliers




plt.scatter('out_monpre', 'out_npvis', data = birth_weight)
plt.title('Prenatal Care and Visits Outliers')
plt.xlabel('monpre outliers')
plt.ylable('npvis outliers')
plt.show()


plt.hist('bwght', data = birth_weight.loc[birth_weight['out_monpre']==1,:])
plt.title('Month of Prenatal Care Outliers')
plt.xlabel('birthweight')
plt.axvline(x = 2500, color = 'r')
plt.axvline(x = 4000, color = 'r')
plt.show()


plt.hist('bwght', data = birth_weight.loc[birth_weight['out_npvis']==1,:])
plt.title('Prenatal Visits Outliers')
plt.xlabel('birthweight')
plt.axvline(x = 2500, color = 'r')
plt.axvline(x = 4000, color = 'r')
plt.show()



###############################################################################
##### STATSMODEL 
##############################################################################
import statsmodels.formula.api as smf

# regression model 1
lm_1 = smf.ols(formula = """bwght ~ mage + fage + cigs + drink + monpre""", data = birth_weight)

results = lm_1.fit()
print(results.summary())
print('R_squared:',results.rsquared.round(3))
"""
R_squared: 0.709
"""

#  regression model 2
lm_2 = smf.ols(formula = """bwght ~ mage + fage + cigs + drink""", data = birth_weight)

results_2 = lm_2.fit()
print(results_2.summary())
print('R_squared:',results_2.rsquared.round(3))
"""
R_squared: 0.708
"""

#  regression model 3
lm_3 = smf.ols(formula = """bwght ~ mage  + cigs + drink + meduc""", 
               data = birth_weight)

results_3 = lm_3.fit()
print(results_3.summary())
print('R_squared:',results_3.rsquared.round(34))

"""
R_squared: 0.7097
"""

#  regression model 4
lm_4 = smf.ols(formula = """bwght ~ mage  + cigs + drink + meduc""", 
               data = birth_weight)

results_4 = lm_4.fit()
print(results_4.summary())
print('R_squared:',results_4.rsquared.round(4))

"""
R_squared: 0.7097
"""

###############################################################################
##### Linear regression
##############################################################################
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# linear model 1
X = birth_weight.drop(['out_npvis', 'npvis','mblck', 'feduc', 'fblck', 'omaps', 'fmaps','bwght', 'wclass'], axis = 1)
y = birth_weight.loc[:,'bwght'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=508)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred_linear = reg_all.predict(X_test)
reg_all.score(X_test, y_test)

""" Score =  0.5846 """


# linear model 2
X = birth_weight.drop(['npvis','feduc','omaps', 'fmaps','bwght', 'wclass'], axis = 1)
y = birth_weight.loc[:,'bwght'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=508)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred_linear = reg_all.predict(X_test)
reg_all.score(X_test, y_test)

""" Score = 0.5876"""
""" Subsetting high, low, vlow groups does not work in linear regression model """


# linear model 3
X = birth_weight.drop([ 'npvis','mblck', 'feduc','omaps', 'fmaps','bwght', 'wclass'], axis = 1)
y = birth_weight.loc[:,'bwght'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=508)


X_train_scale=scale(X_train)
X_test_scale=scale(X_test)

reg_all = LinearRegression()
reg_all.fit(X_train_scale, y_train)
y_pred_linear = reg_all.predict(X_test_scale)
reg_all.score(X_test_scale, y_test)

""" Score = 0.6026"""


# linear model 4
X = birth_weight.drop(['out_monpre', 'monpre', 'npvis','mblck', 'moth', 'mwhte', 'foth', 'fwhte', 'fblck','feduc','omaps', 'fmaps','bwght', 'wclass'], axis = 1)
y = birth_weight.loc[:,'bwght'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=508)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred_linear = reg_all.predict(X_test)
reg_all.score(X_test, y_test)


""" Score = 0.6445 """

###############################################################################
##### KNN Regression
##############################################################################
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics


# KNN model 1
X = birth_weight.drop(['out_monpre', 'out_npvis','omaps', 'fmaps','bwght', 'wclass'], axis = 1)
y = birth_weight.loc[:,'bwght'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=508)

neighbors = np.arange(1,20)
train_score = []
test_score = []

for k in neighbors:
    knn = KNeighborsRegressor(algorithm = 'auto', 
                              n_neighbors = k)
    knn.fit(X_train, y_train)
    train_score.append(knn.score(X_train, y_train))
    test_score.append(knn.score(X_test, y_test))
    
plt.figure(figsize=(12,6))
plt.plot(neighbors, train_score, color = 'blue', label='Train score')
plt.plot(neighbors, test_score, color = 'red', label='Test score')
plt.legend()
plt.title('Test Score vs Train Score by Number of Neighbors')
plt.ylabel('Score')
plt.xlabel('Number of Neighbors')
plt.tight_layout()
plt.show()

print('Optimal number of neightbors:', test_score.index(max(test_score))+1)


knn = KNeighborsRegressor(algorithm='auto', n_neighbors = 4)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(knn.score(X_test, y_test).round(4))
print('Squared Root Mean Error:', np.sqrt(metrics.mean_squared_error(y_pred_knn, y_test)).round(2))

""" Score = 0.4888 """



# KNN model 2
X = birth_weight.drop(['out_monpre', 'out_npvis','monpre', 'npvis', 'feduc', 'omaps', 'fmaps','bwght', 'wclass'], axis = 1)
y = birth_weight.loc[:,'bwght'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=508)

neighbors = np.arange(1,20)
train_score = []
test_score = []

for k in neighbors:
    knn = KNeighborsRegressor(algorithm = 'auto', 
                              n_neighbors = k)
    knn.fit(X_train, y_train)
    train_score.append(knn.score(X_train, y_train))
    test_score.append(knn.score(X_test, y_test))

print('Optimal number of neightbors:', test_score.index(max(test_score))+1)


knn = KNeighborsRegressor(algorithm='auto', n_neighbors = 8)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(knn.score(X_test, y_test).round(4))
print('Squared Root Mean Error:', np.sqrt(metrics.mean_squared_error(y_pred_knn, y_test)).round(2))

""" Score = 0.6861 """



# KNN model 3
X = birth_weight.drop(['out_monpre', 'out_npvis', 'omaps', 'fmaps','bwght', 'wclass'], axis = 1)
y = birth_weight.loc[:,'bwght'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=508)

# Standardizing the train and test data
X_train_scale=scale(X_train)
X_test_scale=scale(X_test)


neighbors = np.arange(1,40)
train_score = []
test_score = []

for k in neighbors:
    knn = KNeighborsRegressor(algorithm = 'auto', 
                              n_neighbors = k)
    knn.fit(X_train_scale, y_train)
    train_score.append(knn.score(X_train_scale, y_train))
    test_score.append(knn.score(X_test_scale, y_test))

print('Optimal number of neightbors:', test_score.index(max(test_score))+1)


knn = KNeighborsRegressor(algorithm='auto', n_neighbors = 5)
knn.fit(X_train_scale, y_train)
y_pred_knn = knn.predict(X_test_scale)
print(knn.score(X_test_scale, y_test))
print('Squared Root Mean Error:', np.sqrt(metrics.mean_squared_error(y_pred_knn, y_test)).round(2))

""" score = 0.6732 """ 



###############################################################################
##### KNN Classification
##############################################################################
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


for val in range(0, 196):
    if birth_weight.loc[val, 'bwght'] > 4000 or birth_weight.loc[val, 'bwght'] < 2500:
        birth_weight.loc[val, 'wclass'] = 1
    else:
        birth_weight.loc[val, 'wclass'] = 0

X = birth_weight.drop(['out_monpre', 'out_npvis','omaps', 'fmaps','bwght','wclass'], axis = 1)
y = birth_weight.loc[:,'wclass'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=508)



error = []

# K values range between 1 and 40
for x in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = x)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error.append(np.mean(y_pred != y_test)) #Why?
    
plt.figure(figsize=(12,6))    
plt.plot(range(1,40), error, color='blue', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate for K')
plt.ylabel('Mean Error')
plt.xlabel('K Values')
plt.show()

print('Optimal number of neightbors:', error.index(min(error))+1)




knn_classifier = KNeighborsClassifier(n_neighbors = 2)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
print(accuracy_score(y_test, y_pred_knn).round(4))
print(classification_report(y_test, y_pred_knn))


""" Score =  0.9 """
""" However, the classication model failed to predict any obnormal birthweight """


# KNN classifier model 2
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression

X_train_scale=scale(X_train)
X_test_scale=scale(X_test)

# Fitting logistic regression on our standardized data set
log=LogisticRegression(penalty='l2',C=.01)
log.fit(X_train_scale,y_train)
y_pred_log = log.predict(X_test_scale)
accuracy_score(y_test, y_pred_log)

""" Score =  0.9 """


