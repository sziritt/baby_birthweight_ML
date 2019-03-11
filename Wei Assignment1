# Loading Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # regression modeling

from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import sklearn.metrics # more metrics for model performance evaluation
from sklearn.model_selection import cross_val_score # k-folds cross validation

# Loading excel dataset

file = 'birthweight_feature_set.xlsx'

health = pd.read_excel(file)


# Missing data detection

print(
      health
      .isnull()
      .sum()
      )

# Impute missing data with median

for col in health:
    
    """ Impute missing values using the median of each column """
    
    if health[col].isnull().any():
        
        col_median = health[col].median()
        
        health[col] = health[col].fillna(col_median).round(3)

# Create secondary data for analytics purpose 
        
health['edu'] = health['feduc']+health['meduc']

health['sd'] = health['cigs']+health['drink']

health['diff'] = abs(health['mage'] - health['fage'])

health['young'] = health['mage'] <=25

    # health.loc[health['young']==0,'young']=0

    # health.loc[health['young']==1,'young']=26-health.loc[health['young']==1,'mage']

health['old'] = health['mage'] >=36

    # health.loc[health['old']==0,'old']=0

    # health.loc[health['old']==1,'old']=health.loc[health['old']==1,'mage']-35



health['overwt'] = health['bwght'] >=4000

health['lowwt'] = health['bwght'] >=2500



# Second time missing data detection before data analysis

print(
      health
      .isnull()
      .sum()
      )



########################
# Visual EDA (Histograms)
########################


plt.subplot(2, 2, 1)
sns.distplot(health['mage'],
             bins = 35,
             color = 'g')

plt.xlabel('mage')


########################


plt.subplot(2, 2, 2)
sns.distplot(health['meduc'],
             bins = 30,
             color = 'y')

plt.xlabel('meduc')



########################


plt.subplot(2, 2, 3)
sns.distplot(health['monpre'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('monpre')


########################


plt.subplot(2, 2, 4)

sns.distplot(health['npvis'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('npvis')


plt.tight_layout()
plt.savefig('Health Data Histograms 1 of 5.png')

plt.show()


#########################
 
plt.subplot(2, 2, 1)
sns.distplot(health['fage'],
             bins = 35,
             color = 'g')

plt.xlabel('fage')


########################


plt.subplot(2, 2, 2)
sns.distplot(health['feduc'],
             bins = 30,
             color = 'y')

plt.xlabel('feduc')



########################


plt.subplot(2, 2, 3)
sns.distplot(health['omaps'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('omaps')



########################


plt.subplot(2, 2, 4)

sns.distplot(health['fmaps'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('fmaps')



plt.tight_layout()
plt.savefig('Health Data Histograms 2 of 5.png')

plt.show()



#########################
 
plt.subplot(2, 2, 1)
sns.distplot(health['cigs'],
             bins = 35,
             color = 'g')

plt.xlabel('cigs')


########################


plt.subplot(2, 2, 2)
sns.distplot(health['drink'],
             bins = 30,
             color = 'y')

plt.xlabel('drink')



########################


plt.subplot(2, 2, 3)
sns.distplot(health['male'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('male')



########################


plt.subplot(2, 2, 4)

sns.distplot(health['mwhte'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('mwhte')



plt.tight_layout()
plt.savefig('Health Data Histograms 3 of 5.png')

plt.show()


#########################
 
plt.subplot(2, 2, 1)
sns.distplot(health['mblck'],
             bins = 35,
             color = 'g')

plt.xlabel('mblck')


########################


plt.subplot(2, 2, 2)
sns.distplot(health['moth'],
             bins = 30,
             color = 'y')

plt.xlabel('moth')



########################


plt.subplot(2, 2, 3)
sns.distplot(health['fwhte'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('fwhte')



########################


plt.subplot(2, 2, 4)

sns.distplot(health['fblck'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('fblck')



plt.tight_layout()
plt.savefig('Health Data Histograms 4 of 5.png')

plt.show()

#########################
 
plt.subplot(2, 2, 1)
sns.distplot(health['foth'],
             bins = 35,
             color = 'g')

plt.xlabel('foth')


########################


plt.subplot(2, 2, 2)
sns.distplot(health['bwght'],
             bins = 30,
             color = 'y')

plt.xlabel('bwght')



plt.tight_layout()
plt.savefig('Health Data Histograms 5 of 5.png')

plt.show()

#########################################

# Correlation check

health.head()


df_corr = health.corr().round(2)


print(df_corr)


df_corr.loc['bwght'].sort_values(ascending = False)



########################
# Correlation Heatmap
########################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))

df_corr2 = df_corr.iloc[1:19, 1:19]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig('Brith weight Correlation Heatmap.png')
plt.show()

########################
# Full Model
########################
lm_full = smf.ols(formula = """bwght ~     mage + meduc + monpre
                                                + npvis + fage
                                                + feduc + cigs
                                                + drink
                                           """,
                         data = health)


# Fitting Results
results = lm_full.fit()

# Printing Summary Statistics
print(results.summary())

print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")

#NEW CODE BEGIN

    
    
    
    
    
    
    
y='bwght'
xlist=['mage','meduc','monpre','monpre','npvis','fage','feduc','cigs','drink']

def formula_gen(lst):
    form = y+' ~ '
    for i in lst:
        form = form + i +' +'
    return form[:-1]

from itertools import combinations

formulalist=[]

for i in range(1,len(xlist)+1,1):
    for combo in combinations(xlist, i): 
        formulalist=formulalist+[list(combo)]

len(formulalist)


opt_form=''
opt_R_Squared=0
opt_y_score=0
opt_i=0

i=formulalist[400]

for i in formulalist:
    health_data=health[i]
    health_target = health.loc[:, y]

    X_train, X_test, y_train, y_test = train_test_split(
            health_data,
            health_target,
            test_size = 0.10,
            random_state = 508)        

    form=formula_gen(i)    
    lm_iterate = smf.ols(formula = form,
                         data = health)

    results = lm_iterate.fit()

    training_accuracy = []
    test_accuracy = []

    neighbors_settings = range(1, 51)

    for n_neighbors in neighbors_settings:
    # Building the model
        clf = KNeighborsRegressor(n_neighbors = n_neighbors)
        clf.fit(X_train, y_train)
    
    # Recording the training set accuracy
        training_accuracy.append(clf.score(X_train, y_train))
    
    # Recording the generalization accuracy
        test_accuracy.append(clf.score(X_test, y_test))

    k=max(1,test_accuracy.index(max(test_accuracy)))

    knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = k)

    # Fitting the model based on the training data
    knn_reg.fit(X_train, y_train)

    # Scoring the model
    y_score = knn_reg.score(X_test, y_test)
    
    if y_score>opt_y_score:
        opt_form=form
        opt_R_Squared=results.rsquared.round(3)
        opt_y_score=y_score
        opt_i=i

'''
We define optimal situation as the case when we get highest y-score
opt_form: optimal formula
opt_R_Squared: optimal R squared
opt_y_score: optimal y score
'''



###############################################################################
# Storing Model Predictions and Summary
###############################################################################

# We can store our predictions as a dictionary.

i=opt_i

health_data=health[i]
health_target = health.loc[:, y]

X_train, X_test, y_train, y_test = train_test_split(
            health_data,
            health_target,
            test_size = 0.10,
            random_state = 508)        

form=formula_gen(i)    
lm_iterate = smf.ols(formula = form,
                         data = health)

results = lm_iterate.fit()

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)

for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

k=max(1,test_accuracy.index(max(test_accuracy)))

knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = k)

    # Fitting the model based on the training data
knn_reg.fit(X_train, y_train)

    # Scoring the model
y_score = knn_reg.score(X_test, y_test)



knn_reg_fit = knn_reg.fit(X_train, y_train)



# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)



# The score is directly comparable to R-Square
print(y_score_knn_optimal)



# Generating Predictions based on the optimal KNN model


knn_reg_optimal_pred = knn_reg_fit.predict(X_test)

model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'KNN_Predicted': knn_reg_optimal_pred})



model_predictions_df.to_excel("Birthweight_Model_Predictions.xlsx")
