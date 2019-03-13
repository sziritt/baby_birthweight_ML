# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:03:53 2019

@author: Stephy Zirit
"""

from sklearn.model_selection import train_test_split # train/test split
import statsmodels.formula.api as smf # regression modeling

###############################################################################
# Different methods with three varables more significant
###############################################################################

df_data = birth_weight.drop(['omaps','fmaps','bwght'], axis = 1)

df_target = birth_weight.loc[:,'bwght']
# train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_data,
                                                    df_target,
                                                    test_size = 0.1,
                                                    random_state = 508)

df_train = pd.concat([X_train, y_train], axis = 1)

ols_1 = smf.ols(formula = """bwght ~ mage +
                                     cigs +
                                     drink """,
                                     data = df_train)

results_ols_1 = ols_1.fit()

print(results_ols_1.summary())

# Building a dataset based on interesting variables
df_data = birth_weight.loc[:,['mage','cigs','drink']]

df_target = birth_weight.loc[: ,'bwght']

X_train, X_test, y_train, y_test = train_test_split(df_data,
                                                    df_target,
                                                    test_size = 0.1,
                                                    random_state = 508)
##############################################################################
from sklearn.linear_model import LinearRegression

# Instantiate
lr = LinearRegression()

# Fit
lr_fit = lr.fit(X_train, y_train)

# Predict
lr_pred = lr_fit.predict(X_test)

# Score
y_score_ols = lr_fit.score(X_test, y_test)

print(y_score_ols)

##############################################################################
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression

# Instantiate
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 2)

# Fit
knn_reg.fit(X_train, y_train)

# Predict
y_pred = knn_reg.predict(X_test)

# Score
y_score_knn = knn_reg.score(X_test, y_test)

print(y_score_knn)

##############################################################################
from sklearn.tree import DecisionTreeRegressor # Regression trees

# Instantiate
tree_reg = DecisionTreeRegressor(criterion = 'mse',
                                 min_samples_leaf = 15,
                                 random_state = 508)

# Fit
tree_reg.fit(X_train, y_train)

# Predict
y_pred = tree_reg.predict(X_test)

# Score
y_score_tree = tree_reg.score(X_test, y_test)

print(y_score_tree)

##############################################################################
# Compare scores
print(f"""
OLS score: {y_score_ols.round(3)}
KNN score: {y_score_knn.round(3)}
Tree score: {y_score_tree.round(3)}
""")