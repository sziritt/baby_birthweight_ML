# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:08:24 2019

@author: Stephy Zirit
"""

# Importing new libraries
from sklearn.tree import DecisionTreeRegressor # Regression trees
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
import pydotplus # Interprets dot objects

# Importing other libraries
import pandas as pd
from sklearn.model_selection import train_test_split # train/test split

file = 'birthweight_feature_set.xlsx'
birth_weight = pd.read_excel(file)

###############################################################################
# Decission Tree
###############################################################################

df_data = birth_weight.drop(['omaps','fmaps','bwght'],axis=1)

df_target = birth_weight.loc[:,'bwght']

X_train, X_test, y_train, y_test = train_test_split(
                                                    df_data,
                                                    df_target,
                                                    test_size = 0.1,
                                                    random_state = 508)

###### Full tree
tree_full = DecisionTreeRegressor(random_state = 508)
tree_full_fit = tree_full.fit(X_train, y_train)

print('Training Score', tree_full.score(X_train, y_train).round(4))
print('Testing Score:', tree_full.score(X_test, y_test).round(4))

##### Max 3 levels
tree_2 = DecisionTreeRegressor(max_depth = 3, random_state = 508)
tree_2_fit = tree_2.fit(X_train, y_train)

print('Training Score', tree_2.score(X_train, y_train).round(4))
print('Testing Score:', tree_2.score(X_test, y_test).round(4))

##### Tree graph
dot_data = StringIO()

export_graphviz(decision_tree = tree_2_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = df_data.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png(),
      height = 500,
      width = 800)
