# Partial Dependence PLots
# while feature importance shows what vriables most affect predictions, partial dependence plots show how a feature aggects predictions.

# How it works

# Like permutation importance, **partial dependence plots are calculated after a
# model has been fit.** The model is fit on real data that has not been
# artificially manipulated in any way.
# 
# In our soccer example, teams may differ in many ways. How many passes they made,
# shots they took, goals they scored, etc. At first glance, it seems difficult to
# disentangle the effect of these features.
# 
# To see how partial plots separate out the effect of each feature, we start by
# considering a single row of data. For example, that row of data might represent
# a team that had the ball 50% of the time, made 100 passes, took 10 shots and
# scored 1 goal.
# 
# We will use the fitted model to predict our outcome (probability their player
# won "man of the match"). But we **repeatedly alter the value for one variable**
# to make a series of predictions. We could predict the outcome if the team had
# the ball only 40% of the time. We then predict with them having the ball 50% of
# the time. Then predict again for 60%. And so on. We trace out predicted outcomes
# (on the vertical axis) as we move from small values of ball possession to large
# values (on the horizontal axis).
# 
# In this description, we used only a single row of data. Interactions between
# features may cause the plot for a single row to be atypical. So, we repeat that
# mental experiment with multiple rows from the original dataset, and we plot the
# average predicted outcome on the vertical axis.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from pdpbox import pdp, get_dataset, info_plots
import matplotlib

print(f"Matplotlib Version: {matplotlib.__version__}")
print(f"Matplotlib Backend: {matplotlib.get_backend()}")

from matplotlib import pyplot as plt
import pydotplus
from sklearn.externals.six import StringIO

data = pd.read_csv("~/github/DataAnalysis/data/FIFA_2018_Statistics.csv")
data.head()

# convert from string "Yes/No" to binary
y = (data['Man of the Match'] == 'Yes')

feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
print(f"[-] feature name:\n[^--] {feature_names}")

x = data[feature_names]
print(f"x shape: {x.shape}")

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)
# Decision Tree
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5,
                                    min_samples_split=5).fit(train_x, train_y)
# visualise decision tree
dot_data = StringIO()
tree_graph = tree.export_graphviz(tree_model, out_file=dot_data,
                                  feature_names=feature_names,
                                  filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')

# Creating Partial Dependence PLot using PDPBox Library
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_x, model_features=feature_names,
                            feature='Goal Scored')
pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()

# Random Forest
rf_model = RandomForestClassifier(random_state=0).fit(train_x, train_y)
rf_pdp_goals = pdp.pdp_isolate(model=rf_model, dataset=val_x, model_features=feature_names,
                            feature='Goal Scored')
pdp.pdp_plot(rf_pdp_goals, 'Goal Scored')
plt.show()

# New York Taxi Data--------------------------------------------------------------
ny_data = pd.read_csv("~/github/DataAnalysis/data/train.csv")
ny_data.head()
ny_data.columns

# remove data with extremes
ny_data = ny_data.query(
    'pickup_latitude > 40.68 and pickup_latitude < 40.82 and ' +
    'dropoff_latitude > 40.68 and dropoff_latitude < 40.82 and ' +
    'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +
    'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +
    'fare_amount > 0')
y = ny_data.fare_amount
ny_data.info()

base_features = [i for i in ny_data.columns if ny_data[i].dtype in [np.float64, np.int64]]
print(f"base features: {base_features}")
ny_data_model = ny_data[base_features]
ny_data_model.info()
x = ny_data_model.iloc[:,1:]
x.info()
y.shape
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

# Random Forest Regressor
rfg_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(train_x, train_y)
rfg_model
print(rfg_model.feature_importances_)

rfg_pdp_dolat = pdp.pdp_isolate(model=rfg_model, dataset=val_x, model_features=val_x.columns,
                            feature='dropoff_latitude')
pdp.pdp_plot(rfg_pdp_dolat, 'dropoff_latitude')
plt.show()
