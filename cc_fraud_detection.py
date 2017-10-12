#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:28:01 2017

@author: muralikorrapati
"""

"""
Credit Card fraud detection
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


sns.set(style="darkgrid")

#X and y are assumed to be a Pandas DataFrame and Series respectively.
def resample(X, y, sample_type=None, sample_size=None, class_weights=None, seed=None):

    # Nothing to do if sample_type is 'abs' or not set.  sample_size should then be int
    # If sample type is 'min' or 'max' then sample_size should be float
    if sample_type == 'min':
        sample_size_ = np.round(sample_size * y.value_counts().min()).astype(int)
    elif sample_type == 'max':
        sample_size_ = np.round(sample_size * y.value_counts().max()).astype(int)
    else:
        sample_size_ = max(int(sample_size), 1)

    if seed is not None:
        np.random.seed(seed)

    if class_weights is None:
        class_weights = dict()

    X_resampled = pd.DataFrame()

    for yi in y.unique():
        size = np.round(sample_size_ * class_weights.get(yi, 1.)).astype(int)

        X_yi = X[y == yi]
        sample_index = np.random.choice(X_yi.index, size=size)
        X_resampled = X_resampled.append(X_yi.reindex(sample_index))

    return X_resampled

creditcards = pd.read_csv('creditcard.csv')

list(creditcards.columns)

creditcards.shape

for _ in creditcards.columns:
    print("The number of null values in {} = {}".format(_, creditcards[_].isnull().sum()))

#The response variable Class tell us whether a transaction was fraudulent 
#(value = 1) or not (value = 0).
ncount = len(creditcards)
ax = sns.countplot(x="Class", data=creditcards)
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.2f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text

creditcards.Time.describe()

# how many seconds are 24 hours
# 1 hr = 60 mins = 60 x 60 s = 3600 s
3600 * 24

# separate transactions by day
creditcards['day'] = creditcards.apply(lambda row: "day2" if row['Time'] > (3600*24) else "day1", axis=1)

creditcards.day.describe()

# make transaction relative to day
creditcards['Time_day'] = creditcards.apply(lambda row: row['Time'] - 86400 if row['day'] == "day2"  else row['Time'], axis=1)
creditcards.Time_day.describe()

creditcards.loc[creditcards['day']=="day1"]['Time_day'].describe()
creditcards.loc[creditcards['day']=="day2"]['Time_day'].describe()

creditcards['Time_New'] = creditcards.apply(lambda row: "gr1" if row['Time_day'] <= 38138 else \
                                                                   "gr2" if row['Time_day'] <= 52327 else \
                                                                               "gr3" if row['Time_day'] <= 69580 else "gr4" \
                                                                                           , axis=1)

#Make sure new column is created correctly
creditcards.Time_New.unique()

#Update Time column with Time_New column and remove all new columns
creditcards['Time'] = creditcards['Time_New'].astype('category')

ax = sns.countplot(x="day", data=creditcards)
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.2f}%'.format(100.*y/ncount), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text
    
#Remove all newly added columns
creditcards.drop(['day', 'Time_day', 'Time_New'], axis=1, inplace=True)
creditcards.shape

# convert class variable to factor
creditcards['Class'] = creditcards['Class'].astype('category')

ax = sns.countplot(x="Time", data=creditcards)

#Compare amounts between fraud and non-fraud transactions
creditcards.loc[creditcards['Class']==0]['Amount'].describe()
creditcards.loc[creditcards['Class']==1]['Amount'].describe()

#Graph by Amount
creditcards['Amount'].hist(by=creditcards['Class'], bins=50)

response = "Class"
s = set([response])
features = [x for x in creditcards.columns if x not in s]


##Modeling
#Try isolationForest from sklearn
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X = creditcards[features]
X = X.iloc[:, :].values
y=creditcards['Class']

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#Split between train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rng = np.random.RandomState(42)

# fit the model
clf = IsolationForest(n_estimators=500, max_samples=1000, random_state=rng)
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = pd.DataFrame({'predict':clf.predict(X_test)})
y_pred_test['predict'] = y_pred_test.apply(lambda row: 0 if row['predict'] == -1.0 else 1, axis=1)

clf.decision_function(X_train)
confusion_matrix(y_test, y_pred_test['predict'])


#initialize h2o
import h2o
h2o.init()

# convert data to H2OFrame
creditcards_hf = h2o.H2OFrame(creditcards)

creditcards_hf

#Split data into unsupervised, supervised, and test datasets
train_unsupervised,train_supervised,test = creditcards_hf.split_frame(ratios=[.4, .4])
train_supervised['Class'] = train_supervised['Class'].asfactor()
train_unsupervised['Class'] = train_unsupervised['Class'].asfactor()
test['Class'] = test['Class'].asfactor()

#Autoencoders
#With h2o, we can simply set autoencoder = TRUE
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator

nfeatures = 2  # number of features (smallest hidden layer)

model_nn = H2OAutoEncoderEstimator(model_id = "model_nn",
                             reproducible = True, #slow - turn off for real problems
                             ignore_const_cols = False,
                             seed = 123,
                             hidden = [10, 2, 10], 
                             epochs = 100,
                             activation = "Tanh")

#Train the model
model_nn.train(list(range(len(features))), training_frame=train_unsupervised[features])

#Save model to disk so we can load t=it again
h2o.save_model(model_nn, path= os.getcwd() + "/model_nn", force=True)

#Load model from disk
model_nn = h2o.load_model(os.getcwd() + "/model_nn//model_nn")

#Print encoder model
model_nn

#Dimensionality reduction with hidden layers(from layer 2)
train_unsupervised_features = model_nn.deepfeatures(train_unsupervised[0:len(features)], 1)
train_unsupervised_features = train_unsupervised_features.cbind(train_unsupervised[response])
train_unsupervised_features['Class'] = train_unsupervised_features['Class'].asfactor()

#Plot train_features
axes = sns.pairplot(x_vars=["DF.L2.C1"], y_vars=["DF.L2.C2"], data=train_unsupervised_features.as_data_frame(), hue="Class", size=5)


# convert train_supervised with autoencoder to lower-dimensional space(from layer 3)
train_supervised_features = model_nn.deepfeatures(train_supervised[0:len(features)], 2)

#Add Class column
train_supervised_features = train_supervised_features.cbind(train_supervised[response])
train_supervised_features['Class'] = train_supervised_features['Class'].asfactor()

#Get feature column names
features_dim = [x for x in train_supervised_features.columns if x not in s]

# Test the DRF model on the test set (processed through deep features)
#Dimensionality reduction on test data
test_features = model_nn.deepfeatures(test[0:len(features_dim)], 2)
test_features = test_features.cbind(test[response])
test_features['Class'] = test_features['Class'].asfactor()


#Train Isolation Forest
#convert data to pd.DataFrame
train_supervised_featured_pd = train_supervised_features.as_data_frame(use_pandas=True)
test_supervised_featured_pd = train_supervised_features.as_data_frame(use_pandas=True)

X_train = train_supervised_featured_pd[features_dim]
y_train = train_supervised_featured_pd['Class']
X_test = test_supervised_featured_pd[features_dim]
y_test = test_supervised_featured_pd['Class']

clf = IsolationForest(n_estimators=500, max_samples=1000, random_state=rng)
clf.fit(X_train, y_train)
y_pred_train = clf.predict(X_train)
y_pred_test = pd.DataFrame({'predict':clf.predict(X_test)})
y_pred_test['predict'] = y_pred_test.apply(lambda row: 0 if row['predict'] == -1.0 else 1, axis=1)

clf.decision_function(X_train)
confusion_matrix(y_test, y_pred_test['predict'])


# Train DRF on extracted feature space
drf_model = H2ORandomForestEstimator(ntrees=50, max_depth=20, seed=1234)
drf_model.train(x=features_dim, y=response, training_frame=train_supervised_features)

# Confusion Matrix and assertion
cm = drf_model.confusion_matrix()
cm.show()

#Excludes the "Cover_Type" column from the features provided
rf_predictions = drf_model.predict(test_features[features_dim])

#validation set accuracy
drf_model.model_performance(test_data = test_features)

"""
drf_model.score_history()
drf_model.model_performance(test_data = test_features)
drf_model.show()
help(H2ORandomForestEstimator)
"""
# Train Deeplearning model on extracted feature space
deep_model = H2ODeepLearningEstimator(model_id = "model_deep",
                             reproducible = True, #slow - turn off for real problems
                             ignore_const_cols = False,
                             hidden = [10, 2, 10], 
                             epochs = 1000,
                             activation = "Tanh")

deep_model.train(x=features_dim, y=response, training_frame=train_supervised_features)

# Confusion Matrix and assertion
cm = deep_model.confusion_matrix()
cm.show()

#Excludes the "Class" column from the features provided
deep_predictions = deep_model.predict(test_features[features_dim])

deep_model.model_performance(test_data = test_features)

#validation set accuracy
deep_predictions = deep_predictions.cbind(test_features[response].asfactor()).as_data_frame(use_pandas=True)
(deep_predictions['Class'] == deep_predictions['predict']).mean()

from sklearn.metrics import confusion_matrix
confusion_matrix(deep_predictions['Class'], deep_predictions['predict'])

#Anomaly detection
#anomaly = model_nn.anomaly(test, per_feature=False)
#anomaly = anomaly.cbind(test_features[response].asfactor()).as_data_frame(use_pandas=True)
#anomaly['id'] = anomaly.index

anomaly = model_nn.anomaly(train_supervised[:,features], per_feature=False)
anomaly = anomaly.cbind(train_supervised[response].asfactor()).as_data_frame(use_pandas=True)
anomaly['id'] = anomaly.index
       
mean_mse = anomaly.groupby('Class')['Reconstruction.MSE'].mean()
min_mse = anomaly.groupby('Class')['Reconstruction.MSE'].min()
max_mse = anomaly.groupby('Class')['Reconstruction.MSE'].max()

axes = sns.pairplot(x_vars=["id"], y_vars=["Reconstruction.MSE"], data=anomaly, hue="Class", size=5, kind="scatter")
x = plt.gca().axes.get_xlim()
plt.plot(x, len(x) * [mean_mse[0]], sns.xkcd_rgb["denim blue"])
plt.plot(x, len(x) * [mean_mse[1]], sns.xkcd_rgb["green"])
plt.show()

anomaly['Reconstruction.MSE'].hist(by=anomaly['Class'], bins=50)

#Mark which ones are outliers
anomaly['outlier'] = anomaly.apply(lambda row: "outlier" if (row['Reconstruction.MSE'] >= 0.006)  else "no-outlier", axis=1)

anomaly.groupby(['Class', 'outlier']).size()

#Apply RF based on anomaly
# Picking the indices of samples that have higher MSE
fraud_indices =anomaly[anomaly['Reconstruction.MSE'] > 0.006].index

# Under sample dataset
under_sample_train_supervised = train_supervised[list(fraud_indices),:]
                     
drf_model = H2ORandomForestEstimator(ntrees=50, max_depth=20, seed=1234)
drf_model.train(x=features, y=response, training_frame=under_sample_train_supervised)

# Confusion Matrix and assertion
cm = drf_model.confusion_matrix()
cm.show()

#Excludes the "Cover_Type" column from the features provided
rf_predictions = drf_model.predict(test[features])
rf_predictions = rf_predictions.cbind(test[response].asfactor()).as_data_frame(use_pandas=True)

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

average_precision = average_precision_score(rf_predictions['Class'],rf_predictions['p1'])

precision, recall, _ = precision_recall_curve(rf_predictions['Class'], rf_predictions['p1'])

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
          average_precision))

#look at this a little bit differently, by manually going through different prediction 
#thresholds and calculating how many cases were correctly classified in the two classes
thresholds = np.arange(0, 1.1, 0.1)
pred_thresholds =pd.DataFrame(dict(actual = rf_predictions['Class']))

for threshold in thresholds:
    prediction = rf_predictions.apply(lambda row: 1 if row['p1'] > threshold else 0, axis=1)
    prediction_true = pred_thresholds['actual']==prediction
    pred_thresholds['{0:.2f}'.format(threshold)] = prediction_true


def gather( df, key, value, cols ):
    id_vars = [ col for col in df.columns if col not in cols ]
    id_values = cols
    var_name = key
    value_name = value
    return pd.melt( df, id_vars, id_values, var_name, value_name )

pred_thresholds_gather = gather( pred_thresholds, 'x', 'y', [x for x in pred_thresholds.columns if x != 'actual'] )

pred_grp = pred_thresholds_gather.groupby(['actual', 'x', 'y'], as_index=False)
pred_grp_counts = pred_grp['y'].agg(['count']).reset_index().rename(columns={'count':'n'})
pred_grp_counts['actual'] = pred_grp_counts['actual'].astype('category')

grouped = pred_grp_counts.groupby(['actual', 'y'])
rowlength = grouped.ngroups/2                         # fix up if odd number of groups

colors = ['red', 'blue']
fig, axs = plt.subplots(figsize=(9,6), 
                        nrows=2, ncols=int(rowlength),     # fix as above
                        gridspec_kw=dict(hspace=0.4)) # Much control of gridspec

targets = zip(grouped.groups.keys(), axs.flatten())
for i, (key, ax) in enumerate(targets):
    x=grouped.get_group(key)['x']
    y=grouped.get_group(key)['n']
    ax.plot(x,y, color=colors[key[0]], label= key[0])
    ax.grid(True)
    ax.set_title('actual={:d} & predict={:d}'.format(*key))
    ax.axvline(x=0.1)
    ax.legend()
plt.xlabel('prediction threshold', fontsize=12)
plt.ylabel('number of instances', fontsize=12)
plt.show()


#validation set accuracy
drf_model.model_performance(test_data = test)



#Pre-trained supervised model
pre_trained_deep_model = H2ODeepLearningEstimator(model_id = "pre_trained_deep_model",
                                                 pretrained_autoencoder  = "model_nn",
                                                 #reproducible = True, #slow - turn off for real problems
                                                 #ignore_const_cols = False,
                                                 hidden = [10, 2, 10], 
                                                 epochs = 100,
                                                 activation = "Tanh")
  
pre_trained_deep_model.train(x=features, y=response, training_frame=train_supervised)

# Confusion Matrix and assertion
cm = pre_trained_deep_model.confusion_matrix()
cm.show()

#Excludes the "Class" column from the features provided
deep_predictions = pre_trained_deep_model.predict(test[features])

pre_trained_deep_model.model_performance(test_data = test)  

deep_predictions = deep_predictions.cbind(test[response].asfactor()).as_data_frame(use_pandas=True)
(deep_predictions['Class'] == deep_predictions['predict']).mean()

confusion_matrix(deep_predictions['Class'], deep_predictions['predict'])

pred_group = deep_predictions.groupby(["Class", "predict"], as_index=False).count()

#Measuring model performance on highly unbalanced data
#An alternative to AUC is to use the precision-recall curve or the sensitivity 
#(recall)-specificity curve. To calculate and plot these metrics, we can use the 
#ROCR package. There are different ways to calculate the area under a curve 
#(see the PRROC package for details) but I am going to use a simple function that 
#calculates the area between every consecutive points-pair of 
#x (i.e. x1 - x0, x2 - x1, etc.) under the corresponding values of y.

def line_integral(x, y):
    dx = np.diff(x)
    end = len(y)
    my = (y[1:(end - 1)] + y[2:end]) / 2
    
    return sum(dx * my)

average_precision = average_precision_score(deep_predictions['Class'], deep_predictions['p1'])

precision, recall, _ = precision_recall_curve(deep_predictions['Class'], deep_predictions['p1'])

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
          average_precision))

from sklearn import metrics
from ggplot import *
fpr, tpr, _ = metrics.roc_curve(deep_predictions['Class'], deep_predictions['p1'])
df_roc = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df_roc, aes(x='fpr', y='tpr')) +\
    geom_line() +\
    geom_abline(linetype='dashed')

#look at this a little bit differently, by manually going through different prediction 
#thresholds and calculating how many cases were correctly classified in the two classes
thresholds = np.arange(0, 1.1, 0.1)
pred_thresholds =pd.DataFrame(dict(actual = deep_predictions['Class']))

for threshold in thresholds:
    prediction = deep_predictions.apply(lambda row: 1 if row['p1'] > threshold else 0, axis=1)
    prediction_true = pred_thresholds['actual']==prediction
    pred_thresholds['{0:.2f}'.format(threshold)] = prediction_true


def gather( df, key, value, cols ):
    id_vars = [ col for col in df.columns if col not in cols ]
    id_values = cols
    var_name = key
    value_name = value
    return pd.melt( df, id_vars, id_values, var_name, value_name )

pred_thresholds_gather = gather( pred_thresholds, 'x', 'y', [x for x in pred_thresholds.columns if x != 'actual'] )

pred_grp = pred_thresholds_gather.groupby(['actual', 'x', 'y'], as_index=False)
pred_grp_counts = pred_grp['y'].agg(['count']).reset_index().rename(columns={'count':'n'})
pred_grp_counts['actual'].astype('category')

grouped = pred_grp_counts.groupby(['actual', 'y'])
rowlength = grouped.ngroups/2                         # fix up if odd number of groups

colors = ['red', 'blue']
fig, axs = plt.subplots(figsize=(9,6), 
                        nrows=2, ncols=int(rowlength),     # fix as above
                        gridspec_kw=dict(hspace=0.4)) # Much control of gridspec

targets = zip(grouped.groups.keys(), axs.flatten())
for i, (key, ax) in enumerate(targets):
    x=grouped.get_group(key)['x']
    y=grouped.get_group(key)['n']
    ax.plot(x,y, color=colors[key[0]], label= key[0])
    ax.grid(True)
    ax.set_title('actual={:d} & predict={:d}'.format(*key))
    ax.axvline(x=0.8)
    ax.legend()
plt.xlabel('prediction threshold', fontsize=12)
plt.ylabel('number of instances', fontsize=12)
plt.show()

#Take only with probabilties of 80% or more
deep_predictions['predict'] = deep_predictions.apply(lambda row: 1 if row['p1'] >= 0.6 else 0, axis=1)
pred_grp = deep_predictions.groupby(['Class', 'predict']).count()
pred_grp_counts = deep_predictions.groupby('Class').count()

pred_grp/pred_grp_counts
