# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:55:56 2019

@author: Admin
"""
#look at file : Hyperparameter Optimization For Xgboost using RandomizedSearchCV
#kaggle :-  https://www.kaggle.com/hj5992/bank-churn-modelling
#buisness problem : predict bank dataset wheather cutomers exited from bank in future 

# Dataset : Bank cutomer information, we have varibles of age, credit_card,Balance, num of prod,Estimated salary  
# based on that whether that perticular person exit the bank in future or not
# model has prdicted cutomer exited then bank will be lauch new offer so they not leave or stay into the bank....
# data 10000 rows and 14 columns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bank = pd.read_csv('E:\\Data science\\DSothers\\projects\\chrun_bank\\Churn_Modelling.csv')

bank.columns
bank.shape
bank.isnull().sum()  #not null
bank.describe()
bank.dtypes  # 3 are cateogrical
bank.info()
bank['Exited'].value_counts()  # Data is imbalaced

import seaborn as sns
sns.pairplot(bank, hue='Exited')    #data complety non linear , better to do knn or random forest
# we see data is too much overlapped apply : knn , RF

#correlation
corr = bank.corr()
col = corr.index
plt.figure(figsize=(20,20))
g = sns.heatmap(bank[col].corr(),annot =True , cmap="RdYlGn")
# see here is Age , balace , active member high correlated to our Exited
# row no , cust id is not correlation of Exited dependent varible , so remove from our analysis

bank.head()
#to remove first three columns from dataset
bank = bank.iloc[:,3:]

#Getting bar-plot for catogorical varible
# 0 not Exited , 1 for Exited from bank
sns.countplot(x="Exited" , data = bank, palette = "hls")    #0 for not exited and 1 for exited , 0= 8000, 1=2000
sns.countplot(x="NumOfProducts", data=bank)                 #most of customers has 1 or 2 products
sns.countplot(x="HasCrCard", data=bank)                     #70% has credit cards 
sns.countplot(x="IsActiveMember", data=bank)                #Equally cutomers has Acitve and non Active

pd.crosstab(bank.Exited, bank.NumOfProducts).plot(kind = "bar")  #not exited cutomers has 1 or 2 products
pd.crosstab(bank.Exited, bank.HasCrCard).plot(kind = "bar")       #Exited customer has low Credit card holders, 0 for 60% , 1 for 10%
pd.crosstab(bank.Exited, bank.IsActiveMember).plot(kind = "bar")   # 12% Exzited customer has no Active with bank

#Data visulization using boxplot of countineous varibles wrt to category varible
sns.boxplot(x="Exited",y="Balance", data= bank,palette = "hls")  
sns.boxplot(x="Exited",y="Age", data= bank,palette = "hls")        #exited cutomers has age of 40 to 50
sns.boxplot(x="Exited",y="CreditScore", data= bank,palette = "hls")
sns.boxplot(x="Exited",y="EstimatedSalary", data= bank,palette = "hls")  #loan person they dont have money

#this catogorical varibles to make it dummies
bank.loc[:,['Geography','Gender']]

sns.heatmap(bank.corr())

#dummy
dummy = pd.get_dummies(bank.loc[:,['Geography','Gender']], drop_first=True)  #drop_first : to prevent dummy varible trap, model has understand skip varible from data
bank.drop(['Geography','Gender'], axis=1, inplace= True)
new_bank = pd.concat([bank,dummy],axis=1)
new_bank.head()
new_bank.dtypes

# Data completly overlapped to apply model KNN and RandomForest , xgboost

# Training and Testing data: stratified sampling 
# beacause of y has catogorical to make sense of do stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit as sss
split = sss(n_splits = 5, test_size = 0.2 , random_state = 42)
for train_index , test_index in split.split(new_bank, new_bank['Exited']):
    bank_train = new_bank.loc[train_index]
    bank_test = new_bank.loc[test_index]

y = bank_train['Exited']
X = bank_train.drop(['Exited'], axis = 1)

y_t = bank_test['Exited']
X_t = bank_test.drop(['Exited'], axis = 1)

#----------------------------KNN----------------------------------------------
#KNN : nearest neighbor , new data points prediction happens to find nearest one (depend upon k) and 
#depend upon majority or their is amibiguty (selection based on distance) likewies new data points assign that class/category

from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNC(n_neighbors = k)
    score=cross_val_score(knn_classifier,new_bank.drop(['Exited'], axis = 1),new_bank['Exited'],cv=10)   #Experiments
    knn_scores.append(score.mean())

plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')

knn_classifier = KNC(n_neighbors = 14)
#cross validation score : train and test data gives differnt accuracy , to do cross_val_score it work like iteration base train and test , he do all combination experimetns
#below cv=10 means , 10 experiments like( train and test 9:1 out of 10, to 10 iternation till the end), finally cal avarage of 10 experiments results
score=cross_val_score(knn_classifier,new_bank.drop(['Exited'], axis = 1),new_bank['Exited'],cv=10)   #its also help to which type alogorithm select for this dataset
score.mean()      #0.7938
score

# build with train and test data calculate accuracy
acc= []
#running KNN model to 3 to 50 neighbors
for i in range(3,50,2):
    neigh = KNC(n_neighbors = i)
    neigh.fit(X,y)
    train_acc = np.mean(neigh.predict(X) == y)
    test_acc = np.mean(neigh.predict(X_t) == y_t)
    acc.append([train_acc, test_acc])
    
plt.plot(np.arange(3,50,2),[i[0] for i in acc], "bo-")
plt.plot(np.arange(3,50,2), [i[1] for i in acc], "ro-")
plt.legend(["train","test"])
# we select our k =15

near5 = KNC(n_neighbors = 15)
near5.fit(X, y)
train_acc5 = np.mean(near5.predict(X) == y)
train_acc5   #79.7125
test_acc5 = np.mean(near5.predict(X_t)== y_t)
test_acc5  #79.47

#----------------------Random Forest-------------------------------------------
# used ensembled technique to build combine multiple model (Decision Tree), bagging , finally prediction depends upon multiple DT o/p majority
from sklearn.ensemble import RandomForestClassifier
randomforest_classifier= RandomForestClassifier(n_estimators=10)
score=cross_val_score(randomforest_classifier,new_bank.drop(['Exited'], axis = 1),new_bank['Exited'],cv=10)
score.mean()  #0.85

RFmodel = RandomForestClassifier(n_jobs = 2, oob_score=True , n_estimators = 10, criterion = 'entropy')

RFmodel.fit(X , y)   #random_state = none (boosting)
RFmodel.estimators_  #15 decision tree 
RFmodel.classes_ # class labels (output)
RFmodel.n_classes_ # Number of levels in class labels 
RFmodel.n_features_  # Number of input features in model 8 here.
RFmodel.n_outputs_ # Number of outputs when fit performed
RFmodel.oob_score_  # 0.8601

#predict on test data
X_t['pred'] = RFmodel.predict(X_t)
#del X_t['pred']

from sklearn.metrics import confusion_matrix, recall_score, precision_score , f1_score  , classification_report , accuracy_score
confusion_matrix(y_t , X_t['pred'])
#y_t.value_counts()
#206+201 : 206 cutomers wrongly predict false negative/not exited (actually its total is 407 as exited)

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_t , X_t['pred'])
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

accuracy = 0.866 
precision = 0.857 
recall = 0.866 
f1 = 0.854

print(classification_report(y_t,X_t['pred']))
#-----------------------------XGBOOST----------------------------------------------------------------------------------
## Hyper Parameter Optimization : pass fisrt those value to model , using that model has do all permutation and combination to find
# best value of our dataset
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}

from sklearn.model_selection import RandomizedSearchCV   #used for which paramerter best for that datasets
import xgboost

classifier = xgboost.XGBClassifier()
random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc',n_jobs=-1,cv=10,verbose=3)
random_search

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
        
from datetime import datetime
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(new_bank.drop(['Exited'], axis = 1),new_bank['Exited'])  #USING that pass all list to model , its find best values for this dataset
timer(start_time) # timing ends here for "start_time" variable

random_search.best_estimator_   #use these parameter for final model
random_search.best_params_

final = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.0,
              learning_rate=0.15, max_delta_step=0, max_depth=5,
              min_child_weight=7, missing=None, n_estimators=10, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

score = cross_val_score(final, new_bank.drop(['Exited'], axis = 1),new_bank['Exited'] , cv=10)
score
score.mean()  #82

#---------------------------------------------------------------------------------------------------------------
#we select random forest for that dataset
#see here result not great to Exited(1) customer , beacause dataset is imbalaced and result might be goes to no Exited , 
#model has high data points from non exited customers results may affect

# To make data balance and aging do random forest to it to check to increase our accuracy
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
os =  RandomOverSampler(ratio=1)                 #ratio=1  add records to lower category that much both becomes equal
X_res,y_res=os.fit_sample(new_bank.iloc[:,0:8],new_bank['Exited'])

#new_bank.iloc[:,0:8]

#from imblearn.under_sampling import NearMiss
#nm = NearMiss(random_state=42)
#X_res,y_res=nm.fit_sample(new_bank.drop(['Exited'], axis = 1),new_bank['Exited'])
X_res.shape,y_res.shape

new_yres=pd.DataFrame(y_res)
new_yres[0].value_counts()  #to add same dimestion of data from 1 category side 

from collections import Counter
print('Original dataset shape {}'.format(Counter(new_bank['Exited'])))
print('Resampled dataset shape {}'.format(Counter(y_res)))   #two catoegory make same propotional first category that much record add size of second category and irrespective other side of category

randomforest_classifier= RandomForestClassifier(n_estimators=10)
score=cross_val_score(randomforest_classifier,X_res,y_res,cv=10)
score.mean()  #0.95


model = randomforest_classifier.fit(X_res,y_res)
pred = model.predict(X_res)
score = accuracy_score(pred,y_res)
score #99

confusion_matrix(pred, y_res)

import pickle 

import os
os.getcwd()

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))  #precompiled foremat model file

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
X.columns
print(model.predict([[999,32,3,50000,3,1,1,20000]]))

