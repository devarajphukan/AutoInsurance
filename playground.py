import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from pprint import pprint

def LabelEncoding(col):
        lbl_enc = LabelEncoder()
        lbl_enc.fit(col)
        cat_col = lbl_enc.transform(col)
        return cat_col

def clean_my_df(df):

    numeric_features = df._get_numeric_data()
    numeric_features = (numeric_features) / (numeric_features.max())
    categorical_features = df.select_dtypes(include=['object'])

    for col_name in categorical_features.columns :
        categorical_features[col_name] = categorical_features[col_name].astype("category",
                                        categories=pd.unique(categorical_features[col_name].values.ravel()))
        data_oneHot = pd.get_dummies(categorical_features[col_name])
        numeric_features = pd.concat([numeric_features,data_oneHot],axis = 1)

    return numeric_features


data = pd.read_csv("data.csv")
data.drop(["Customer"],axis = 1, inplace = True)
data.drop(["Effective To Date"],axis = 1, inplace = True)
labels = pd.DataFrame({"Response":LabelEncoding(data["Response"])})
data.drop(["Response"],axis = 1, inplace = True)
features = clean_my_df(data)


import xgboost as xgb
from collections import OrderedDict
xgb_params = {"objective": "binary:logistic", "eta": 0.01, "max_depth": 8, "seed": 42}
num_rounds = 100
xg_train = xgb.DMatrix(features,label=labels)
bst = xgb.train(xgb_params, xg_train, num_rounds)
importances = bst.get_fscore()
importances = OrderedDict(sorted(importances.items(), key=lambda x: x[1]))
imp_features = features[importances.keys()[-30:]]

from sklearn.cross_validation import *
from sklearn.metrics import *
from sklearn.grid_search import GridSearchCV

xgb_model = xgb.XGBClassifier()
parameters = {'nthread':[4],
              'objective':['binary:logistic'],
              'learning_rate': [0.01],
              'max_depth': [6,9],
              'min_child_weight': [8],
              'subsample': [0.85],
              'colsample_bytree': [0.9],
              'n_estimators': [100]
              }

def score_func(model, X, y_true) :
    preds = model.predict(X)
    return accuracy_score(y_true,preds)
labels = np.array([l[0] for l in labels.values.tolist()])
clf = GridSearchCV(xgb_model,parameters,n_jobs=4,cv=StratifiedKFold(labels,n_folds=2, shuffle=True),
                   scoring=score_func,refit=True)
clf.fit(imp_features,labels)
best_parameters,score,filler = max(clf.grid_scores_,key=lambda x:x[1])
print "\n\nscore:", score, "\n"
pprint(best_parameters)

from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(imp_features,labels)
d_train = xgb.DMatrix(x_train, label=y_train)
d_test = xgb.DMatrix(x_test)
param = best_parameters
param['n_estimators'] = 10
bst = xgb.train(param,d_train)
pred = bst.predict(d_test)
pred_transformed = []

for i in range(len(pred)) : 
    if pred[i] > 0.5 :
        pred_transformed.append(1)
    else :
        pred_transformed.append(0)
pred_transformed = np.array(pred_transformed)
print(classification_report(y_test,pred_transformed))