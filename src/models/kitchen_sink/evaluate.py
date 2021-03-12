import os
import sys
import yaml
import json
import pickle
import dagshub
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

param = yaml.safe_load(open('params.yaml'))['evaluate']

train_input = os.path.join(sys.argv[1], 'cba_train.csv')
val_input = os.path.join(sys.argv[1], 'cba_validation.csv')
model_path = os.path.join(sys.argv[2])

tr_df = pd.read_csv(train_input)
val_df = pd.read_csv(val_input)

with open('parameters.json', 'r')as pf:
    params = json.load(pf)

lr = param['lr']
n_est = param['n_estimators']
obj = param['objective']


xgbc = XGBClassifier(learning_rate=lr, n_estimators=n_est, objective=obj,
                    silent=True, nthread=1, **params)


xgbc.fit(tr_df.iloc[:, :-1], tr_df.iloc[:, -1])

with open(model_path,'wb') as pf:
    pickle.dump(xgbc, pf)

val_len = val_df.shape[0]
tr_sub = tr_df.sample(n=val_len, random_state=443)
y_tr_pred = xgbc.predict(tr_sub.iloc[:, :-1])
y_train = tr_sub.iloc[:, -1]

y_ts_pred = xgbc.predict(val_df.iloc[:, :-1])
y_test = val_df.iloc[:, -1]

with open('score.json','w') as pf:
    json.dump({'train' : {'accuracy' : accuracy_score(y_train, y_tr_pred), 'precision' : precision_score(y_train, y_tr_pred, average='macro'), 'recall' : recall_score(y_train, y_tr_pred, average='macro'), 'f1-score' : f1_score(y_train, y_tr_pred, average='macro')},
    'test' : {'accuracy' : accuracy_score(y_test, y_ts_pred), 'precision' : precision_score(y_test, y_ts_pred, average='macro'), 'recall' : recall_score(y_test, y_ts_pred, average='macro'), 'f1-score' : f1_score(y_test, y_ts_pred, average='macro')}
    }, pf)