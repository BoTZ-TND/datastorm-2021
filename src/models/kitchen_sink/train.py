import os 
import sys
import warnings

from sklearn.utils import shuffle
warnings.filterwarnings('ignore')
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

param = yaml.safe_load(open('params.yaml'))['train']

np.set_printoptions(suppress=True)

if len(sys.argv) != 2:
    sys.stderr.write('Argument error. Usage:\n')
    sys.stderr.write('\tpython featurization.py data-dit-path feature-dir-path\n')
    sys.exit(1)

df_path = os.path.join(sys.argv[1])

tr_df = pd.read_csv(df_path)
X = tr_df.iloc[:, :-1]
y = tr_df.iloc[:, -1]

# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

lr = param['lr']
n_est = param['n_estimators']
obj = param['objective']


xgb = XGBClassifier(learning_rate=lr, n_estimators=n_est, objective=obj, shuffle=True,
                    silent=True, nthread=1)

folds = param['folds']
param_comb = param['param_comb']
score = param['score']
seed = param['seed']

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = seed)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring=score, n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=seed )

random_search.fit(X, y)

with open('parameters.json', 'w') as pd:
    json.dump(random_search.best_params_, pd)