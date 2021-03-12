from inspect import Parameter
import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
from time import time
import pprint
import joblib
import yaml 
import json 
import warnings
warnings.filterwarnings("ignore")

# Classifiers
from catboost import CatBoostClassifier

# Model selection
from sklearn.model_selection import StratifiedKFold, PredefinedSplit

# Metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, VerboseCallback, DeltaXStopper
from skopt.space import Real, Categorical, Integer
from time import time

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

frac = param['train_size']
train_sz = int(tr_df.shape[0]*frac)
val_sz  = tr_df.shape[0]-train_sz
ind_list = [-1,]*train_sz+[0,]*val_sz

prd = PredefinedSplit(ind_list)

def report_perf(optimizer, X, y,title, callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    start = time()
    if callbacks:
        optimizer.fit(X, y,callback=callbacks)
    else:
        optimizer.fit(X, y)
    d=pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           +u"\u00B1"+" %.3f") % (time() - start, 
                                  len(optimizer.cv_results_['params']),
                                  best_score,
                                  best_score_std))    
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params

avg = param['f1-type']
sd = param['seed']

score = make_scorer(f1_score, average=avg)
# score = make_scorer(accuracy_score)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=sd)

loss_func = param['loss_function']
device = param['device']

clf = CatBoostClassifier(thread_count=3,
                         loss_function=loss_func,
                         od_type = 'Iter',
                         task_type=device,
                         verbose= False)

search_spaces = {'iterations': Integer(10, 1000),
                 'depth': Integer(1, 8),
                 'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                 'random_strength': Real(1e-9, 10, 'log-uniform'),
                 'bagging_temperature': Real(0.0, 1.0),
                 'border_count': Integer(1, 255),
                 'l2_leaf_reg': Integer(2, 30)}

n_iteration = param['n_iter']

opt = BayesSearchCV(clf,
                    search_spaces,
                    scoring=score,
                    cv=skf,
                    n_iter=n_iteration,
                    n_jobs=1,  # use just 1 job with CatBoost in order to avoid segmentation fault
                    return_train_score=False,
                    optimizer_kwargs={'base_estimator': 'GP'},
                    refit=True,
                    random_state=sd)

best_params = report_perf(opt, X, y,'CatBoost', 
                          callbacks=[VerboseCallback(100), 
                                     DeadlineStopper(60*10)])

with open('parameters.json', 'w') as pd:
    json.dump(best_params, pd)