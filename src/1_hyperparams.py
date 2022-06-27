import numpy as np
import pandas as pd 
import pickle
import datetime
import sys
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import UndefinedMetricWarning

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


nback = int(sys.argv[1])
sex = sys.argv[2]

print(sex)


train = pd.read_pickle('../../data/scaled_onlyhb/'+str(sex)+'_'+str(nback)+'_train.pkl')
X = train[train.columns[:-1]]
y = train['Hb_deferral']

params = {'C': [100, 10, 1, 0.1],
          'gamma': [1, 0.1, 0.01, 0.001],
          'kernel': ['rbf']}

gridsearch = GridSearchCV(estimator=SVC(class_weight='balanced'),
                          param_grid=params,
                          scoring = 'balanced_accuracy',
                          error_score='raise',
                          cv=5,
                          verbose=2)
gridsearch.fit(X, y)

filename = '../results/hyperparams_onlyhb/output_hyperparams_' + str(sex) + '_' + str(nback) + '.pkl'
with open(filename, 'wb') as handle:
    pickle.dump(gridsearch.cv_results_, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename, 'a') as f:
    print(str(datetime.datetime.now()), '\n\n', gridsearch.cv_results_)

