import argparse
import datetime
from itertools import product
from pathlib import Path
import pickle
import warnings

import pandas as pd 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings('ignore') # This suppresses the sklearn 'UndefinedMetricWarning' that occurs when one class is not predicted in a combination of hyperparameters

data_path = Path('../../data')
results_path = Path('../../results/netherlands')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('nback', type=int,
                        help='[int] number of previous Hb values to use in prediction')
    parser.add_argument('sex', type=str, choices=['men', 'women'],
                        help='[men/women] sex to use in model')
    parser.add_argument('--foldersuffix', type=str, default='',
                        help='[str] optional suffix indicating non-default run')
    args = parser.parse_args()
    return args

def prep_xy(args):
    train = pd.read_pickle(data_path / f'scaled{args.foldersuffix}/{args.sex}_{args.nback}_train.pkl')
    X = train[train.columns[:-1]]
    y = train['HbOK']
    return X, y

def tune_hyperparams(X, y):
    params = {'C': [10, 1, 0.1, 0.01],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf']}

    gridsearch = GridSearchCV(estimator=SVC(class_weight='balanced'),
                              param_grid=params,
                              scoring = 'balanced_accuracy',
                              error_score='raise',
                              cv=5,
                              verbose=False)
    gridsearch.fit(X, y)
    return gridsearch

def save_results(gridsearch, args):
    output_folder = results_path / f'hyperparams{args.foldersuffix}'
    output_folder.mkdir(parents=True, exist_ok=True)  # create folder if it does not yet exist
    filename = output_folder / f'hyperparams_{args.sex}_{args.nback}.pkl'

    with open(filename, 'wb') as handle:
        pickle.dump(gridsearch.cv_results_, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(args):
    X, y = prep_xy(args)
    gs = tune_hyperparams(X, y)
    save_results(gs, args)
    print(f'    Hyperparameters tuned and saved for SVM-{args.nback}, {args.sex}, {args.foldersuffix}')
    
if __name__ == '__main__':
    args = parse_args()
    main(args)