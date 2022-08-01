import argparse
import datetime
from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.svm import SVC

data_path = Path('../data')
results_path = Path('../testresults')

parser = argparse.ArgumentParser()
parser.add_argument('nback', type=int,
                    help='[int] number of previous Hb values to use in prediction')
parser.add_argument('sex', type=str, choices=['men', 'women'],
                    help='[men/women] sex to use in model')
parser.add_argument('--foldersuffix', type=str, default='',
                    help='[str] optional suffix indicating non-default run')
args = parser.parse_args()

nback, sex, foldersuffix = args.nback, args.sex, args.foldersuffix

def train_svm(data, hyperparams):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]
    
    hyp_c = hyperparams['C']
    hyp_g = hyperparams['gamma']
    hyp_k = hyperparams['kernel']
    
    clf = SVC(C=hyp_c, gamma=hyp_g, kernel=hyp_k, probability=True, class_weight='balanced')
    clf.fit(X, y.values.ravel())
    
    return clf

def calc_accuracy(clf, data):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]
    
    y_pred = clf.predict(X)
    
    return classification_report(y, y_pred, output_dict=True)

def do_svm(sex, nback):
    train = pd.read_pickle(data_path / f'scaled{foldersuffix}/{sex}_{nback}_train.pkl')
    test = pd.read_pickle(data_path / f'scaled{foldersuffix}/{sex}_{nback}_test.pkl')

    hyps_all = pd.read_pickle(results_path / f'hyperparams{foldersuffix}/hyperparams_{sex}_{nback}.pkl')
    hyps_all = pd.DataFrame.from_dict(hyps_all)
    hyps = hyps_all.loc[hyps_all.rank_test_score == 1, 'params']
    hyps = hyps[hyps.index[0]]

    clf = train_svm(train, hyps)

    cl_rep_train = calc_accuracy(clf, train)
    cl_rep_val = calc_accuracy(clf, test)
    results = [cl_rep_train, cl_rep_val]
    
    return results, clf

output_path = results_path / f'models{foldersuffix}'
output_path.mkdir(parents=True, exist_ok=True) # creates folder if it does not exist

res, clf = do_svm(sex, nback)
filename1 = output_path / f'res_{sex}_{nback}.pkl'
filename2 = output_path / f'clf_{sex}_{nback}.sav'
pickle.dump(res, open(filename1, 'wb'))
pickle.dump(clf, open(filename2, 'wb'))