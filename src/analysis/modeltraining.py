import argparse
import datetime
from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

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
    
    cr = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    
    return cr, cm

def do_svm(args):
    train = pd.read_pickle(data_path / f'scaled{args.foldersuffix}/{args.sex}_{args.nback}_train.pkl')
    test = pd.read_pickle(data_path / f'scaled{args.foldersuffix}/{args.sex}_{args.nback}_test.pkl')

    hyps_all = pd.read_pickle(results_path / f'hyperparams{args.foldersuffix}/hyperparams_{args.sex}_{args.nback}.pkl')
    hyps_all = pd.DataFrame.from_dict(hyps_all)
    hyps = hyps_all.loc[hyps_all.rank_test_score == 1, 'params']
    hyps = hyps[hyps.index[0]]

    clf = train_svm(train, hyps)

    cl_rep_train, conf_matrix_train = calc_accuracy(clf, train)
    cl_rep_test, conf_matrix_test = calc_accuracy(clf, test)
    results = [cl_rep_train, cl_rep_test]
    
    return results, clf, conf_matrix_train, conf_matrix_test

def save_models(args, results, clf, conf_matrix_train, conf_matrix_test):
    output_path = results_path / f'models{args.foldersuffix}'
    output_path.mkdir(parents=True, exist_ok=True) # creates folder if it does not exist
    filename1 = output_path / f'res_{args.sex}_{args.nback}.pkl'
    filename2 = output_path / f'clf_{args.sex}_{args.nback}.sav'
    pickle.dump(results, open(filename1, 'wb'))
    pickle.dump(clf, open(filename2, 'wb'))
    pd.DataFrame(conf_matrix_train).to_csv(output_path / f'confusionmatrix_train_{args.sex}_{args.nback}.csv')
    pd.DataFrame(conf_matrix_test).to_csv(output_path / f'confusionmatrix_test_{args.sex}_{args.nback}.csv')

def main(args):
    res, clf, conf_matrix_train, conf_matrix_test = do_svm(args)
    save_models(args, res, clf, conf_matrix_train, conf_matrix_test)
    print(f'    Trained and saved SVM-{args.nback}, {args.sex}, {args.foldersuffix}')

if __name__ == '__main__':
    args = parse_args()
    main(args)