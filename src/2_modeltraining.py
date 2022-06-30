import numpy as np
import pandas as pd 
import datetime
import pickle
import sys
from sklearn.metrics import classification_report
from sklearn.svm import SVC

foldersuffix = sys.argv[1]
nbacks = sys.argv[2:]

def train_svm(data, hyperparams):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]
    
    hyp_c = hyperparams['C']
    hyp_g = hyperparams['gamma']
    hyp_k = hyperparams['kernel']
    
    clf = SVC(C = hyp_c, gamma = hyp_g, kernel = hyp_k, probability=True, class_weight='balanced')
    clf.fit(X, y.values.ravel())
    
    return(clf)

def calc_accuracy(clf, data):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]
    
    y_pred = clf.predict(X)
    
    return(classification_report(y, y_pred, output_dict=True))

def do_svm(nback):
    results = []
    clfs = []
    for sex in ['men', 'women']:
        print('Sex:', sex, ' - ', datetime.datetime.now())
        train = pd.read_pickle('../../data/scaled'+foldersuffix+'/'+str(sex)+'_'+str(nback)+'_train.pkl')
        test = pd.read_pickle('../../data/scaled'+foldersuffix+'/'+str(sex)+'_'+str(nback)+'_test.pkl')
        
        hyps_all = pd.read_pickle('../results/hyperparams'+foldersuffix+'/output_hyperparams_'+sex+'_'+str(nback)+'.pkl')
        hyps = hyps_all.loc[hyps.rank_test_score == 1, 'params']
        hyps = hyps[hyps.index[0]]
        
        print('  Training SVM - ', datetime.datetime.now())
        clf = train_svm(train, hyps)
        
        print('  Calculating accuracy - ', datetime.datetime.now())
        cl_rep_train = calc_accuracy(clf, train)
        cl_rep_val = calc_accuracy(clf, test)
        results.append(cl_rep_train)
        results.append(cl_rep_val)
        clfs.append(clf)
    return(results, clfs)

for nback in nbacks:
    res, clf = do_svm(int(nback))
    filename1 = '../results/models'+foldersuffix+'/res_' + str(nback) + '.pkl'
    filename2 = '../results/models'+foldersuffix+'/clf_' + str(nback) + '.sav'
    pickle.dump(res, open(filename1, 'wb'))
    pickle.dump(clf, open(filename2, 'wb'))