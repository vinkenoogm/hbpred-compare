import numpy as np
import pandas as pd 
import datetime
import pickle
import sys
from sklearn.metrics import classification_report
from sklearn.svm import SVC

usevars = sys.argv[1]
nbacks = sys.argv[2:]

if usevars == 'all':
    hyp_male = {1: {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'},
                2: {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'},
                3: {'C': 0.1, 'gamma': 0.001, 'kernel': 'rbf'},
                4: {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}, 
                5: {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}}

    hyp_female = {1: {'C': 0.1, 'gamma': 0.01, 'kernel': 'rbf'},
                  2: {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'},
                  3: {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'},
                  4: {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'},
                  5: {'C': 1, 'gamma': 0.001, 'kernel': 'rbf'}}
    
elif usevars == 'onlyhb':
    hyp_male = {1: {'C': 0.1, 'gamma': 0.1, 'kernel': 'rbf'},
                2: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'},
                3: {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'},
                4: {'C': 0.1, 'gamma': 0.001, 'kernel': 'rbf'}, 
                5: {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}}

    hyp_female = {1: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'},
                  2: {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'},
                  3: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'},
                  4: {'C': 0.1, 'gamma': 0.01, 'kernel': 'rbf'},
                  5: {'C': 0.1, 'gamma': 0.001, 'kernel': 'rbf'}}   
    
else:
    print('not a valid method, choose all or onlyhb')
    exit()

hyperparams = {'women': hyp_female,
               'men': hyp_male}

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

def do_svm(hyperparam_dict, nback):
    results = []
    clfs = []
    for sex in ['men', 'women']:
        print('Sex:', sex, ' - ', datetime.datetime.now())
        folder = 'scaled' if usevars == 'all' else 'scaled_onlyhb'
        train = pd.read_pickle('../../data/'+folder+'/'+str(sex)+'_'+str(nback)+'_train.pkl')
        test = pd.read_pickle('../../data/'+folder+'/'+str(sex)+'_'+str(nback)+'_test.pkl')

        print('  Training SVM - ', datetime.datetime.now())
        clf = train_svm(train, hyperparam_dict[sex][nback])
        
        print('  Calculating accuracy - ', datetime.datetime.now())
        cl_rep_train = calc_accuracy(clf, train)
        cl_rep_val = calc_accuracy(clf, test)
        results.append(cl_rep_train)
        results.append(cl_rep_val)
        clfs.append(clf)
    return(results, clfs)

for nback in nbacks:
    res, clf = do_svm(hyperparams, int(nback))
    folder = 'models' if usevars == 'all' else 'models_onlyhb'
    filename1 = '../results/'+folder+'/res_' + str(nback) + '.pkl'
    filename2 = '../results/'+folder+'/clf_' + str(nback) + '.sav'
    pickle.dump(res, open(filename1, 'wb'))
    pickle.dump(clf, open(filename2, 'wb'))