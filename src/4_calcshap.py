import numpy as np
import pandas as pd 
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
import pickle
from sklearn.svm import SVC
import sys
import datetime

import warnings 
warnings.filterwarnings('ignore')

sex = sys.argv[1]
nback = sys.argv[2]
foldersuffix = sys.argv[3]
n = sys.argv[4]

def calc_shap(nback, sex, n=100):
    filename = '../results/models'+foldersuffix+'/clf_' + str(nback) + '.sav'
    clf = pickle.load(open(filename, 'rb'))
    
    index = 0 if sex == 'men' else 1
    val = pd.read_pickle('../../data/scaled'+foldersuffix+'/'+str(sex)+'_'+str(nback)+'_test.pkl')
    clf_s = clf[index]
    X_val = val[val.columns[:-1]]
    X_shap = shap.sample(X_val, n)
    explainer = shap.KernelExplainer(clf_s.predict, X_shap)
    shapvals = explainer.shap_values(X_shap, nsamples=500)
        
    path = '../results/shap'+foldersuffix+'/'
    filename1 = 'Xshap_' + sex + '_' + str(nback) + '_' + str(n) + '.pkl'
    filename2 = 'shapvals_' + sex + '_' + str(nback) + '_' + str(n) + '.pkl'
        
    pickle.dump(X_shap, open(path+filename1, 'wb'))
    pickle.dump(shapvals, open(path+filename2, 'wb'))
        
print('Calculating shap values for ', sex, ' nback', nback, '( N =',  n, ') starting at', datetime.datetime.now())
calc_shap(nback, sex, int(n))