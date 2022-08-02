import argparse
import datetime
from pathlib import Path
import pickle
import warnings

import pandas as pd
import shap

warnings.filterwarnings('ignore')
data_path = Path('../data')
results_path = Path('../testresults')

parser = argparse.ArgumentParser()
parser.add_argument('nback', type=int,
                    help='[int] number of previous Hb values to use in prediction')
parser.add_argument('sex', type=str, choices=['men', 'women'],
                    help='[men/women] sex to use in model')
parser.add_argument('--n', type=int, default=100,
                    help='[int] number of donors to calculate SHAP values on')
parser.add_argument('--foldersuffix', type=str, default='',
                    help='[str] optional suffix indicating non-default run')
args = parser.parse_args()

sex, nback, n, foldersuffix = args.sex, args.nback, args.n, args.foldersuffix

def calc_shap(nback, sex, n=100):
    filename = results_path / f'models{foldersuffix}/clf_{sex}_{nback}.sav'
    clf = pickle.load(open(filename, 'rb'))
    
    test = pd.read_pickle(data_path / f'scaled{foldersuffix}/{sex}_{nback}_test.pkl')
    
    X_test = test[test.columns[:-1]]
    X_shap = shap.sample(X_test, n)
    explainer = shap.KernelExplainer(clf.predict, X_shap)
    shapvals = explainer.shap_values(X_shap, nsamples=100)
    
    output_path = results_path / f'shap{foldersuffix}/'
    output_path.mkdir(parents=True, exist_ok=True)
    filename1 = f'Xshap_{sex}_{nback}_{n}.pkl'
    filename2 = f'shapvals_{sex}_{nback}_{n}.pkl'

    pickle.dump(X_shap, open(output_path / filename1, 'wb'))
    pickle.dump(shapvals, open(output_path / filename2, 'wb'))

calc_shap(nback, sex, n)