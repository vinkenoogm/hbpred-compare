import argparse
import datetime
from pathlib import Path
import pickle
import warnings

import pandas as pd
import shap

warnings.filterwarnings('ignore')
data_path = Path('../../data')
results_path = Path('../results')

def parse_args():
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
    return args

def calc_shap(args):
    filename = results_path / f'models{args.foldersuffix}/clf_{args.sex}_{args.nback}.sav'
    clf = pickle.load(open(filename, 'rb'))
    
    test = pd.read_pickle(data_path / f'scaled{args.foldersuffix}/{args.sex}_{args.nback}_test.pkl')
    scaler = pd.read_pickle(results_path / f'scalers{args.foldersuffix}/{}_{}.pkl')
    
    X_test = test[test.columns[:-1]]
    X_shap = shap.sample(X_test, args.n)
    explainer = shap.KernelExplainer(clf.predict, X_shap)
    shapvals = explainer.shap_values(X_shap)
    
    output_path = results_path / f'shap_subset{args.foldersuffix}/'
    output_path.mkdir(parents=True, exist_ok=True)
    filename1 = f'Xshap_{args.sex}_{args.nback}_{args.n}.pkl'
    filename2 = f'shapvals_{args.sex}_{args.nback}_{args.n}.pkl'

    pickle.dump(X_shap, open(output_path / filename1, 'wb'))
    pickle.dump(shapvals, open(output_path / filename2, 'wb'))

def main(args):
    calc_shap(args)
    print(f'    SHAP values for SVM-{args.nback}, {args.sex}, {args.foldersuffix} calculated and saved')
    
if __name__ == '__main__':
    args = parse_args()
    main(args)