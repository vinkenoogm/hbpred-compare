import argparse
from pathlib import Path
import pickle

import pandas as pd

results_path = Path('../../results/netherlands')

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

def anon_shap(args):
    input_path = results_path / f'shap{args.foldersuffix}/'
    filename1 = f'Xshap_{args.sex}_{args.nback}_{args.n}.pkl'
    filename2 = f'shapvals_{args.sex}_{args.nback}_{args.n}.pkl'
    
    Xshap = pd.read_pickle(input_path / filename1)
    shapvals = pd.read_pickle(input_path / filename2)
    
    shapdf = pd.DataFrame({'variable': list(Xshap.columns) * Xshap.shape[0],
                           'value': Xshap.values.flatten(),
                           'shap': shapvals.flatten()}).sample(frac=1).sort_values('variable').reset_index(drop=True)
    
    output_path = results_path / f'anonshap{args.foldersuffix}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    shapdf.to_pickle(output_path / f'shapdf_{args.sex}_{args.nback}.pkl')
    
def main(args):
    anon_shap(args)
    print(f'    SHAP values for SVM-{args.nback}, {args.sex}, {args.foldersuffix} anonymized and saved')
    
if __name__ == '__main__':
    args = parse_args()
    main(args)