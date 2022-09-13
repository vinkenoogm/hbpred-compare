import argparse
from itertools import product
import pandas as pd
from pathlib import Path
import pickle
from pyprojroot import here
import shap

import matplotlib.pyplot as plt

results_path = Path('../../results/netherlands/shap/')
plot_path = Path('../../results/netherlands/plots_shap/')
plot_path.mkdir(parents=True, exist_ok=True)

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

def main(args):
    filename1 = f'Xshap_{args.sex}_{args.nback}_{args.n}.pkl'
    filename2 = f'shapvals_{args.sex}_{args.nback}_{args.n}.pkl'

    X_test = pickle.load(open(results_path / filename1, 'rb'))
    shapvals = pickle.load(open(results_path / filename2, 'rb'))

    plt.figure(figsize=(8, 8))
    fig = shap.summary_plot(-1*shapvals, X_test, max_display=50, show=False)
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.title(f'SHAP values for SVM-{args.nback}, {args.sex}')
    plt.savefig(plot_path / f'shapvals_{args.sex}_{args.nback}{args.foldersuffix}.png', bbox_inches='tight')
    plt.close()
    
if __name__ == '__main__':
    args = parse_args()
    main(args)