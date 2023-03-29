import argparse
from itertools import product
import pandas as pd
from pathlib import Path
import pickle
from pyprojroot import here
import shap

import matplotlib.pyplot as plt
import numpy as np

results_path = Path('../../results/netherlands/')
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
    filename = f'shapdf_{args.sex}_{args.nback}.pkl'
    shapdf = pickle.load(open(results_path / f'anonshap{args.foldersuffix}' / filename, 'rb'))
        
    df_shap = pd.DataFrame()
    df_x = pd.DataFrame()

    for varname in np.unique(shapdf['variable']):
        df_shap[varname] = list(shapdf.loc[shapdf['variable'] == varname, 'shap'])
        df_x[varname] = list(shapdf.loc[shapdf['variable'] == varname, 'value'])

    plt.figure(figsize=(8, 8))
    fig = shap.summary_plot(df_shap.to_numpy(), df_x, max_display=50, show=False, cmap='viridis')
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.title(f"SHAP values for SVM-{'f' if args.sex == 'women' else 'm'}-{args.nback}, {'full' if args.foldersuffix == '' else 'reduced'} model")
    plt.savefig(plot_path / f'shapvals_{args.sex}_{args.nback}{args.foldersuffix}.png', bbox_inches='tight')
    plt.close()
    
if __name__ == '__main__':
    args = parse_args()
    main(args)