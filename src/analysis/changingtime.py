import argparse
import pickle
from pathlib import Path

import pandas as pd

data_path = Path('../../data/NL_timefirstdon')
result_path = Path('../../results/netherlands_timefirstdon')

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


def load_files(args):
    data = pd.read_pickle(data_path / f'scaled{args.foldersuffix}/{args.sex}_{args.nback}_test.pkl')
    clf = pickle.load(open(result_path / f'models{args.foldersuffix}/clf_{args.sex}_{args.nback}.sav', 'rb'))
    scaler = pickle.load(open(result_path / f'scalers{args.foldersuffix}/{args.sex}_{args.nback}.pkl', 'rb'))
    return data, clf, scaler


def make_preds(data, clf):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]
    y_pred = clf.predict(X)
    return(y_pred)


def main(args):
    data, clf, scaler = load_files(args)
    data_res = data.copy()
    y_pred_first = make_preds(data, clf)
    data_res['HbOK_pred'] = y_pred_first

    timecols = []
    if args.foldersuffix == '':
        timecols.extend(['DaysSinceFirstDon'])
    for n in range(1, args.nback+1):
        timecols.extend(['DaysSinceHb'+str(n)])

    for timestep in range(-364, 371, 7):
        data_timestep = data.copy()
        data_timestep[data_timestep.columns[:-1]] = scaler.inverse_transform(data_timestep[data_timestep.columns[:-1]])
        data_timestep[timecols] = data_timestep[timecols].add(timestep)
        data_timestep['month'] = (data_timestep['month'] - 1 + round(timestep/30)) % 12 + 1
        data_timestep['age'] = data_timestep['age'] + (timestep / 365)
        data_timestep[data_timestep.columns[:-1]] = scaler.transform(data_timestep[data_timestep.columns[:-1]])

        y_pred = make_preds(data_timestep, clf)

        varname = 'HbOK_pred_' + str(timestep)
        data_res[varname] = y_pred
        if timestep == 0:
            data_res = data_res.copy()

    output_path = data_path / f'pred_timechange{args.foldersuffix}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_res.to_pickle(output_path / f'data_res_{args.sex}_{args.nback}.pkl')
    print(f'    Predictions for SVM-{args.nback}, {args.sex}, {args.foldersuffix} calculated and saved')

    
if __name__ == '__main__':
    args = parse_args()
    main(args)