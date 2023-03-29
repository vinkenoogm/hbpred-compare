from pathlib import Path

import pandas as pd

path_FL = Path('../results/finland')
path_NL = Path('../results/netherlands')

path_results = Path('../results/comparison')

def pretty_results(filenames, foldersuffix, results_path):
    reslist = []
    for index1, filename in enumerate(filenames):
        _, sex, nback = filename.split('_')
        res = pd.read_pickle(results_path / f'models{foldersuffix}' / f'{filename}.pkl')
        for index2, cr in enumerate(res):
            traintest = ['train', 'test'][index2 % 2]
            reslist.append([int(nback), traintest, sex, cr['0']['precision'], cr['0']['recall'], cr['0']['support'],
                            cr['1']['precision'], cr['1']['recall'], cr['1']['support']])
    res = pd.DataFrame(reslist).set_axis(['nback', 'traintest', 'sex', 'low_precision', 'low_recall', 'low_support',
                                          'ok_precision', 'ok_recall', 'ok_support'], axis=1)
    return res
    
def agg_run_results(path, save=False):
    res_df = pretty_results([f'res_{sex}_{nback}' for sex in ['men','women'] for nback in range(1,6)], '', path)
    res_df_hbonly = pretty_results([f'res_{sex}_{nback}' for sex in ['men','women'] for nback in range(1,6)], '_hbonly', path)
    res_tog = pd.concat([df.assign(variables=k) for k,df in {'all':res_df, 'hbonly':res_df_hbonly}.items()])
    res_tog = res_tog.loc[res_tog['traintest'] == 'test', ['nback','sex','ok_precision','ok_recall','variables']]
    
    get_diff_table(res_tog, save=save)
    
    return res_tog

def get_diff_table(res, save=False):
    res = res.pivot(index=['nback','sex'], columns='variables')
    res['ok_precision', 'diff'] = res['ok_precision', 'all'] - res['ok_precision', 'hbonly']
    res['ok_recall', 'diff'] = res['ok_recall', 'all'] - res['ok_recall', 'hbonly']
    
    if save:
        res.to_pickle(path_results / f'{save}.pkl')
    
    return res


def main():
    res_NL = agg_run_results(path_NL, save='performance_NL')
    res_FL = agg_run_results(path_FL, save='performance_FL')
    res_tog = pd.concat([df.assign(country=k) for k,df in {'Finland':res_FL, 'Netherlands':res_NL}.items()])
    res_tog.to_pickle(path_results / 'performance_both.pkl')

    
if __name__ == '__main__':
    main()