import argparse
import datetime
from itertools import product
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

data_path = Path('../../data')
results_path = Path('../../results/netherlands')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldersuffix', type=str, default='',
                        help='[str] optional suffix indicating non-default run')
    args = parser.parse_args()
    return args

def pretty_results(filenames, args):
    reslist = []
    for index1, filename in enumerate(filenames):
        _, sex, nback = filename.split('_')
        res = pd.read_pickle(results_path / f'models{args.foldersuffix}' / f'{filename}.pkl')
        for index2, cr in enumerate(res):
            traintest = ['train', 'test'][index2 % 2]
            reslist.append([int(nback), traintest, sex, cr['0']['precision'], cr['0']['recall'], cr['0']['support'],
                            cr['1']['precision'], cr['1']['recall'], cr['1']['support']])
    res = pd.DataFrame(reslist).set_axis(['nback', 'traintest', 'sex', 'ok_precision', 'ok_recall', 'ok_support',
                                          'low_precision', 'low_recall', 'low_support'], axis=1)
    return res
    
def get_scores(res_df):
    res_df['old_defrate'] = res_df['low_support'] / (res_df['low_support'] + res_df['ok_support'])
    res_df['new_defrate'] = 1 - res_df['ok_precision']
    res_df['missed_dons'] = 1 - res_df['ok_recall']
    res_df['prevented_defs'] = res_df['low_recall']
    res_df['missed_per_prev'] = (res_df['ok_support'] - res_df['ok_recall'] * res_df['ok_support']) / (res_df['low_support'] - (1 - res_df['ok_precision']) * res_df['ok_support'])
    
    res_df['old_def_n'] = res_df['low_support']
    res_df['new_def_n'] = round((1 - res_df['ok_precision']) * res_df['ok_support'])
    res_df['old_don_n'] = res_df['ok_support']
    res_df['new_don_n'] = res_df['ok_recall'] * res_df['ok_support']
    
    return res_df
 
def plot_precision_recall(res_df, measure, ylim, ylab, args, save=False):
    pl_df = res_df.groupby(['sex', 'traintest'])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for name, group in pl_df:
        y = 0 if name[0] == 'women' else 1
        off = -0.2 if name[1] == 'train' else 0.2
        hatch = '/' if name[1] == 'test' else ''
        ax[y].bar(group.nback + off, group[measure], label=name[1], width=0.4, edgecolor='black', hatch=hatch)
        ax[y].set_ylim(ylim)
        ax[y].set_xticks(group.nback)
        ax[y].set_xticklabels(['SVM-1','SVM-2','SVM-3','SVM-4','SVM-5'], size='large')
        ax[y].set_ylabel(ylab, size='large')
    
    legloc = 'upper right' if measure == 'low_precision' else 'lower right'
    legbox = (1, 1) if measure == 'low_precision' else (1, 0)
    ax[0].legend(labels=['train', 'test'], bbox_to_anchor=legbox, loc=legloc, title='Dataset')
    ax[1].legend(labels=['train', 'test'], bbox_to_anchor=legbox, loc=legloc, title='Dataset')
    
    ax[0].set_title('Women')
    ax[1].set_title('Men')

    fig.tight_layout()
    plt.set_cmap('tab20')
    
    if save:
        plot_path = results_path / 'plots_performance'
        plot_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path / f'{save}{args.foldersuffix}.png')
    
def load_probas(args, sexes=('men', 'women'), nbacks=range(1, 6)):
    dfs = []
    for sex, nback in product(sexes, nbacks):
        df = pd.read_pickle(results_path / f'probas{args.foldersuffix}' / f'proba_{sex}_{nback}.pkl')
        df['sex'] = sex
        df['nback'] = nback
        dfs.append(df)
    return pd.concat(dfs)

def plot_prs(probas, def_f, def_m, save=False):
    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    
    for x, sex in enumerate(['men', 'women']):
        df = probas.loc[probas['sex'] == sex, ]
        
        for key, group in df.groupby('nback'):
            precision_0, recall_0, thresholds_0 = precision_recall_curve(group.HbOK, group.prob_low, pos_label=0)
            precision_1, recall_1, thresholds_1 = precision_recall_curve(group.HbOK, group.prob_ok, pos_label=1)
            
            aupr_0 = round(average_precision_score(group.HbOK, group.prob_low, pos_label=0), 3)
            aupr_1 = round(average_precision_score(group.HbOK, group.prob_ok, pos_label=1), 3)

            ax[1,x].plot(recall_0, precision_0, label='SVM-'+str(key)+', AUPR: '+str(aupr_0))
            ax[0,x].plot(recall_1, precision_1, label='SVM-'+str(key)+', AUPR: '+str(aupr_1))
        
        ax[0,x].set_title('PR-curve class deferral - ' + ['men', 'women'][x])
        ax[0,x].set_xlabel('Recall')
        ax[0,x].set_ylabel('Precision')
        ax[1,x].set_title('PR-curve class no deferral - ' + ['men', 'women'][x])
        ax[1,x].set_xlabel('Recall')
        ax[1,x].set_ylabel('Precision')
        ax[0,x].legend(loc='upper right')
        ax[1,x].legend(loc='lower left')
        ax[0,x].set_ylim(0,0.5)
    
    #horizontal lines for baseline
    ax[0,0].hlines(y=def_m, xmin=0, xmax=1, color='grey', ls='--')
    ax[1,0].hlines(y=1-def_m, xmin=0, xmax=1, color='grey', ls='--')
    ax[0,1].hlines(y=def_f, xmin=0, xmax=1, color='grey', ls='--')
    ax[1,1].hlines(y=1-def_f, xmin=0, xmax=1, color='grey', ls='--')
    
    if save:
        plot_path = results_path / 'plots_performance/'
        plot_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path / f'{save}.png')
    
def main(args):
    res_df = pretty_results([f'res_{sex}_{nback}' for sex in ['men','women'] for nback in range(1,6)], args)
    res_df = get_scores(res_df)

    plot_precision_recall(res_df, 'ok_precision', (0.95, 1), 'Precision\nclass no deferral', args, save='ok_precision')
    plot_precision_recall(res_df, 'ok_recall', (0, 1), 'Recall\nclass no deferral', args, save='ok_recall')
    plot_precision_recall(res_df, 'low_precision', (0, 0.15), 'Precision\nclass deferral', args, save='low_precision')
    plot_precision_recall(res_df, 'low_recall', (0, 1), 'Recall\nclass deferral', args, save='low_recall')
    
    sexes = ['men', 'women']
    nbacks = range(1, 6)

    for sex, nback in product(sexes, nbacks):
        clf = pickle.load(open(results_path / f'models{args.foldersuffix}/clf_{sex}_{nback}.sav', 'rb'))
        test = pd.read_pickle(data_path / f'scaled{args.foldersuffix}/{sex}_{nback}_test.pkl')
        y_true = test[test.columns[-1:]].copy()
        y_pred = clf.predict_proba(test[test.columns[:-1]])
        y_true[['prob_low', 'prob_ok']] = y_pred

        output_path = results_path / f'probas{args.foldersuffix}'
        output_path.mkdir(parents=True, exist_ok=True)
        pickle.dump(y_true, open(output_path / f'proba_{sex}_{nback}.pkl', 'wb'))
    
    proba_m = pd.read_pickle(results_path / f'probas{args.foldersuffix}/proba_men_1.pkl')
    proba_f = pd.read_pickle(results_path / f'probas{args.foldersuffix}/proba_women_1.pkl')
    def_m = 1-np.mean(proba_m.HbOK)
    def_f = 1-np.mean(proba_f.HbOK)

    probas = load_probas(args)
    plot_prs(probas, def_f, def_m, save=f'PR_curve{args.foldersuffix}')