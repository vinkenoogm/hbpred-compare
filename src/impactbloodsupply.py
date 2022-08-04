import bisect
import datetime
from itertools import product
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

data_path = Path('../../data')
results_path = Path('../testresults')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldersuffix', type=str, default='',
                        help='[str] optional suffix indicating non-default run')
    args = parser.parse_args()
    return args

def get_closest_pred(x):
    intvals = list(range(-364, 371, 7))
    i = bisect.bisect_right(intvals, x)
    return(intvals[i])

def get_first_pos_pred(x, fulldf):
    firstcol = list(fulldf.columns).index(f"HbOK_pred_{int(x['first_possible_invite'])}")
    all_pred_1 = x[firstcol:-2][x[firstcol:-2] == 1]
    if len(all_pred_1) == 0:
        return(np.nan)
    else:
        return(x[firstcol:-2][x[firstcol:-2] == 1].index[0][10:])

def gather_results(sex, nbacks, foldersuffix=''):
    df = list()
    for nback in nbacks:
        scaler = pickle.load(open(results_path / f'scalers{foldersuffix}/{sex}_{nback}.pkl', 'rb'))
        dfsub = pd.read_pickle(data_path / f'pred_timechange{foldersuffix}/data_res_{sex}_{nback}.pkl')
        dfsub[dfsub.columns[:-107]] = scaler.inverse_transform(dfsub[dfsub.columns[:-107]])
        df.append(dfsub)
    df = pd.concat(df)
    df = df[~df.index.duplicated(keep='first')]
    interval = 57 if sex == 'men' else 122
    df['first_possible_donation'] = df['DaysSinceHb1'] * -1 + interval
    df['first_possible_invite'] = df['first_possible_donation'].apply(get_closest_pred)
    df['first_pos_pred'] = pd.to_numeric(df.apply(get_first_pos_pred, fulldf=df, axis=1))
    return(df)
    
def plot_newinvites(args, df_f, df_m, save=False):
    fig, ax = plt.subplots(2, 2, figsize=(12,12))

    df_f_nonan = df_f.dropna(subset=['first_pos_pred'])
    df_m_nonan = df_m.dropna(subset=['first_pos_pred'])

    ax[0,0].hist(df_f_nonan.loc[df_f_nonan['HbOK'] == 1, 'first_pos_pred'], bins=range(-364, 371, 30), cumulative=True, 
                 facecolor='xkcd:purple', edgecolor='k',fill=True)
    ax[0,1].hist(df_f_nonan.loc[df_f_nonan['HbOK'] == 0, 'first_pos_pred'], bins=range(-364, 371, 30), cumulative=True, 
                 facecolor='xkcd:light blue', hatch='/', edgecolor='k',fill=True)
    ax[1,0].hist(df_m_nonan.loc[df_m_nonan['HbOK'] == 1, 'first_pos_pred'], bins=range(-364, 371, 30), cumulative=True, 
                 facecolor='xkcd:purple', edgecolor='k',fill=True)
    ax[1,1].hist(df_m_nonan.loc[df_m_nonan['HbOK'] == 0, 'first_pos_pred'], bins=range(-364, 371, 30), cumulative=True, 
                 facecolor='xkcd:light blue', hatch='/', edgecolor='k',fill=True)

    ax[0,0].set_title('Female donors, not deferred at t=0')
    ax[0,1].set_title('Female donors, deferred at t=0')
    ax[1,0].set_title('Male donors, not deferred at t=0')
    ax[1,1].set_title('Male donors, deferred at t=0')

    ax[0,0].axvline(0, color='black')
    ax[0,1].axvline(0, color='black')
    ax[1,0].axvline(0, color='black')
    ax[1,1].axvline(0, color='black')

    for aks in ax.flatten():
        aks.set_xlabel('Change in donation date in days \n(t=0 is original donation day)')
        aks.set_ylabel('Number of donors invited (cumulative)')

    if save:
        plt.savefig(results_path / f'plots_performance/{save}{args.foldersuffix}.png')
        
def get_invitecats(df_f, df_m, output_path, save=False):
    df_f['sex'] = 'F'
    df_m['sex'] = 'M'
    df_intervals = pd.concat([df_f, df_m]).loc[:, ['sex', 'DaysSinceHb1', 'first_pos_pred', 'HbOK', 'HbOK_pred']]
    df_intervals['DaystoPrevNew'] = df_intervals['DaysSinceHb1'] + df_intervals['first_pos_pred']
    
    df_intervals['invite_category'] = pd.cut(df_intervals['first_pos_pred'], 
                                             bins=[-400, -14, 14, 90, 400],
                                             labels=['More than 2 weeks earlier',
                                                     'Within 2 weeks',
                                                     'Between 2 weeks and 3 months later',
                                                     'Between 3 months and 1 year later']).values.add_categories('Not within 1 year')
    df_intervals.loc[df_intervals['invite_category'].isnull(), 'invite_category'] = 'Not within 1 year'
    
    if save:
        df_intervals.to_pickle(output_path / f'{save}.pkl')
    
    return df_intervals

def print_impact(args, df_intervals, output_path):
    summ_int = df_intervals.groupby(['HbOK', 'invite_category']).count().drop(columns=['DaysSinceHb1','first_pos_pred','HbOK_pred','DaystoPrevNew']).rename(columns={'sex':'count'})
    summ_int = summ_int.reset_index().pivot(index='invite_category', columns='HbOK', values='count').reset_index()
    summ_int = summ_int.rename(columns={0:'Deferred donors (count)', 1:'Non-deferred donors (count)'})
    summ_int['Deferred donors (%)'] = summ_int['Deferred donors (count)'] / summ_int['Deferred donors (count)'].sum() * 100
    summ_int['Non-deferred donors (%)'] = summ_int['Non-deferred donors (count)'] / summ_int['Non-deferred donors (count)'].sum() * 100
    summ_int = summ_int[['invite_category','Deferred donors (count)','Deferred donors (%)','Non-deferred donors (count)','Non-deferred donors (%)']]
    
    summ_int.to_csv(output_path / f'new_invitecats{args.foldersuffix}.csv')
    
    df_ints = df_intervals.dropna()

    median_old_m = np.median(df_ints.loc[df_ints.sex == 'M', 'DaysSinceHb1'])
    median_old_f = np.median(df_ints.loc[df_ints.sex == 'F', 'DaysSinceHb1'])

    median_new_m = np.median(df_ints.loc[df_ints.sex == 'M', 'DaystoPrevNew'])
    median_new_f = np.median(df_ints.loc[df_ints.sex == 'F', 'DaystoPrevNew'])
    
    lines = []
    lines.append(f'Median donation interval for men goes from {median_old_m} to {median_new_m} days.')
    lines.append(f'Median donation interval for women goes from {median_old_f} to {median_new_f} days.')

    visits_relative = (sum(df_ints['DaysSinceHb1'])) / (sum(df_ints['DaystoPrevNew']))
    lines.append(f'Number of blood bank visits increases by {round((visits_relative-1)*100)}%.')
    
    with open(output_path / f'impact{args.foldersuffix}.txt', 'w') as f:
        f.write('\n'.join(lines))
    
def main(args):
    df_f = gather_results('women', range(5, 1, -1))
    df_m = gather_results('men', range(5, 1, -1))
    
    output_path = results_path / f'pred_timechange{args.foldersuffix}'
    output_path.mkdir(parents=True, exist_ok=True)

    df_f = df_f.drop(columns=list(df_f.columns[:10]) + list(df_f.columns[12:20])).reset_index()
    df_f.to_pickle(output_path / 'predictions_women.pkl')

    df_m = df_m.drop(columns=list(df_m.columns[:10]) + list(df_m.columns[12:20])).reset_index()
    df_m.to_pickle(output_path / 'predictions_men.pkl')
    
    plot_newinvites(args, df_f, df_m, save='invites_datechange')
    
    df_intervals = get_invitecats(df_f, df_m, output_path, save='invite_intervals')
    print_impact(args, df_intervals, output_path)
    
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)