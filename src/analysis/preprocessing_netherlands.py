import ast
import datetime
from itertools import product
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Setting paths
data_path_in = Path('../../data')
results_path = Path('../../results/netherlands')

def add_variables(df):
    df['date'] = pd.to_datetime(df['Date'])
    df['DoB'] = pd.to_datetime(df['DoB'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['age'] = (df['date'] - df['DoB']) / pd.Timedelta('365.25d')
    df['successful_don'] = df['Volume'] > 250
    df = df.sort_values(['KeyID', 'date']).reset_index(drop=True)
    return df

def add_timefirstdon(df):
    df['date_of_first_donation'] = df.loc[df.DonType == 'N', 'date']
    df['date_of_first_donation'] = df['date_of_first_donation'].ffill()
    df['DaysSinceFirstDon'] = (df['date'] - df['date_of_first_donation']) / pd.Timedelta('1d')
    return df

def add_prev_hb_time(df, number):
    colnames = [f'HbPrev{str(number)}', f'DaysSinceHb{str(number)}']
    df[colnames[0]] = df['Hb'].shift(number)
    df[colnames[1]] = (df['date'] - df['date'].shift(number)) / pd.Timedelta('1 day')
    return df

def add_numdon_inner(df):
    df['NumDon'] = df['successful_don'].rolling('730d', closed='left').sum()
    return df

def add_numdon(df):
    df['index'] = df.index
    df = df.set_index('date', drop=False)
    df = df.groupby('KeyID').apply(add_numdon_inner)
    df = df.set_index('index')
    return df

def add_last_ferritin(df):
    fers = df.loc[df['Ferritin'].notnull(), ['date', 'Ferritin']]
    if fers.shape[0] == 0:
        df['FerritinPrev'] = np.NaN
        df['Last_Fer_Date'] = np.NaN
    else:
        df = pd.merge_asof(df, fers, left_index=True, right_index=True, allow_exact_matches=False, direction='backward', suffixes=['','_fer'])
        df = df.rename(columns={'Ferritin_fer':'FerritinPrev',
                                'date_fer':'Last_Fer_Date'})
    return df

def split_train_test(df):
    # Splitting into train and test sets based on date (last year is test set)
    var = ['KeyID', 'date', 'Sex', 'year', 'age', 'month', 'NumDon', 'FerritinPrev', 'DaysSinceFer', 'DaysSinceFirstDon']
    for n in range(1, 6):
        var.extend([f'HbPrev{n}', f'DaysSinceHb{n}'])
    var.append('HbOK')

    train_men = df.loc[(df.Sex == 'M') & (df.year <= 2020), var]
    train_men = train_men[train_men.columns[4:]]
    train_women = df.loc[(df.Sex == 'F') & (df.year <= 2020), var]
    train_women = train_women[train_women.columns[4:]]

    test_men = df.loc[(df.Sex == 'M') & (df.year > 2020), var]
    test_men = test_men[test_men.columns[4:]]
    test_women = df.loc[(df.Sex == 'F') & (df.year > 2020), var]
    test_women = test_women[test_women.columns[4:]]
    
    return train_men, test_men, train_women, test_women
    
def save_subsets(train_men, test_men, train_women, test_women,
                 predvars, save_path_dfs, save_path_scalers, foldersuffix=''):  
    for nback in range(1, 6):
        selvars = predvars.copy()
        for n in range(1, nback+1):
            selvars.extend([f'HbPrev{n}', f'DaysSinceHb{n}'])
        selvars.append('HbOK')

        unscaled_folder = save_path_dfs / f'unscaled{foldersuffix}'
        unscaled_folder.mkdir(parents=True, exist_ok=True)
        scaled_folder = save_path_dfs / f'scaled{foldersuffix}'
        scaled_folder.mkdir(parents=True, exist_ok=True)
        scaler_folder = save_path_scalers / f'scalers{foldersuffix}'
        scaler_folder.mkdir(parents=True, exist_ok=True)
        
        # Select subset
        train_m_sub = train_men[selvars].dropna()
        test_m_sub = test_men[selvars].dropna()
        train_f_sub = train_women[selvars].dropna()
        test_f_sub = test_women[selvars].dropna()
         
        # Save unscaled subdfs
        train_m_sub.to_pickle(unscaled_folder / f'men_{nback}_train.pkl')
        test_m_sub.to_pickle(unscaled_folder / f'men_{nback}_test.pkl')
        train_f_sub.to_pickle(unscaled_folder / f'women_{nback}_train.pkl')
        test_f_sub.to_pickle(unscaled_folder / f'women_{nback}_test.pkl')
        
        # Make&save scalers and scale subdfs
        scaler_men = StandardScaler()
        scaler_women = StandardScaler()
        scaler_men.fit(train_m_sub[train_m_sub.columns[:-1]])
        scaler_women.fit(train_f_sub[train_f_sub.columns[:-1]])
        pickle.dump(scaler_men, open(scaler_folder / f'men_{nback}.pkl', 'wb'))
        pickle.dump(scaler_women, open(scaler_folder / f'women_{nback}.pkl', 'wb'))
        
        train_m_sub[train_m_sub.columns[:-1]] = scaler_men.transform(train_m_sub[train_m_sub.columns[:-1]])
        test_m_sub[test_m_sub.columns[:-1]] = scaler_men.transform(test_m_sub[test_m_sub.columns[:-1]])
        train_f_sub[train_f_sub.columns[:-1]] = scaler_women.transform(train_f_sub[train_f_sub.columns[:-1]])
        test_f_sub[test_f_sub.columns[:-1]] = scaler_women.transform(test_f_sub[test_f_sub.columns[:-1]])
    
        # Save scaled subdfs 
        train_m_sub.to_pickle(scaled_folder / f'men_{nback}_train.pkl')
        test_m_sub.to_pickle(scaled_folder / f'men_{nback}_test.pkl')
        train_f_sub.to_pickle(scaled_folder / f'women_{nback}_train.pkl')
        test_f_sub.to_pickle(scaled_folder / f'women_{nback}_test.pkl')

def main():
    ## TO USE FAKE TESTING DATA: put fakedata.csv in data folder,
    ## uncomment following line:
    # data = pd.read_csv(data_path_in / 'fakedata_netherlands.csv')
    
    data = pd.read_pickle(data_path_in / 'data_clean.pkl')
    df = add_variables(data)
    df = df.groupby('KeyID').apply(add_timefirstdon)
    print('add_variables done', datetime.datetime.now())
    
    # We will use donations from >2017, first select 2 more years to calculate donations in last 24 months
    df = df.loc[df.year > 2014, ].copy()    
    # Add NumDon variable, then take donations >2015
    df = add_numdon(df)
    print('add_numdon done', datetime.datetime.now())
    df = df.loc[df.year > 2016, ].copy()

    # Add remaining predictor variables
    df_1 = df.groupby('KeyID').apply(add_prev_hb_time, number=1)
    print('hb1 done', datetime.datetime.now())
    df_2 = df_1.groupby('KeyID').apply(add_prev_hb_time, number=2)
    print('hb2 done', datetime.datetime.now())
    df_3 = df_2.groupby('KeyID').apply(add_prev_hb_time, number=3)
    print('hb3 done', datetime.datetime.now())
    df_4 = df_3.groupby('KeyID').apply(add_prev_hb_time, number=4)
    print('hb4 done', datetime.datetime.now())
    df_5 = df_4.groupby('KeyID').apply(add_prev_hb_time, number=5)
    print('hb5 done', datetime.datetime.now())

    df_5f = df_5.groupby('KeyID').apply(add_last_ferritin)
    print('ferr done', datetime.datetime.now())
    df_5f.to_pickle(data_path_in / 'df_5f.pkl')
    
    df_5f = pd.read_pickle(data_path_in / 'df_5f.pkl')  
    df_5f['DaysSinceFer'] = (df_5f['date'] - df_5f['Last_Fer_Date']) / pd.Timedelta('1d')
    # df = df_5f.loc[~df_5f.FerritinPrev.isna(), ].copy()
    
    df = df_5f.dropna(subset=['age', 'month', 'NumDon', 'FerritinPrev', 'DaysSinceFirstDon']).copy()
    
    train_men, test_men, train_women, test_women = split_train_test(df)
    
    # Save unscaled full dataframes
    train_men.to_pickle(data_path_in / 'train_men.pkl')
    test_men.to_pickle(data_path_in / 'test_men.pkl')
    train_women.to_pickle(data_path_in / 'train_women.pkl')
    test_women.to_pickle(data_path_in / 'test_women.pkl')

    train_men = pd.read_pickle(data_path_in / 'train_men.pkl')
    test_men = pd.read_pickle(data_path_in / 'test_men.pkl')
    train_women = pd.read_pickle(data_path_in / 'train_women.pkl')
    test_women = pd.read_pickle(data_path_in / 'test_women.pkl')
    
    save_subsets(train_men, test_men, train_women, test_women, 
                 predvars=['age', 'month', 'NumDon', 'FerritinPrev'], 
                 save_path_dfs=data_path_in / 'NL', 
                 save_path_scalers=results_path / 'netherlands',
                 foldersuffix='')
    save_subsets(train_men, test_men, train_women, test_women,
                 predvars=['age', 'month', 'NumDon'], 
                 save_path_dfs=data_path_in / 'NL', 
                 save_path_scalers=results_path / 'netherlands', 
                 foldersuffix='_hbonly')
    
if __name__ == '__main__':
    main()
