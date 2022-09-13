import ast
import datetime
from itertools import product
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Setting paths
data_path = Path('../../data/')
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
    var = ['KeyID', 'date', 'Sex', 'year', 'age', 'month', 'NumDon', 'FerritinPrev', 'DaysSinceFer']
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

def save_scaled_train_test_sets(train_men, test_men, train_women, test_women, predvars, foldersuffix=''):  
    for nback in range(1, 6):
        selvars = predvars.copy()
        for n in range(1, nback+1):
            selvars.extend([f'HbPrev{n}', f'DaysSinceHb{n}'])
        selvars.append('HbOK')

        scalers_folder = results_path / f'scalers{foldersuffix}'
        scalers_folder.mkdir(parents=True, exist_ok=True)  # create folder if it does not yet exist
        scaled_folder = data_path / f'scaled{foldersuffix}'
        scaled_folder.mkdir(parents=True, exist_ok=True)   # create folder if it does not yet exist
        
        train_men_sub = train_men[selvars].dropna()
        train_women_sub = train_women[selvars].dropna()
        test_men_sub = test_men[selvars].dropna()
        test_women_sub = test_women[selvars].dropna()

        scaler_men = StandardScaler()
        scaler_women = StandardScaler()
        scaler_men.fit(train_men_sub[train_men_sub.columns[:-1]])
        scaler_women.fit(train_women_sub[train_men_sub.columns[:-1]])

        train_men_sub[train_men_sub.columns[:-1]] = scaler_men.transform(train_men_sub[train_men_sub.columns[:-1]])
        train_women_sub[train_women_sub.columns[:-1]] = scaler_women.transform(train_women_sub[train_women_sub.columns[:-1]])
        test_men_sub[test_men_sub.columns[:-1]] = scaler_men.transform(test_men_sub[test_men_sub.columns[:-1]])
        test_women_sub[test_women_sub.columns[:-1]] = scaler_women.transform(test_women_sub[test_women_sub.columns[:-1]])

        pickle.dump(scaler_men, open(scalers_folder / f'men_{nback}.pkl', 'wb'))
        pickle.dump(scaler_women, open(scalers_folder / f'women_{nback}.pkl', 'wb'))

        train_men_sub.to_pickle(scaled_folder / f'men_{nback}_train.pkl')
        train_women_sub.to_pickle(scaled_folder / f'women_{nback}_train.pkl')
        test_men_sub.to_pickle(scaled_folder / f'men_{nback}_test.pkl')
        test_women_sub.to_pickle(scaled_folder / f'women_{nback}_test.pkl')

        
def main():
    # data = pd.read_pickle(data_path / 'data_clean.pkl')
    ## TO USE FAKE TESTING DATA: put fakedata.csv in data folder, comment previous line out
    ## uncomment following line:
    data = pd.read_csv(data_path / 'fakedata_netherlands.csv')
    df = add_variables(data)
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
    df_5f.to_pickle('df_5f.pkl')
    df_5f = pd.read_pickle('df_5f.pkl')
    df_5f['DaysSinceFer'] = (df_5f['date'] - df_5f['Last_Fer_Date']) / pd.Timedelta('1d')

    df = df_5f.loc[~df_5f.FerritinPrev.isna(), ].copy()
    
    train_men, test_men, train_women, test_women = split_train_test(df)
    train_men.to_pickle('train_men.pkl')
    test_men.to_pickle('test_men.pkl')
    print('split done')

    # Scaled train/test sets for Hb variables + genetic data
    save_scaled_train_test_sets(train_men, test_men, train_women, test_women, 
                                predvars=['age', 'month', 'NumDon', 'FerritinPrev', 'DaysSinceFer'], 
                                foldersuffix='')
    print('all vars done')

    # Scaled train/test sets for Hb variables only
    save_scaled_train_test_sets(train_men, test_men, train_women, test_women, 
                                predvars=['age', 'month', 'NumDon'], 
                                foldersuffix='_hbonly')
    print('onlyhb vars done')
    
if __name__ == '__main__':
    main()
