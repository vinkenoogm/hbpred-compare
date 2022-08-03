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
results_path = Path('../results/')

def add_variables(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['successful_don'] = df['Hb_deferral'] == 0
    df = df.sort_values(['vdonor', 'date']).reset_index(drop=True)
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
    df = df.groupby('vdonor').apply(add_numdon_inner)
    df = df.set_index('index')
    return df

def split_train_test(df):
    # Splitting into train and test sets based on date (last year is test set)
    var = ['vdonor', 'date', 'sex', 'year', 'age', 'month', 'NumDon', 'smoking', 'height', 'weight', 'bmi', 
           'snp_17_58358769', 'snp_6_32617727', 'snp_15_45095352', 'snp_1_169549811', 'prs_anemia', 'prs_ferritin',
           'prs_hemoglobin']
    for n in range(1, 6):
        var.extend([f'HbPrev{n}', f'DaysSinceHb{n}'])
    var.append('HbOK')

    df['smoking'] = df['smoking'].astype(int)

    train_men = df.loc[(df.sex == 'Men') & (df.date <= '2019-05-01'), var]
    train_men = train_men[train_men.columns[4:]]
    train_women = df.loc[(df.sex == 'Women') & (df.date <= '2019-05-01'), var]
    train_women = train_women[train_women.columns[4:]]

    test_men = df.loc[(df.sex == 'Men') & (df.date > '2019-05-01'), var]
    test_men = test_men[test_men.columns[4:]]
    test_women = df.loc[(df.sex == 'Women') & (df.date > '2019-05-01'), var]
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
    data = pd.read_pickle(data_path / 'alldata.pkl')
    ## TO USE FAKE TESTING DATA: put fakedata.csv in data folder, comment previous line out
    ## uncomment following line:
    # data = pd.read_csv(data_path / 'fakedata.csv')
    df = add_variables(data)
    
    # We will use donations from >2015, first select 2 more years to calculate donations in last 24 months
    df = df.loc[df.year > 2013, ].copy()    
    # Add NumDon variable, then take donations >2015
    df = add_numdon(df)
    df = df.loc[df.year > 2015, ].copy()

    # Add remaining predictor variables
    df_1 = df.groupby('vdonor').apply(add_prev_hb_time, number=1)
    df_2 = df_1.groupby('vdonor').apply(add_prev_hb_time, number=2)
    df_3 = df_2.groupby('vdonor').apply(add_prev_hb_time, number=3)
    df_4 = df_3.groupby('vdonor').apply(add_prev_hb_time, number=4)
    df_5 = df_4.groupby('vdonor').apply(add_prev_hb_time, number=5)

    df_5.to_pickle(data_path / 'df_2016_2020.pkl')
    df = pd.read_pickle(data_path / 'df_2016_2020.pkl')
    df['HbOK'] = (df['Hb_deferral'] - 1) * -1
    
    train_men, test_men, train_women, test_women = split_train_test(df)

    # Scaled train/test sets for Hb variables + genetic data
    save_scaled_train_test_sets(train_men, test_men, train_women, test_women, 
                                predvars=['age', 'month', 'NumDon', 
                                          'snp_17_58358769', 'snp_6_32617727', 'snp_15_45095352', 
                                          'snp_1_169549811', 'prs_anemia', 'prs_ferritin', 'prs_hemoglobin'], 
                                foldersuffix='')

    # Scaled train/test sets for Hb variables only
    save_scaled_train_test_sets(train_men, test_men, train_women, test_women, 
                                predvars=['age', 'month', 'NumDon'], 
                                foldersuffix='_hbonly')
    
if __name__ == '__main__':
    main()
