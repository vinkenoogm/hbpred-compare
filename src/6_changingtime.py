import numpy as np
import pandas as pd 
import datetime
import pickle
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

sex = sys.argv[1]
nback = int(sys.argv[2])
foldersuffix = sys.argv[3]
print(sex, nback, foldersuffix)

def make_preds(data, clf):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]
    y_pred = clf.predict(X)
    return(y_pred)

data = pd.read_pickle('../../data/scaled'+foldersuffix+'/'+str(sex)+'_'+str(nback)+'_test.pkl')
clf = pickle.load(open('../results/models'+foldersuffix+'/clf_' + str(nback) + '.sav', 'rb'))
clf_s = clf[0] if sex == 'men' else clf[1]
scaler = pickle.load(open('../results/scalers'+foldersuffix+'/'+sex+'_'+str(nback)+'.pkl', 'rb'))

print('loaded data')

data_res = data.copy()
y_pred_first = make_preds(data, clf_s)
data_res['Hbdef_pred'] = y_pred_first

timecols = []
for n in range(1, nback+1):
    timecols.extend(['DaystoPrev'+str(n)])

for timestep in range(-364, 371, 7):
    print(timestep)
    data_timestep = data.copy()
    data_timestep[data_timestep.columns[:-1]] = scaler.inverse_transform(data_timestep[data_timestep.columns[:-1]])
    data_timestep[timecols] = data_timestep[timecols].add(timestep)
    data_timestep['month'] = (data_timestep['month'] - 1 + round(timestep/30)) % 12 + 1
    data_timestep['age'] = data_timestep['age'] + (timestep / 365)
    data_timestep[data_timestep.columns[:-1]] = scaler.transform(data_timestep[data_timestep.columns[:-1]])

    y_pred = make_preds(data_timestep, clf_s)

    varname = 'Hbdef_pred_' + str(timestep)
    data_res[varname] = y_pred
    if timestep == 0:
        data_res = data_res.copy()

print(data_res.head())
path = '../../data/pred_timechange'+foldersuffix+'/'        
data_res.to_pickle(path + 'data_res_' + sex + '_' + str(nback) + '.pkl')