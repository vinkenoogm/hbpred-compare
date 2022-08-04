import preprocessing
import hyperparams
import modeltraining
import modelperformance
import calcshap
import plotshap
import changingtime
import impactbloodsupply

from argparse import Namespace
from datetime import datetime
from itertools import product
from multiprocessing import Pool

def main():
    firststart = datetime.now().replace(microsecond=0)
    
    options = {'nback': [1,2,3,4,5], 
               'sex': ['men','women'], 
               'foldersuffix': ['', '_hbonly']}
    combs = [x for x in product(*options.values())]
    list_args = [Namespace(**dict(zip(options.keys(), p))) for p in combs]
    
    options_shap = {'nback': [1,2,3,4,5], 
                   'sex': ['men','women'], 
                   'n': [100],
                   'foldersuffix': ['', '_hbonly']}
    combs_shap = [x for x in product(*options_shap.values())]
    list_args_shap = [Namespace(**dict(zip(options_shap.keys(), p))) for p in combs_shap]    
    
    options_folderonly = {'foldersuffix': ['', '_hbonly']}
    combs_folderonly = [x for x in product(*options_folderonly.values())]
    list_args_folderonly = [Namespace(**dict(zip(options_folderonly.keys(), p))) for p in combs_folderonly]    
    
    if preprocess:
        # Preprocessing
        print(f'Preprocessing data...')
        
        preprocessing.main()
        
        now = datetime.now().replace(microsecond=0)
        print(f'Done preprocessing. \n--Time elapsed: {now - firststart} \n--Total time elapsed: {now - firststart}\n')
        
    if hyperparam_tuning:
        # Tuning hyperparameters
        start = datetime.now().replace(microsecond=0)
        print(f'Tuning hyperparameters for {len(list_args)} different models. This may take a while. \n--Started at: {firststart}')

        with Pool(len(list_args)) as pool:
            results = pool.map(hyperparams.main, list_args)

        now = datetime.now().replace(microsecond=0)
        print(f'All sets of hyperparameters are tuned and saved. \n--Time elapsed: {now - start} \n--Total time elapsed: {now - firststart}\n')
    
    if model_training:
        # Training models
        start = datetime.now().replace(microsecond=0)
        print(f'Training and saving all {len(list_args)} models. \n--Started at: {start}')

        with Pool(len(list_args)) as pool:
            results = pool.map(modeltraining.main, list_args)

        now = datetime.now().replace(microsecond=0)
        print(f'All models trained and saved. \n--Time elapsed: {now - start} \n--Total time elapsed: {now - firststart}\n')

    if model_performance:
        # Calculating model performance
        start = datetime.now().replace(microsecond=0)
        print(f'Calculating model performance and saving performance plots. \n--Started at: {start}')
        
        with Pool(len(list_args_folderonly)) as pool:
            results = pool.map(modelperformance.main, list_args_folderonly)
            
        now = datetime.now().replace(microsecond=0)
        print(f'Model performance metrics and plots saved. \n--Time elapsed: {now - start} \n--Total time elapsed: {now - firststart}\n')
    
    if shap_values:
        # Calculating SHAP values
        start = datetime.now().replace(microsecond=0)
        print(f'Calculating SHAP values for all models. This will take a few hours. \n--Started at: {start}')

        with Pool(len(list_args_shap)) as pool:
            results = pool.map(calcshap.main, list_args_shap)

        now = datetime.now().replace(microsecond=0)
        print(f'SHAP values for all models are calculated and saved. \n--Time elapsed: {now - start} \n--Total time elapsed: {now - firststart}\n')
    
    if shap_plots:
        # Creating SHAP value plots
        start = datetime.now().replace(microsecond=0)
        print(f'Making plots for SHAP values. \n--Started at: {start}')

        with Pool(len(list_args_shap)) as pool:
            results = pool.map(plotshap.main, list_args_shap)

        now = datetime.now().replace(microsecond=0)
        print(f'All SHAP plots saved. \n--Time elapsed: {now - start} \n--Total time elapsed: {now - firststart}\n')

    if changing_time:
        # Predictions at different timepoints
        start = datetime.now().replace(microsecond=0)
        print(f'Predicting deferral at different timepoints. \n--Started at: {start}')

        with Pool(len(list_args)) as pool:
            results = pool.map(changingtime.main, list_args)

        now = datetime.now().replace(microsecond=0)
        print(f'All predictions saved. \n--Time elapsed: {now - start} \n--Total time elapsed: {now - firststart}\n')
        
    if impact_bloodsupply:
        # Calculate impact on blood supply from predictions at different timepoints
        start = datetime.now().replace(microsecond=0)
        print(f'Calculating impact on blood supply. \n--Started at: {start}')

        with Pool(len(list_args_folderonly)) as pool:
            results = pool.map(impactbloodsupply.main, list_args_folderonly)

        now = datetime.now().replace(microsecond=0)
        print(f'Results saved. \n--Time elapsed: {now - start} \n--Total time elapsed: {now - firststart}\n')

if __name__ == '__main__':
    preprocess = True
    hyperparam_tuning = True
    model_training = True
    model_performance = True
    shap_values = True
    shap_plots = True
    changing_time = True
    impact_bloodsupply = True
    main()

