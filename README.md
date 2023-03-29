# SVM-hb-NL

[![Python Version](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)

This repository contains the code used to obtain the results that are described
in the scientific article "The added value of ferritin levels and genetic markers 
for the prediction of hemoglobin deferral" (to be submitted soon). 
The repository has been indexed on [Zenodo].


## Data 
The data used in this analysis is collected by Finnish Red Cross Blood Service
and Sanquin Blood Supply Foundation, from blood donors who have given permission 
for the use of their data in scientific research. Due to privacy reasons this data 
will not be shared. The /data folder contains a description of the raw data, so 
that researchers with access to similar data may use this code to analyse their own data.

## Scientific abstract
tba

## Installation
To use these files, clone the repository:
`git clone git@github.com:vinkenoogm/hbpred-compare.git`
`cd hbpred-compare`

Create and activate a virtual environment (e.g. `python -m venv venv`) and install the required packages using `pip install -r requirements.txt`. 
This file contains all necessary (Python) packages along with version information. All code was run using Python 3.10.4. With large datasets, 
most scripts are computationally expensive to run and running on a HPC or similar is recommended. 

## Models
For each configuration of data set (country) and predictor variables, five different 
submodels (SVM-1 through SVM-5) are trained separately for men and
women, resulting in ten models total. The number in the model name indicates how
many previous Hb measurements are used in the prediction. As donors can only be
included in SVM-n if they have at least n previous visits, sample sizes decrease
from SVM-1 to SVM-5. The following predictor variables are used:

Variable	 | Unit or values |	Description
-------------|----------------|----------------------------------------------------------------------------------------------
Sex	         | {male, female} |	Biological sex of the donor; separate models are trained for men and women
Age          | years          |	Donor age at time of donation
Time         | hours          |	Registration time when the donor arrived at the blood bank
Month        | {1-12}         |	Month of the year that the visit took place
NumDon       | count          |	Number of successful (collected volume > 250 mL) whole-blood donations in the last 24 months
FerritinPrev | ng/mL          |	Most recent ferritin level measured in this donor
DaysSinceFer | days           |	Time since this donor’s last ferritin measurement
HbPrevn      | mmol/L         |	Hemoglobin level at nth previous visit, for n between 1-5
DaysSinceHbn | days	          | Time since related Hb measurement at nth previous visit, for n between 1-5


## Files
`run_full_analysis.py` calls all other scripts and carries out the entire analysis for any given combination of data and variables. The scripts use the following arguments:

Argument     | Description
-------------|--------------------------------------------------------
nback        | [int] Which model to use (number of previous donations)
sex          | [men/women] Use male or female donors
foldersuffix | [str] Optional foldersuffix to specify a run 

For example, to optimize hyperparameters for SVM-3 for male donors:

```
$ python src/hyperparams.py 3 men
```

To see which arguments are accepted, you can use `--help`:

```
$ python src/hyperparams.py --help
usage: hyperparams.py [-h] [--foldersuffix FOLDERSUFFIX] nback {men,women}

positional arguments:
  nback                 [int] number of previous Hb values to use in prediction
  {men,women}           [men/women] sex to use in model

options:
  -h, --help            show this help message and exit
  --foldersuffix        FOLDERSUFFIX
                        [str] optional suffix indicating non-default run
```

For argument `nback`, values 1 through 5 were used in the analysis in the accompanying paper, but any integer value is accepted.

### preprocessing_netherlands.py
This notebook takes the raw donation data (source files) as collected by Sanquin.
Preprocessing includes
- merging donation files from different years
- selecting relevant donations and variables
- manipulating recorded variables into required predictor variables
- scaling the variables to N(0,1)
- saving train and test data sets
- describing marginal distributions of predictor variables.

### preprocessing_finland.py
This notebook takes the raw donation data (source files) as collected by FRCBS.
Preprocessing includes
- merging donation files from different years
- selecting relevant donations and variables
- manipulating recorded variables into required predictor variables
- scaling the variables to N(0,1)
- saving train and test data sets
- describing marginal distributions of predictor variables.

### hyperparams.py
Run with arguments: 
This script takes as input the scaled train data sets produced in
`preprocessing.py`. Using a grid search with 5-fold cross-validation,
hyperparameters C and gamma are optimized for support vector machines with RBF
kernel. The results of the grid search are saved in `results/{country}/hyperparams/`.

### modeltraining.py
In this script all models are trained using the scaled train data sets and the
optimized hyperparameters. Performance (precision and recall on both outcome
classes) is assessed on both the train and test sets. Trained models and
performance metrics are saved in `results/{country}/models/` (`clf\_{sex}\_{n}.sav` are
trained models files, `res\_{sex}\_{n}.pkl` contain performance metrics).

### modelperformance.py
This notebook reads the `res\_{sex}\_{n}.pkl` files and creates graphs that show
model performance. These plots are also saved in `results/{country}/plots_performance/`.

This script is runnable using intermediate results present on Github to reproduce
the plots.

### calcshap.py
Additional argument:

Argument     | Description
-------------|---------------------------------------------------------------------
n            | [int] Number of randomly selected donors to calculate SHAP values on

This script calculates SHAP values for a random subset of donors from the test
set. Results are not shared because they contain donor-level sensitive information.

### calcshap_subset.py
Additional argument:

Argument     | Description
-------------|---------------------------------------------------------------------
n            | [int] Number of randomly selected donors to calculate SHAP values on

This script calculates SHAP values for a subset of donors from the test
set. The subset is not random, but specifically chosen (e.g. based on different genotypes). Results are not shared because they contain donor-level sensitive information.

### anonshap.py
This script anonymizes SHAP values so that no individual level information is present anymore. 

### plotshap.py
In this notebook, summary plots for the SHAP values calculated by
`calcshap.py` or `anonshap.py` are created and saved in `results/{country}/plots_shap/`.

[Zenodo]: 
