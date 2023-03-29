# Data

The data used in this analysis is collected by Finnish Red Cross Blood Service and Sanquin Blood Supply Foundation, from donors who have given permission for the use of their data in scientific research. Due to privacy reasons this data will not be shared. This file describes the data so that researchers with access to similar data may use this code to analyse their own data, or to simulate data.

## Before pre-processing (raw source files)

Data is extracted from database system eProgesa for donations between 2008 and 2021. The following variables are required for each blood bank visit:

Variable             | Original (Dutch) variable name | Type    | Description
---------------------|--------------------------------|---------|---------------------------------------------------------------------------
KeyID                | KeyID                          | [int]   | Unique ID number per donor
Research permission  | WOtoestemming                  | [str]   | Either 'ja' (yes) or 'nee' (no); only donors with permission are included
Sex                  | Geslacht                       | [str]   | Either 'M' for male or 'F' for female
Date of birth        | Geboortedatum                  | [str]   | Formatted as (m)m/(d)d/yyyy
EIN number           | Einnummer                      | [str]   | Unique ID number per donation
Donation date        | Donatiedatum                   | [str]   | Formatted as (m)m/(d)d/yyyy
Donation start time  | Donatie_Tijd_Start             | [str]   | Formatted as hh:mm
Donation center code | Donatiecentrumcode             | [str]   | ID for donation center
Donation type code   | Donatiesoortcode               | [str]   | 'V' for whole-blood and 'N' for new donor intake, other types are excluded
Volume               | AfgenomenVolume                | [int]   | Volume taken during donation in mL
Hb                   | Hb                             | [float] | Hb as measured before donation in mmol/L
HbOK                 | HbGoedgekeurd                  | [bool]  | 1 if Hb > threshold, else 0
Ferritin             | Ferritine                      | [int]   | optional: ferritin as measured in donated blood in ng/mL

## After pre-processing

The following variables are created/calculated from the raw data and used as predictor variables:

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