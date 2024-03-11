
# ENS Data Challenge 2023: Prediction of daily stock end-of-day movements on the US market
This project is a part of the ENS Data Challenge 2023: https://challengedata.ens.fr/challenges/84

#### -- Project Status: [Active]

## Objective
The goal is to estimate the main direction that will occur during the last two hours of trading session, given the preceding history of the day.

### Methods Used
* XGBoost
* RNN
* Time Series

### Technologies
* Python
* Sikit-Learn
* Tensorflow

## Project Description


## Getting Started

1. Clone this repo
2. Raw Data is being kept in Data/ folder and can be downloaded from: https://challengedata.ens.fr/challenges/84
    
3. Data processing/transformation scripts are being kept in Scripts/ folder

4. Run ``` python3 Scripts/Data_cleaning.py``` to get clean data and save them in Data/Data_clean folder.
5. * Run ```python3 Scripts/feature_engeenering_xgb.py``` to prepare data for xgb model.
   * Run ```python3 Scripts/feature_engeneering_RNN.py``` to prepare data for RNN model.
   
6. * Run XGB model by ```python3 Scripts/xgboost_run.py```.
   * Run RNN model by ```python3 Scripts/RNN_run.py```.

## Notebooks
* In the notebook: Notebooks/Data_exploration.py, I explore some basic properties of the data.


## Author
Ana Flack (flackana)
