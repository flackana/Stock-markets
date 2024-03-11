'''Run this script to prepare features for xgboost model.'''
from statistics import mode
import pandas as pd
import numpy as np
import sys
sys.path.append('./src')
from my_functions.data_cleaning_feature_eng_functions import label_it_up, add_min_max, add_nr_positive, add_moments
from my_functions.data_cleaning_feature_eng_functions import smooth_by_exponential, translate_into_30_min_returns_fast, autocorr_by_equity, autocorr_by_day

derivatives = True
downsample = True
moments = True
group_characteristics = True
remove_daily_der = True
exp_window = 10
rescale = False
log_tr = False
min_max = True
positive = True
labels = True

# Import Data
x_train_cl = pd.read_csv('./Data/Data_clean/x_train_clean.csv')
x_test_cl = pd.read_csv('./Data/Data_clean/x_test_clean.csv')
Y_train_cl = pd.read_csv('./Data/Data_clean/Y_train_clean.csv')
# Create features
df_features_train = pd.DataFrame()
df_features_test = pd.DataFrame()
if rescale:
    avrg_train, var = smooth_by_exponential(x_train_cl, exp_window, [3, 56], include_day_eq=True)
    avrg_test, var = smooth_by_exponential(x_test_cl, exp_window, [3, 56], include_day_eq=True)
    avrg_train.insert(0, 'ID', x_train_cl['ID'])
    avrg_test.insert(0, 'ID', x_test_cl['ID'])
    x_train_cl = avrg_train.copy()
    x_test_cl = avrg_test.copy()

if downsample:
    x_train_features = translate_into_30_min_returns_fast(x_train_cl)
    x_test_features = translate_into_30_min_returns_fast(x_test_cl)
df_returns_train = x_train_features.iloc[:, 2:]
df_returns_test = x_test_features.iloc[:, 2:]

if min_max:
    df_features_train = add_min_max(df_returns_train, df_features_train, [0, 8])
    df_features_test = add_min_max(df_returns_test, df_features_test, [0, 8])
if moments:
    df_features_train = add_moments(df_returns_train, df_features_train, [0, 8])
    df_features_test = add_moments(df_returns_test, df_features_test, [0, 8])
if positive:
    df_features_train = add_nr_positive(df_returns_train, df_features_train, [0, 8])
    df_features_test = add_nr_positive(df_returns_test, df_features_test, [0, 8])
if labels:
    df_features_train = label_it_up(df_returns_train, df_features_train, [0, 8])
    df_features_test = label_it_up(df_returns_test, df_features_test, [0, 8])
# add features calculated as average over day/equity
df_features_train['day'] = x_train_features['day']
df_features_train['equity'] = x_train_features['equity']
df_features_test['day'] = x_test_features['day']
df_features_test['equity'] = x_test_features['equity']

df_features_train2 = df_features_train.copy()
df_features_test2 = df_features_test.copy()
columns = df_features_train.columns[:-2]
for col in columns:
    df_features_train2[col + '_day'] = df_features_train2['day'].map(df_features_train2.groupby('day')[col].mean())
    df_features_test2[col + '_day'] = df_features_test2['day'].map(df_features_test2.groupby('day')[col].mean())
    df_features_train2[col + '_equity'] = df_features_train2['equity'].map(df_features_train2.groupby('equity')[col].mean())
    df_features_test2[col + '_equity'] = df_features_test2['equity'].map(df_features_test2.groupby('equity')[col].mean())

df_features_train2['first_2h_day'] = df_features_train.groupby(['day'])['first_2h'].transform(mode).values
df_features_test2['first_2h_day'] = df_features_test.groupby(['day'])['first_2h'].transform(mode).values

df_features_train2['middle_2h_day'] = df_features_train.groupby(['day'])['middle_2h'].transform(mode).values
df_features_test2['middle_2h_day'] = df_features_test.groupby(['day'])['middle_2h'].transform(mode).values

df_features_train2['last_2h_day'] = df_features_train.groupby(['day'])['last_2h'].transform(mode).values
df_features_test2['last_2h_day'] = df_features_test.groupby(['day'])['last_2h'].transform(mode).values

df_features_train2['first_2h_equity'] = df_features_train.groupby(['equity'])['first_2h'].transform(mode).values
df_features_test2['first_2h_equity'] = df_features_test.groupby(['equity'])['first_2h'].transform(mode).values

df_features_train2['middle_2h_equity'] = df_features_train.groupby(['equity'])['middle_2h'].transform(mode).values
df_features_test2['middle_2h_equity'] = df_features_test.groupby(['equity'])['middle_2h'].transform(mode).values

df_features_train2['last_2h_equity'] = df_features_train.groupby(['equity'])['last_2h'].transform(mode).values
df_features_test2['last_2h_equity'] = df_features_test.groupby(['equity'])['last_2h'].transform(mode).values

# Add autocorr length by equity and by day
df_features_train2 = autocorr_by_day(x_train_cl.iloc[:, 1:], df_features_train2)
df_features_train2 = autocorr_by_equity(x_train_cl.iloc[:, 1:], df_features_train2)

df_features_test2 = autocorr_by_day(x_test_cl.iloc[:, 1:], df_features_test2)
df_features_test2 = autocorr_by_equity(x_test_cl.iloc[:, 1:], df_features_test2)

df_features_train2.drop(['day', 'equity'], axis = 1, inplace = True)
df_features_test2.drop(['day', 'equity'], axis = 1, inplace = True)
# Fill small number of NaN
COL = 'skew'
df_features_train2[COL].fillna(np.mean(df_features_train2[COL]), inplace=True)
df_features_test2[COL].fillna(np.mean(df_features_test2[COL]), inplace=True)
COL = 'kurt'
df_features_train2[COL].fillna(np.mean(df_features_train2[COL]), inplace=True)
df_features_test2[COL].fillna(np.mean(df_features_test2[COL]), inplace=True)

# Save
df_features_train2.to_csv('./Data/Data_after_feature_en/xgb_train_features_renewed.csv',
                          index=False)
df_features_test2.to_csv('./Data/Data_after_feature_en/xgb_test_features_renewed.csv',
                         index=False)

