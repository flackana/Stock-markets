'''Run this to prepare features for RNN model.'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('./src')
from my_functions.data_cleaning_feature_eng_functions import add_derivatives, translate_into_40_min_returns_fast, add_variance_and_mean, handling_outliers_new, handling_outliers_test_new
from my_functions.data_cleaning_feature_eng_functions import smooth_by_exponential, log_returns, add_same_day_properties, add_same_equity_properties, print_locations_nan, translate_into_30_min_returns_fast
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer

derivatives = True
downsample = False
moments = True
group_characteristics = True
remove_daily_der = True
exp_window = 10
exp_smooth = True
log_tr = False
outliers = True
# Import the data
x_train_cl = pd.read_csv('./Data/Data_clean/x_train_clean.csv')
x_test_cl = pd.read_csv('./Data/Data_clean/x_test_clean.csv')
Y_train_cl = pd.read_csv('./Data/Data_clean/Y_train_clean.csv')

if downsample:
    x_train_features = translate_into_30_min_returns_fast(x_train_cl)
    x_test_features = translate_into_30_min_returns_fast(x_test_cl)
x_train_features = x_train_cl.copy().iloc[:, 1:]
x_test_features = x_test_cl.copy().iloc[:, 1:]
# Remove outliers
if outliers:
    means = np.mean(x_train_features.iloc[:, 2:], axis=1)
    varss = np.var(x_train_features.iloc[:, 2:], axis=1)
    x_train_mid = handling_outliers_new(x_train_features.iloc[:, 2:], 3, 1)
    x_train_features.update(x_train_mid)
    x_test_mid = handling_outliers_new(x_test_features.iloc[:, 2:], 3, 1)
    x_test_features.update(x_test_mid)
# Do exponential smoothing
if exp_smooth:
    avrg_train, var_train = smooth_by_exponential(x_train_features, exp_window, [2, 54], include_day_eq=True)
    avrg_test, var_test = smooth_by_exponential(x_test_features, exp_window, [2, 54], include_day_eq=True)
    var_train['r0'] = var_train['r1']
    var_train.insert(54, 'r52', var_train['r51'])
    var_test['r0'] = var_test['r1']
    var_test.insert(54, 'r52', var_test['r51'])
    avrg_train.insert(54, 'r52', avrg_test['r51'])
    avrg_test.insert(54, 'r52', avrg_test['r51'])

raw_series_train = np.zeros((len(x_train_mid), 53, 3))
raw_series_train[:, :, 0] = np.array(x_train_mid)
raw_series_train[:, :, 1] = np.array(avrg_train.iloc[:, 2:])
raw_series_train[:, :, 2] = np.array(var_train.iloc[:, 2:])

raw_series_test = np.zeros((len(x_test_mid), 53, 3))
raw_series_test[:, :, 0] = np.array(x_test_mid)
raw_series_test[:, :, 1] = np.array(avrg_test.iloc[:, 2:])
raw_series_test[:, :, 2] = np.array(var_test.iloc[:, 2:])
# Scaling to -1, 1: 2(X — X_min) / (X_max — X_min) -1,
mins = np.min(raw_series_train, axis= 1)
maks = np.max(raw_series_train, axis= 1)

mins2 = np.min(raw_series_test, axis= 1)
maks2 = np.max(raw_series_test, axis= 1)
for i in range(3):
    raw_series_train[:, :, i] = (2*(raw_series_train[:, :, i] -
                                     mins[:, i].reshape(-1, 1))/(maks[:, i]-mins[:, i]).reshape(-1, 1))-1

    raw_series_test[:, :, i] = (2*(raw_series_test[:, :, i] - 
                                   mins2[:, i].reshape(-1, 1))/(maks2[:, i]-mins2[:, i]).reshape(-1, 1))-1
# Fill the NaN with 0
raw_series_train = np.nan_to_num(raw_series_train)
raw_series_test = np.nan_to_num(raw_series_test)

# Save
encoder = OneHotEncoder(sparse=False)
Y_train_encoded = encoder.fit_transform(Y_train_cl)

np.save('./Data/Data_after_feature_en/x_train_features_RNN', raw_series_train)
np.save('./Data/Data_after_feature_en/x_test_features_RNN', raw_series_test)
np.save('./Data/Data_after_feature_en/y_train_features_RNN', Y_train_encoded)