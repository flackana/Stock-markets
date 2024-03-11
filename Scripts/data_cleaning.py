'''Run this script to remove NaN from raw data'''
import pandas as pd
import sys
sys.path.append('./src')
from my_functions.data_cleaning_feature_eng_functions import fill_nan


x_train = pd.read_csv('./Data/input_training.csv')
x_test = pd.read_csv('./Data/input_test.csv')
Y_train = pd.read_csv('./Data/output_training_gmEd6Zt.csv')
all_train = pd.merge(x_train, Y_train, on='ID', how='inner')
all_train['reod'] +=1
# Fill NaN
strategy = 1
all_train_1 = fill_nan(all_train, strategy)
all_train_1.fillna(0, inplace=True)
x_train_1 = all_train_1.iloc[:,:-1]
Y_train_1 = all_train_1.iloc[:,-1]

x_test_1 = fill_nan(x_test, strategy)
x_test_1.fillna(0, inplace=True)

x_train_1.to_csv('./Data/Data_clean/x_train_clean.csv', index=False)
Y_train_1.to_csv('./Data/Data_clean/Y_train_clean.csv', index = False)
x_test_1.to_csv('./Data/Data_clean/x_test_clean.csv', index = False)
