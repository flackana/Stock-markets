''' In this notebook we explore some properties of the data.'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('../src')
from my_functions.data_cleaning_feature_eng_functions import moving_average
import scipy.stats as stats
from my_functions.data_cleaning_feature_eng_functions import follow_one_equity, follow_one_equity_smoothed

#%% Load the data
x_train = pd.read_csv('./../Data/input_training.csv')
x_test = pd.read_csv('./../Data/input_test.csv')
Y_train = pd.read_csv('./../Data/output_training_gmEd6Zt.csv')
x_train.head()
columns = x_train.columns
# %%
# X train dataset
print('There are '+str(x_train['day'].nunique())+' unique days in x_train dataset.')
print('There are '+str(x_train['equity'].nunique())+' unique equity IDs in x_train dataset.')
grouped_by_day = x_train.groupby('day')
counts_per_day = grouped_by_day['equity'].count().to_frame()
counts_per_day['nr_equity_unique'] = grouped_by_day['equity'].nunique().values
ax = sns.relplot(data=counts_per_day, x='day', y='equity').set(title='Number of equities per day')
plt.savefig("../Figures/Nr_equities_per_day_x_train.png")
print('Number of equities goes from '+str(np.min(counts_per_day['equity']))+' to '+str(np.max(counts_per_day['equity']))+ ' with an average of '+str(np.mean(counts_per_day['equity'])))
if sum(counts_per_day['equity']-counts_per_day['nr_equity_unique'])== 0:
    print('No equities ID appears twice per day')
else:
    print('Same equity ID appearandom_plot(x_train.iloc[:, 3:], Y_train_cl)rs twice or more per day!')
print('In '+str(x_train.isna().any(axis=1).sum())+' rows there is at least one NaN value ('+str(x_train.isna().any(axis=1).sum()/len(x_train))+'%).')
fig, ax = plt.subplots() 
ax.hist(x_train.isna().sum(axis=1).values, bins=50)
ax.set_xlabel('Nr of NaN per row')
ax.set_ylabel('Frequency')
fig.savefig('../Figures/Number_of_NaN_per_row')
#
all_train = pd.merge(x_train, Y_train, on='ID', how='inner')

# %% Choose one equity and show trading for all days
x_train_cl = pd.read_csv('./../Data/Data_clean/x_train_clean.csv')
x_test_cl = pd.read_csv('./../Data/Data_clean/x_test_clean.csv')
Y_train_cl = pd.read_csv('./../Data/Data_clean/Y_train_clean.csv')
print('As an example we look at the values of equity 1465, and plot two figures')
equity_1465 = follow_one_equity(x_train_cl, 1465)
# %% Checking moving averages
ma = moving_average(x_train_cl, 2, [2, 56], 5)
# %% Plot random examples of all three classes
print('-1')
a=moving_average(x_train_cl, 2, [3, 55], 5)
plt.show()
print('0')
b=moving_average(x_train_cl, 34, [3, 55], 5)
plt.show()
print('1')
c=moving_average(x_train_cl, 26, [3, 55], 5)
# %% Plot average r for all r plus the variance.
ploting_df = pd.DataFrame()
ploting_df['period'] =  x_train_cl.columns[7:56]
ploting_df['avrg'] = x_train_cl.iloc[:, 7:56].mean(axis=0).values
ploting_df['vars'] = x_train_cl.iloc[:, 7:56].var(axis=0).values
fig, ax0 = plt.subplots(nrows=1)
ax0.errorbar(range(0, len(ploting_df)), ploting_df['avrg'].values, yerr = ploting_df['vars'].values, fmt='-o')
ax0.set_title('Average r with variance')
ax0.set_xlabel('r_i')
ax0.set_ylabel('Mean(r_i) over all entries')
fig.savefig('../Figures/Average_r_with_variance')
# Important - variance of first 4 points is huge, maybe better to cut them off?
# %% Randomly plot
def random_plot(df, labels):
    """Helper function to visualise data.
    plots returns for the random equity, day
    from a given dataset.

    Args:
        df (Dataframe): Df with rows that describe
        returns for a day, equity.
        labels (DataFrame): DF with labels (0, 1, 2) what we want to predict.
        This will be in the title of the plot.

    Returns:
        _Int: 0"""
    fig, ax0 = plt.subplots(nrows=1)
    nr = np.random.randint(0, len(df))
    ax0.plot(df.iloc[nr, :], marker='.')
    ax0.set_title('Class. '+ str(labels.values.ravel()[nr]))
    return 0
#%%
#######################################################
# Do equities in the same day, have similar predictions?
#########################################################
x_train = pd.read_csv('../Data/input_training.csv')
x_test = pd.read_csv('../Data/input_test.csv')
Y_train = pd.read_csv('../Data/output_training_gmEd6Zt.csv')
all_train = pd.merge(x_train, Y_train, on='ID', how='inner')
all_train['reod'] +=1
# %% 
class_count = all_train['reod'].value_counts().values
print("In the whole training set the ratios of classes is")
print(" Class -1: " + str(class_count[0]/class_count.sum()))
print("Class 0: " + str(class_count[1]/class_count.sum()))
print("Class 1: " + str(class_count[2]/class_count.sum()))
def class_ratios_per_day(df, dayy):
    day_df = df[df['day']==dayy]
    all_in_a_day = len(day_df)
    a = day_df['reod'].value_counts()
    if len(a) == 3:
        #print("In the day "+str(dayy)+" the ratios are: ")
        r1 = a[0]/all_in_a_day
        r2 = a[1]/(all_in_a_day)
        r3 = a[2]/all_in_a_day
        #print(" Class -1: " + str(r1))
        #print("Class 0: " +  str(r2))
        #print("Class 1: " +  str(r3))
    else:
        r1 = 0
        r2 = 0
        r3 = 0
    return r1, r2, r3
grouped_by_day = all_train.groupby('day')
a = grouped_by_day['reod'].value_counts()
ratios = np.zeros((502, 3))
for i in range(0, 502):
    r1, r2, r3 = class_ratios_per_day(all_train, i)
    ratios[i, 0] = r1
    ratios[i, 1] = r2
    ratios[i, 2] = r3
#%%
plt.plot(ratios[:, 0], '.', label='Class -1')
plt.plot(ratios[:, 1], 'x', label='Class 0')
plt.plot(ratios[:, 2], '.',label='Class 1')
plt.xlabel('day')
plt.ylabel('Ratio')
plt.legend()
"""In conclusion, the full data set is slightly
unbalanced with 0.41 percent of class 0. However,
if we look o day-to-day basis we see that in some
days the ratios are very different. This means that
we should take into account day as important factor."""
#%%######################################################
# Does same equity on different days have similar predictions?
#########################################################
def class_ratios_per_equity(df, eq):
    eq_df = df[df['equity'] == eq]
    all_in_a_eq = len(eq_df)
    a = eq_df['reod'].value_counts()
    if len(a) == 3:
        #print("In the day "+str(dayy)+" the ratios are: ")
        r1 = a[0]/all_in_a_eq
        r2 = a[1]/(all_in_a_eq)
        r3 = a[2]/all_in_a_eq
        #print(" Class -1: " + str(r1))
        #print("Class 0: " +  str(r2))
        #print("Class 1: " +  str(r3))
    else:
        r1 = 0
        r2 = 0
        r3 = 0
    return r1, r2, r3
grouped_by_eq = all_train.groupby('equity')
a = grouped_by_eq['reod'].value_counts()
#%%
ratios = np.zeros((1828, 3))
for i in range(0, 1828):
    r1, r2, r3 = class_ratios_per_equity(all_train, i)
    ratios[i, 0] = r1
    ratios[i, 1] = r2
    ratios[i, 2] = r3

plt.plot(ratios[:, 0], '.', label='Class -1')
plt.plot(ratios[:, 1], 'x', label='Class 0')
plt.plot(ratios[:, 2], '.',label='Class 1')
plt.xlabel('equity')
plt.ylabel('Ratio')
plt.legend()
# %%
