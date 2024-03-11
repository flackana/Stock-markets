import random 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import zscore

def drop_rows_with_nan(data_frame, treshold):
    """This function will drop all rows of dataframe where number 
    of NaN values is greater than treshold.

    Args:
        data_frame (DataFrame): input data frame (merge x_train and target)
        treshold (Int): Treshold for number of NaNs in a row.

    Returns:
        DataFrame: Data frame with some dropped rows.
    """
    my_data = data_frame.copy()
    my_data['Nans'] = my_data.iloc[:, 3:].isna().values.sum(axis=1)
    ss = 0
    for i in my_data['Nans'].values:
        if i > treshold:
            ss += 1
    print('Fraction of rows that will be dropped: ' +str(round(ss/len(my_data), 2)))
    #my_data.drop(my_data[my_data['Nans'] > treshold].index, inplace = True)
    my_data = my_data[my_data.Nans > treshold]
    return my_data.drop(['Nans'], axis = 1, inplace = True)

def fill_nan(data_frame, option):
    """This function removes all NaN 
    values from the given data frame.

    Args:
        data_frame (Data Frame): initial data frame
        option (Int): It can be: 1, 2, 3. Each number 
        corresponds to different startegy of removing NaNs.

    Returns:
        DataFrame: Data Frame with no NaN values
    """
    if option==1:
        return data_frame.fillna(method='ffill')
    if option == 2:
        return data_frame.fillna(method='bfill')#not reccomended bcs of lookahead
    if option ==  3:
        return data_frame.dropna()
    return 0

def follow_one_equity(data_frame, equity_ID):
    """This function plots daily price evolution 
    for one equity for all days that are included in the data.
    End a histogram of all basis points for the chosen equity.

    Args:
        data_frame (DataFrame): Data frame.
        equity_ID (Int): Equity ID that we are interested in.
    """
    one_equity_df = data_frame.loc[data_frame['equity'] == equity_ID]
    one_equity_df = one_equity_df.drop(columns=['equity'])
    one_equity_plot = one_equity_df.drop(columns = ['ID', 'day'])
    ax = sns.relplot(data=one_equity_plot.transpose(), kind="line", legend=False)
    plt.savefig('Daily_evolution_for '+str(equity_ID))
    fig, ax = plt.subplots()
    ax.hist(one_equity_plot, bins=50, density=True)
    ax.set_xlabel('Basis points')
    ax.set_ylabel('Frequency')
    ax.set_title(str(equity_ID))
    fig.savefig('Histogram of basis points for '+str(equity_ID))
    return 0

def follow_one_equity_smoothed(data_frame, equity_ID, day, index_of_columns):
    """This function plots daily price evolution 
    for one equity for all days that are included in the data.
    End a histogram of all basis points for the chosen equity.
    The input data frame is assumed to be smoothed to 40min intervals)

    Args:
        data_frame (DataFrame): Data frame.
        equity_ID (Int): Equity ID that we are interested in.
        day (int): the day that we want to see.
        index_of_columns (list): list with indices of columns 
        from r0-r_final.
    """
    one_equity_df = data_frame.loc[data_frame['equity'] == equity_ID]
    one_equity_df = one_equity_df.drop(columns=['equity'])
    one_equity_df = one_equity_df.loc[one_equity_df['day'] == day]
    one_equity_plot = one_equity_df.iloc[:, index_of_columns[0]:index_of_columns[1]]
    ax = sns.relplot(data=one_equity_plot.transpose(), kind="line", legend=False)
    plt.savefig('Daily_evolution_for '+str(equity_ID))
    fig, ax = plt.subplots()
    ax.hist(one_equity_plot, bins=50, density=True)
    ax.set_xlabel('Basis points')
    ax.set_ylabel('Frequency')
    ax.set_title(str(equity_ID) + " smoothed")
    fig.savefig('Histogram of basis points for '+str(equity_ID) + " smoothed")
    return 0

def add_derivatives(data_frame_0):
    """This function adds three features, all of them are derivatives

    Args:
        data_frame (DataFrame):

    Returns:
        DataFrame: original df with three added features (derivative_5,
          derivative_15, and derivative_30)
    """
    # We add feature that shows the derivative calculated from last 5 minutes
    data_frame = data_frame_0.copy()
    data_frame['derivative_5'] = (data_frame['r52']-data_frame['r51'])/5

    # We add derivative calculated from last 3 points (15min)

    data_frame['derivative_15'] = (data_frame['r52']-data_frame['r49'])/15

    # We add derivative calculated from last 30mins
    data_frame['derivative_30'] = (data_frame['r52']-data_frame['r46'])/30

    # We add derivative calculated from last 60mins
    data_frame['derivative_60'] = (data_frame['r52']-data_frame['r40'])/60

    # We add derivative calculated from last 120mins
    data_frame['derivative_120'] = (data_frame['r52']-data_frame['r28'])/120
    return data_frame

def add_variance_and_volatility(data_frame_0):
    """This function adds features which are variance calculated with full day info and volatility.

    Args:
        data_frame (DataFrame): original data frame

    Returns
        DataFrame: original df with added columns for variance and volatility
    """
    data_frame = data_frame_0.copy()
    data_frame['variance'] = np.var(data_frame.iloc[:, 3:], axis=1).values
    data_frame['volatility_15'] = np.multiply(np.var(data_frame.iloc[:, -4:], axis=1).values, 15)
    data_frame['volatility_1h'] = np.multiply(np.var(data_frame.iloc[:, -13:], axis=1).values, 60)
    return data_frame

def add_variance_and_mean(data_frame_0):
    """This function adds features which are variance calculated with full day info and mean.

    Args:
        data_frame (DataFrame): original data frame

    Returns
        DataFrame: original df with added columns for variance and mean, skewness and kurtosis.
    """
    data_frame = data_frame_0.copy()
    data_frame['variance'] = np.var(data_frame.iloc[:, 2:], axis=1).values
    data_frame['mean'] = np.mean(data_frame.iloc[:, 2:-1], axis = 1).values
    data_frame['skew'] = scipy.stats.skew(data_frame.iloc[:, 2:-1], axis = 1)
    data_frame['kurt'] = scipy.stats.kurtosis(data_frame.iloc[:, 2:-1], axis = 1)
    #data_frame['min'] = np.min(data_frame.iloc[:, 2:-1], axis=1).values
    #data_frame['max'] = np.max(data_frame.iloc[:, 2:-1], axis=1).values
    return data_frame

def add_same_equity_properties(data_frame_der):
    """Add some aggregated properties over all entries of one single equity ID. 
    (average over mean, variance, derivatives..)

    Args:
        data_frame_der (DataFrame): Input data frame that need to have 'equity' 
        column and 'mean', 'variance' and 'derivative_5', 'derivative_15', 'derivative_30'. 

    Returns:
        DataFrame: new data frame with added columns: 'equity_mean', 'equity_variance',
          'equity_derivative5','equity_derivative15', 'equity_derivative30'
    """
    grouped_by_equity = data_frame_der.groupby('equity')
    data_frame_n = data_frame_der.copy()
    data_frame_n['equity_mean'] = 0
    data_frame_n['equity_variance'] = 0
    data_frame_n['equity_derivative_5'] = 0
    data_frame_n['equity_derivative_30'] = 0
    data_frame_n['equity_derivative_15'] = 0
    eqs = data_frame_n['equity'].unique()
    for eq in eqs:
        data_frame_n.loc[data_frame_n['equity'] == eq,
                          'equity_mean'] = grouped_by_equity['mean'].mean()[eq]
        data_frame_n.loc[data_frame_n['equity'] == eq,
                          'equity_variance'] = grouped_by_equity['variance'].mean()[eq]
        data_frame_n.loc[data_frame_n['equity'] == eq,
                          'equity_derivative_5'] = grouped_by_equity['derivative_5'].mean()[eq]
        data_frame_n.loc[data_frame_n['equity'] == eq,
                          'equity_derivative_15'] = grouped_by_equity['derivative_15'].mean()[eq]
        data_frame_n.loc[data_frame_n['equity'] == eq,
                          'equity_derivative_30'] = grouped_by_equity['derivative_30'].mean()[eq]
    return data_frame_n

def add_same_day_properties(data_frame_der):
    """Add some aggregated properties over all entries of one single day. 
    (average over mean, variance, derivatives..)

    Args:
        data_frame_der (DataFrame): Input data frame that need to have 'equity' 
        column and 'mean', 'variance' and 'derivative_5', 'derivative_15', 'derivative_30'. 

    Returns:
        DataFrame: new data frame with added columns: 'daily_mean', 'daily_variance',
          'daily_derivative5','daily_derivative15', 'daily_derivative30'
    """
    grouped_by_day = data_frame_der.groupby('day')
    data_frame_n = data_frame_der.copy()
    data_frame_n['daily_mean'] = 0
    data_frame_n['daily_variance'] = 0
    data_frame_n['daily_derivative_5'] = 0
    data_frame_n['daily_derivative_30'] = 0
    data_frame_n['daily_derivative_15'] = 0
    days = data_frame_n['day'].unique()
    for d in days:
        data_frame_n.loc[data_frame_n['day'] == d,
                          'daily_mean'] = grouped_by_day['mean'].mean()[d]
        data_frame_n.loc[data_frame_n['day'] == d,
                          'daily_variance'] = grouped_by_day['variance'].mean()[d]
        data_frame_n.loc[data_frame_n['day'] == d,
                          'daily_derivative_5'] = grouped_by_day['derivative_5'].mean()[d]
        data_frame_n.loc[data_frame_n['day'] == d,
                          'daily_derivative_15'] = grouped_by_day['derivative_15'].mean()[d]
        data_frame_n.loc[data_frame_n['day'] == d,
                          'daily_derivative_30'] = grouped_by_day['derivative_30'].mean()[d]
    return data_frame_n

def compound_return(lower_ind, upper_ind, all_returns):
    """This function gives return between upper_ind and lower_ind 
    as (p_{upper_ind+1}-p_{lower_ind})/p_{lower_ind}. It computes it
      based on returns (r_{lower_ind} to r_{upper_ind}.['equity_mean']

    Args:
        lower_ind (int): we start measuring return from this point. 
        Can be in 0<= lower_ind<=52.
        upper_ind (_type_): We stop measuring return at this point. 
        Can be in lower_index<= upper_ind <=52.
        all_returns (_type_): Array with 53 elements-returns (float) from r_0 to r_52 

    Returns:
        float: return between lower_ind and upper_ind
    """
    result = 1
    for i in range(lower_ind, upper_ind+1):
        result *= (all_returns[i]/10**4 + 1)
    return result - 1

def translate_into_40_min_returns_slow(data_frame):
    """This function takes in a data frame with columns: 
    (ID, day, equity, r1-r52) and returns a new data frame with 
    columns (day, equity, rr0-rr6), where rr0-rr6 contain returns 
    over 8 consequtive 5min periods. This function makes a loop over 
    data frame and is very slow.

    Args:
        data_frame (DataFrame): Input data frame with returns r0-r52 in 
        columns with indices (2-54). First two columns should be 'day' and 'equity'.

    Returns:
        _type_: DataFrame
    """
    result = pd.DataFrame()
    result['day'] = data_frame['day']
    result['equity'] = data_frame['equity']
    for i in range(7):
        result['rr' + str(i)] = 0.0
    for i in range(len(data_frame)):
        for ret in range(6):
            lower = ret*8
            upper = (ret+1)*8 -1
            result.iloc[i, ret+2] += compound_return(lower, upper, data_frame.iloc[i, -53:])
        result.iloc[i, 8] += compound_return(48, 52, data_frame.iloc[i, -53:])
    return result
def translate_into_40_min_returns_fast(data_frame):
    """This function takes in a data frame with columns: 
    (ID, day, equity, r1-r52) and returns a new data frame with 
    columns (day, equity, rr0-rr6), where rr0-rr6 contain returns 
    over 8 consequtive 5min periods. This function is vectorized and performs 
    much faster than the slow version.

    Args:
        data_frame (DataFrame): Input data frame with returns r0-r52 in 
        columns with indices (2-54). First two columns should be 'day' and 'equity'.

    Returns:
        _type_: DataFrame
    """
    result = pd.DataFrame()
    result['day'] = data_frame['day']
    result['equity'] = data_frame['equity']
    for ret in range(6):
        lower = ret*8
        upper = (ret+1)*8 -1
        result['rr' + str(ret)] = 1
        for i in range(lower, upper+1):
            result.iloc[:, 2+ret] *= data_frame.iloc[:,3+i]/10**4 + 1
        result.iloc[:, 2+ret] -= 1
        result.iloc[:, 2+ret] *= 10**4
    lower = 48
    upper = 52
    result['rr6'] = 1
    for i in range(lower, upper+1):
        result.loc[:, 'rr6'] *= data_frame.iloc[:,3+i]/10**4 + 1
    result.loc[:, 'rr6'] -= 1
    result.loc[:, 'rr6'] *= 10**4
    return result

def translate_into_30_min_returns_fast(data_frame):
    """This function takes in a data frame with columns: 
    (ID, day, equity, r0-r52) and returns a new data frame with 
    columns (day, equity, rr0-rr8), where rr0-rr8 contain returns 
    over 9 consequtive 5min periods. This function is vectorized and performs 
    much faster than the slow version.

    Args:
        data_frame (DataFrame): Input data frame with returns r0-r52 in 
        columns with indices (2-54). First two columns should be 'day' and 'equity'.

    Returns:
        _type_: DataFrame
    """
    result = pd.DataFrame()
    result['day'] = data_frame['day']
    result['equity'] = data_frame['equity']
    for ret in range(8):
        lower = ret*6
        upper = (ret+1)*6 -1
        result['rr' + str(ret)] = 1
        for i in range(lower, upper+1):
            result.iloc[:, 2+ret] *= data_frame.iloc[:,3+i]/10**4 + 1
        result.iloc[:, 2+ret] -= 1
        result.iloc[:, 2+ret] *= 10**4
    lower = 48
    upper = 52
    result['rr8'] = 1
    for i in range(lower, upper+1):
        result.loc[:, 'rr8'] *= data_frame.iloc[:,3+i]/10**4 + 1
    result.loc[:, 'rr8'] -= 1
    result.loc[:, 'rr8'] *= 10**4
    return result

def split_list(list_in, n, seed):
    """A function that splits a list in two parts.

    Args:
        list_in (list): A list
        n (Int): Number of sublists.
    Returns:
        List: List containing new sublists.
    """
    random.seed(seed)
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def moving_average(df, equity_index, indices_of_r, window_size):
    """Calculates & plots a moving mean smoothing.

    Args:
        df (DataFrame): dataFrame 
        equity_index (Int): This is the row that we will analyze.
        indices_of_r (List): [a, b] - Starting and ending index of r (r0-r_final)
        window_size (Int): Size of the window in which we average.
    """
    def moving_average_helper(a, n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    avr_m = moving_average_helper(df.iloc[equity_index, indices_of_r[0]:indices_of_r[1]].values,
                                   window_size)
    plt.plot(df.iloc[equity_index, indices_of_r[0]:indices_of_r[1]].values)
    plt.plot(avr_m)
    plt.show()
    return avr_m

def smooth_by_exponential(data_frame, half_l, indices, include_day_eq = True):
    """This function takes in a data frame with columns: 
    (ID, day, equity, r) and returns a new data frame with 
    columns (r), 

    Args:
        data_frame (DataFrame): Input data frame with returns r0-r52 in 
        columns with indices (3-55). First three columns should be 'ID',day' and 'equity'.
        indices (List) : This should be a list with two integers (eg. [3, 55] - this indicates 
        that returns columns have indeces from 3-54)
        include_day_eq (Bool): if True the output will include 'day' and 'equity' columns.

    Returns:
        _type_: DataFrame, DataFrame (exponential averge and variance).
    """
    df2 = data_frame.iloc[:, indices[0]:indices[1]]
    new_str = df2.ewm(halflife=half_l, axis = 1)
    new_avrg = new_str.mean()
    new_var = new_str.var()
    if include_day_eq:
        new_var.insert(0, 'day', data_frame['day'])
        new_var.insert(1, 'equity', data_frame['equity'])
        new_avrg.insert(0, 'day', data_frame['day'])
        new_avrg.insert(1, 'equity', data_frame['equity'])
    return new_avrg, new_var

def log_returns(data_frame):
    """This function transforms returns to log returns as log(10**4 + r).

    Args:
        data_frame (DataFrame): Input data frame where returns start at column 2.

    Returns:
        DataFrame: Data Frame with log returns instead of returns.
    """
    for j in range(2, len(data_frame.iloc[0, :])):
        print(j)
        data_frame.iloc[:, j] = np.log(data_frame.iloc[:, j]/10000 + 1)
    return data_frame

def handling_outliers(df, cutoff_z, axiss):
    """This function takes a dataframe df and removes outliers with z
    score above value 'cutoff_z'  along the axis (if axiss =1 we do by
    row, if axiss=0 by feature.)

    Args:
        df (DataFrame): Initial df (assuming that it has columns rr0-rr6).
        cutoff_z (Float): z score where we do the cutoff.

    Returns:
        df: dataframe with values that have z score above treshold replaced by bfill/mean.
    """
    z_scores_mask = np.abs(zscore(df, axis = axiss))
    print(np.sum(z_scores_mask>cutoff_z, axis=axiss)/len(z_scores_mask))
    df_n = df.mask(z_scores_mask > cutoff_z)
    inter_columns = ['rr0', 'rr1', 'rr2', 'rr3', 'rr4', 'rr5', 'rr6']
    other = list(set(df.columns) - set(inter_columns))
    print(inter_columns)
    print(other)
    # Replace outliers in columns 'inter_columns' by bfill
    # Other columns just replace by mean
    df_n[other] = df_n[other].fillna(df_n[other].mean())
    df_n[inter_columns] = df_n[inter_columns].fillna(method='bfill')
    return df_n

def handling_outliers_test(df, cutoff_z, means, variances, axiss):
    """This function takes a dataframe df and removes outliers with z
    score above value 'cutoff_z' but it uses pre defined means and variances (for each feature)
      - calculated from train set.

    Args:
        df (DataFrame): Initial df (assuming that it has columns rr0-rr6).
        cutoff_z (Float): z score where we do the cutoff.

    Returns:
        df: dataframe with values that have z score above treshold replaced by bfill/mean.
    """
    z_scores_mask = np.abs((df - means) / np.sqrt(variances))
    print(np.sum(z_scores_mask>cutoff_z, axis=axiss)/len(z_scores_mask))
    df_n = df.mask(z_scores_mask > cutoff_z)
    inter_columns = ['rr0', 'rr1', 'rr2', 'rr3', 'rr4', 'rr5', 'rr6']
    other = list(set(df.columns) - set(inter_columns))
    print(inter_columns)
    print(other)
    # Replace outliers in columns 'inter_columns' by bfill
    # Other columns just replace by mean
    df_n[other] = df_n[other].fillna(df_n[other].mean())
    df_n[inter_columns] = df_n[inter_columns].fillna(method='bfill')
    return df_n

def print_locations_nan(df):
    """This function prints locations of Nan values in ech column.

    Args:
        df (DataFrame): input df.

    Returns:
        Int: zero
    """
    if not df.isnull().values.any():
        print("No NaN in your dataframe!")
        return 0
    else:
        for col in df.columns:
            print('Train')
            print(col)
            print(df[df[col].isnull()].index.tolist())
        for col in df.columns:
            print('Test')
            print(col)
            print(df[df[col].isnull()].index.tolist())
        return 0

def calculate_label_from_data(df):
    """The df is assumed to have returns at times rr0-rr8.
      (30min intervals). We compute if in each 2h window we gain more than 25 (1),
      lose more than 25 (-1) ot sth inbetween (0).

    Args:
        df (_type_): _description_
    """
    return 0

def handling_outliers_new(df, cutoff_z, axiss):
    """This function takes a dataframe df and removes outliers with z
    score above value 'cutoff_z'  along the axis (if axiss =1 we do by
    row, if axiss=0 by feature.)

    Args:
        df (DataFrame): Initial df (assuming that it has columns rr0-rr6).
        cutoff_z (Float): z score where we do the cutoff.
        axiss (Int): 0 if doing by feature(column), 1 id doing by row.

    Returns:
        df: dataframe with values that have z score above treshold replaced by bfill/mean.
    """
    z_scores_mask = np.abs(zscore(df, axis = axiss))
    if axiss == 0:
        size = len(z_scores_mask)
    else:
        size = len(z_scores_mask.iloc[0, :])
    print("Percentage of discarded values")
    print(np.sum(z_scores_mask>cutoff_z, axis=axiss)/size)
    df_n = df.mask(z_scores_mask > cutoff_z)

    # Replace outliers in columns 'inter_columns' by bfill
    # Other columns just replace by mean
    if axiss == 0:
        df_n = df_n.fillna(df_n.mean(), axis = axiss)
    else:
        df_n = df_n.fillna(method='bfill', axis = axiss)
    return df_n

def handling_outliers_test_new(df, cutoff_z, means, variances, axiss):
    """This function takes a dataframe df and removes outliers with z
    score above value 'cutoff_z' but it uses pre defined means and variances (for each feature)
      - calculated from train set.

    Args:
        df (DataFrame): Initial df (assuming that it has columns rr0-rr6).
        cutoff_z (Float): z score where we do the cutoff.

    Returns:
        df: dataframe with values that have z score above treshold replaced by bfill/mean.
    """
    if axiss == 0:
        z_scores_mask = np.abs((df - means) / np.sqrt(variances))
    else:
        z_scores_mask = np.abs((df.sub(means, axis = 0)).div(np.sqrt(variances), axis = 0))
    if axiss == 0:
        size = len(z_scores_mask)
    else:
        size = len(z_scores_mask.iloc[0, :])
    print(np.sum(z_scores_mask>cutoff_z, axis=axiss)/size)
    df_n = df.mask(z_scores_mask > cutoff_z)
    # Replace outliers in columns 'inter_columns' by bfill
    # Other columns just replace by mean
    if axiss == 0:
        df_n = df_n.fillna(df_n.mean(), axis = axiss)
    else:
        df_n = df_n.fillna(method='bfill', axis = axiss)
    return df_n

def add_min_max(df_raw, df_features, inds):
    """Thia function accepts df with returns at each
      time and a df with different features, and indices
        of columns in the first df where returns are located.
        It adds 5 columns related to mins and maxs to df_features and
          returns this df.

    Args:
        df_raw (DataFrame): df with returns in columns from inds[0]-inds[1]
        df_features (DataFrame): df with features
        inds (List): list with 2 elements (Ints).

    Returns:
        DataFrame: original df_features with added 5 columns.
    """    
    mins = np.min(df_raw.iloc[:, inds[0]:inds[1]], axis=1)
    maxes = np.max(df_raw.iloc[:, inds[0]:inds[1]], axis=1)
    difference = np.abs(mins- maxes)
    loc_min = np.argmin(df_raw.iloc[:, inds[0]:inds[1]], axis=1)
    loc_max = np.argmax(df_raw.iloc[:, inds[0]:inds[1]], axis=1)
    df_features['diff_min_max'] = difference
    df_features['loc_min'] = loc_min
    df_features['loc_max'] = loc_max
    df_features['max'] = maxes
    df_features['min'] = mins
    return df_features

def add_moments(df_raw, df_features, inds):
    """This function adds 6 features to the df_features,
      by taking returns data from df_raw.

    Args:
        df_raw (DataFrame): original data frame with returns.
        df_features (DataFrame): df with features, same number of rows as df_raw.
        inds (List): list containing two Ints (starting and ending index of returns in df_raw).

    Returns
        DataFrame: df_features with added 6 columns.
    """
    df_features['variance'] = np.var(df_raw.iloc[:, inds[0]:inds[-1]+1], axis=1).values
    df_features['mean'] = np.mean(df_raw.iloc[:, inds[0]:inds[-1]+1], axis=1).values
    df_features['median'] = np.median(df_raw.iloc[:, inds[0]:inds[-1]+1], axis=1)
    df_features['skew'] = scipy.stats.skew(df_raw.iloc[:, inds[0]:inds[-1]+1], axis = 1)
    df_features['kurt'] = scipy.stats.kurtosis(df_raw.iloc[:, inds[0]:inds[-1]+1], axis = 1)
    mm =  scipy.stats.mode(df_raw.iloc[:, inds[0]:inds[-1]+1], axis = 1, keepdims=True)
    df_features['mode'] = np.array(mm)[0].flatten()
    return df_features

def add_nr_positive(df_raw, df_features, inds):
    """This function adds 2 features, number of
    positive returns and number of 0.

    Args:
        df_raw (DataFrame): original data frame with returns.
        df_features (DataFrame): df with features, same number of rows as df_raw.
        inds (List): list containing two Ints (starting and ending index of returns in df_raw).

    Returns
        DataFrame: df_features with added 2 columns.
    """
    df_features['nr_positive'] = df_raw.iloc[:, inds[0]:inds[1]+1][df_raw.iloc[:, inds[0]:inds[1]+1]
                                                                   > 0].count(axis=1).values
    df_features['nr_zero'] = df_raw.iloc[:, inds[0]:inds[1]+1][df_raw.iloc[:, inds[0]:inds[1]+1]
                                                               == 0].count(axis=1).values
    return df_features

def label_it_up(df_raw, df_features, inds):
    """This function takes df_raw which contains returns for every 30min in
    columns at inds[0] to inds[1]. 
    It ads a3 columns to df_features with -1, 1 or 0 if the returns in corresponding
    2h went up, down, or stagnated.

    Args:
        df_raw (DataFrame): df with returns every 30min.
        df_features (DatFrame): df with features, same number of rows as df_raw.
        inds (List): list containing two Ints (starting and ending index of returns in df_raw).

    Returns:
        DataFrame: df_features with added 3 columns (first 2h, middle 2h, last 2h)
    """
    # here we assume that df_raw is split in 30min intervals
    #first_2h
    r_1 = (((df_raw.iloc[:, 0]/10**4 +1) * (df_raw.iloc[:, 1]/10**4 +1) *
        (df_raw.iloc[:, 2]/10**4 +1) * (df_raw.iloc[:, 3]/10**4 +1) )-1) * 10**4
    result_1 = np.zeros(len(r_1))
    result_1[r_1.values < -25] -= 1
    result_1[r_1.values > 25] += 1
    df_features['first_2h'] = result_1

    r_2 = (((df_raw.iloc[:, 2]/10**4 +1) * (df_raw.iloc[:, 3]/10**4 +1) *
        (df_raw.iloc[:, 4]/10**4 +1) * (df_raw.iloc[:, 5]/10**4 +1) )-1) * 10**4
    result_2 = np.zeros(len(r_2))
    result_2[r_2.values < -25] -= 1
    result_2[r_2.values > 25] += 1
    df_features['middle_2h'] = result_2

    r_3 = (((df_raw.iloc[:, 5]/10**4 +1) * (df_raw.iloc[:, 6]/10**4 +1) *
        (df_raw.iloc[:, 7]/10**4 +1) * (df_raw.iloc[:, 8]/10**4 +1) )-1) * 10**4
    result_3 = np.zeros(len(r_3))
    result_3[r_3.values < -25] -= 1
    result_3[r_3.values > 25] += 1
    df_features['last_2h'] = result_3
    return df_features

def autocorrelation_length2(arr):
    """Function returning autocorrelations of an array.

    Args:
        arr (array): Input array.

    Returns:
        Array: autocorrelations.
    """    
    corr = np.correlate(arr,arr,mode='full')/arr.size
    return corr[corr.size//2:]

def autocorr_by_day(arr, features):
    """This function calculates average
      autocorrelation length for all equities 
      in one day from first 5 points of autocorr function.
      The arr is a dataframe that should have day and equity columns in first two columns.

    Args:
        arr (DataFrame): dataFrame with columns: Day, equity, r0-r52.

    Returns:
        DataFrame: original df with added length_day.
    """
    features['day'] = arr['day'] 
    days = arr['day'].unique()
    arr_n = arr.copy()
    arr_n['length_day'] = 0
    for dayy in days:
        #print(dayy)
        arr_day = arr[arr['day'] == dayy].iloc[:, 2:]
        corr = np.zeros(len(arr_day.iloc[0, :]))
        for eq in range(len(arr_day)):
            corr += autocorrelation_length2(arr_day.iloc[eq, :])
        corr /= len(arr_day)
        miny = np.min(corr)
        corr += np.abs(miny)*1.1
        corr /= np.max(corr)
        gg = np.polyfit(range(len(corr[:5])), np.log(corr[:5]), 1)
        features.loc[arr_n['day'] == dayy, 'length_day'] =  -1/gg[0]
    return features
def autocorr_by_equity(arr, features):
    """This function calculates average
      autocorrelation length for one equity in all days 
     from first 5 points of autocorr function.
      The arr is a dataframe that should have day and equity columns in first two columns.

    Args:
        arr (DataFrame): dataFrame with columns: Day, equity, r0-r52.

    Returns:
        DataFrame: original df with added length_eq.
    """
    features['equity'] = arr['equity'] 
    eqs = arr['equity'].unique()
    arr_n = arr.copy()
    arr_n['length_eq'] = 0
    for eqy in eqs:
        arr_eq = arr[arr['equity'] == eqy].iloc[:, 2:]
        corr = np.zeros(len(arr_eq.iloc[0, :]))
        for day in range(len(arr_eq)):
            corr += autocorrelation_length2(arr_eq.iloc[day, :])
        corr /= len(arr_eq)
        miny = np.min(corr)
        corr += np.abs(miny)*1.1
        corr /= np.max(corr)
        gg = np.polyfit(range(len(corr[:5])), np.log(corr[:5]), 1)
        features.loc[arr_n['equity'] == eqy, 'length_eq'] =  -1/gg[0]
    return features
