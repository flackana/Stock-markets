''' Functions to run xgb model.'''
import xgboost
import pandas as pd
from sklearn.metrics import accuracy_score

def xg_boost(params, name, bare = False):
    """A function that loads the clean data,
      prepares it for xgBoost, trains the model with given
        parameters (params), saves the predictions and prints
          the accuracy on the training set.

    Args:
        params (Dict): Dictionary with parameters passed to the XGBoost.
        name (str): Predictions will be saved under this name.
        bare (bool, defult = False): if True the model will include more features.
    Returns:
        _type_: DataFrame (predictions on the test set)
    """
    # Importing the data
    x_train0 = pd.read_csv('./Data/Data_after_feature_en/x_train_features_XGB.csv')
    Y_train0 = pd.read_csv('./Data/Data_clean/Y_train_clean.csv')
    x_test = pd.read_csv('./Data/Data_after_feature_en/x_test_features_XGB.csv')
    #x_test = x_test.drop(['equity', 'day', 'equity_derivative_5',
    #                   'equity_derivative_15', 'equity_derivative_30'], axis = 1)
    #x_train0 = x_train0.drop(['equity', 'day', 'equity_derivative_5', 'equity_derivative_15',
    #                       'equity_derivative_30'], axis = 1)
    x_test = x_test.drop(['equity'], axis = 1)
    x_train0 = x_train0.drop(['equity'], axis = 1)
    if bare:
        x_test = x_test.drop(['daily_mean', 'daily_variance', 'equity_mean',
                               'equity_variance', 'equity_derivative_5', 'equity_derivative_15',
                                 'equity_derivative_30','rr0', 'rr1'], axis = 1)
        x_train0 = x_train0.drop(['daily_mean', 'daily_variance', 'equity_mean',
                               'equity_variance', 'equity_derivative_5', 'equity_derivative_15',
                                 'equity_derivative_30', 'rr0', 'rr1'], axis = 1)
    print('Features: ')
    print(x_train0.columns)
    # Train the model
    print("Training")
    xgb_class_full = xgboost.XGBClassifier(**params)
    xgb_class_full.fit(x_train0, Y_train0
                  .values.ravel())
    print("Predicting")
    # Predicting
    y_pred = xgb_class_full.predict(x_test)
    y_pred -= 1

    x_test_cl = pd.read_csv('./../../Data/Data_clean/x_test_clean.csv')
    final_df = pd.DataFrame()
    final_df['ID'] = x_test_cl['ID']
    final_df['reod'] = y_pred
    final_df.to_csv('./../../Predictions/xgboost_prediction_' + name + '.csv', index=False)

    # Predicting in the training
    y_pred_train = xgb_class_full.predict(x_train0)
    y_pred_train -= 1
    print('Accuracy on the training set is ' +
          str(round(accuracy_score(Y_train0.values.ravel()-1, y_pred_train), 2)))
    return xgb_class_full


def xg_boost2(params, name):
    """A function that loads the clean data,
      prepares it for xgBoost, trains the model with given
        parameters (params), saves the predictions and prints
          the accuracy on the training set.

    Args:
        params (Dict): Dictionary with parameters passed to the XGBoost.
        name (str): Predictions will be saved under this name.
        bare (bool, defult = False): if True the model will include more features.
    Returns:
        _type_: DataFrame (predictions on the test set)
    """
    # Importing the data
    x_train0 = pd.read_csv('./Data/Data_after_feature_en/xgb_train_features_renewed.csv')
    Y_train0 = pd.read_csv('./Data/Data_clean/Y_train_clean.csv')
    x_test = pd.read_csv('./Data/Data_after_feature_en/xgb_test_features_renewed.csv')
    print('Features: ')
    print(x_train0.columns)
    # Train the model
    print("Training")
    xgb_class_full = xgboost.XGBClassifier(**params)
    xgb_class_full.fit(x_train0, Y_train0
                  .values.ravel())
    print("Predicting")
    # Predicting
    y_pred = xgb_class_full.predict(x_test)
    y_pred -= 1

    x_test_cl = pd.read_csv('./Data/Data_clean/x_test_clean.csv')
    final_df = pd.DataFrame()
    final_df['ID'] = x_test_cl['ID']
    final_df['reod'] = y_pred
    final_df.to_csv('./Predictions/xgboost_prediction_' + name + '.csv', index=False)

    # Predicting in the training
    y_pred_train = xgb_class_full.predict(x_train0)
    y_pred_train -= 1
    print('Accuracy on the training set is ' +
          str(round(accuracy_score(Y_train0.values.ravel()-1, y_pred_train), 2)))
    return xgb_class_full
