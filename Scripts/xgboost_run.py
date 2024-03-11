'''Run xgboost with some set of parameters and save the predictions in the Predictions folder.'''
import sys
sys.path.append('./src')
from my_functions.xgboost_full import xg_boost2

parameters = {'objective':'multi:sofprob', 'eta':0.1, 'n_estimators':100, 'max_depth':3,
              'colsample_bytree':0.6}
model = xg_boost2(parameters, 'renewed1', bare=False)