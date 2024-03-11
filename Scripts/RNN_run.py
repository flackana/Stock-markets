'''Run RNN with some set of parameters and save the predictions in the Predictions folder.'''
import sys
sys.path.append('./src')
from my_functions.RNN_full import RNN_function
BATCH = 64
EPOCH = 2
LEARNING_R = 0.01
NAME = 'test_rnn'

model = RNN_function(BATCH, EPOCH, LEARNING_R, NAME)
