''' Function that run RNN'''
import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
import pandas as pd

def RNN_function(batch_s, nr_epochs, lr, name):
    x_train0 = np.load('./Data/Data_after_feature_en/x_train_features_RNN.npy')
    x_test = np.load('./Data/Data_after_feature_en/x_test_features_RNN.npy')
    y_train0 = np.load('./Data/Data_after_feature_en/y_train_features_RNN.npy')
    x_train, x_val, y_train, y_val = train_test_split(x_train0, y_train0, test_size=0.2,
                                                      random_state=42)
    print('Imported the data and split it into train, validation.')

    model = keras.models.Sequential([
    keras.layers.SimpleRNN(2, return_sequences=True, input_shape = (53, 3)),
    keras.layers.SimpleRNN(2, activation = 'tanh'),
    keras.layers.Dense(3, activation = "softmax")
    ])
    opt = SGD(learning_rate = lr)
    model.compile(loss = "categorical_crossentropy",
                  optimizer = opt,
                  metrics = ['accuracy'])
    print('Fitting the model.')
    history = model.fit(x_train, y_train,
              batch_size = batch_s,
          epochs = nr_epochs,validation_data = (x_val, y_val), verbose = 1)
    print('Predicting.')
    y_predictions_rnn = np.argmax(model.predict(x_test), axis=1)-1

    x_test_cl = pd.read_csv('./Data/Data_clean/x_test_clean.csv')
    final_df = pd.DataFrame()
    final_df['ID'] = x_test_cl['ID']
    final_df['reod'] = y_predictions_rnn
    final_df.to_csv('./Predictions/RNN_prediction_' + name + '.csv', index=False)
    return model, history
