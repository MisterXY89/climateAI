"""
@version: 2022.10.30
@author: Tilman Kerl

https://github.com/joyfang1106/RLANet
https://stackoverflow.com/questions/22431676/neural-networks-in-realtime
"""

import numpy as np
import pandas as pd

import tensorflow as tf

from keras.models import Sequential
from keras import layers
from keras import losses
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

import eli5
from eli5.sklearn import PermutationImportance

from climateai.config import *


class NeuralNet(object):

    def __init__(self, dim = 24):
        self.input_dim = dim # len(self.columns)
        self.units = self.input_dim * 2
        self.kernel_initializer = "normal"
        # self.activation_func = "relu"
        # This is also known as mini-batch gradient descent. 
        # the bigger the number -> underfitting, smaller -> overfitting
        self.batch_size = 15
        self.epochs = 70
        self.model = self.get_model()

    def __init_data(self, X, y):
        # self.columns = X.columns
        return X, y
        
        # self.X_train = X_train.to_numpy()
        # self.y_train = y_train.to_numpy().transpose()
        # self.X_test = X_test.to_numpy()
        # self.y_test = y_test.to_numpy().transpose()

    def train(self, X_train, y_train, save=True):
        self.X_train, self.y_train = self.__init_data(X_train, y_train)        
        self.model.compile(
            loss = 'mean_squared_error',             
            optimizer = 'adam',
            # optimizer = 'rmsprop',
            metrics = ["Precision", "AUC"]
        )

        # Fitting the ANN to the Training set
        self.model.fit(
            self.X_train, self.y_train,
            batch_size = self.batch_size, 
            epochs = self.epochs, 
            verbose = 1,
            # validation_data =
        )

        if save: self.model.save('saved_models/nn.model')

    def __get_train_subset():
        pass

    def find_best_parameters(self, X, y):
        # **sk_params
        sk_wrapper_model = KerasRegressor(build_fn = self.get_model())
        sk_wrapper_model.fit(X,y)

        perm = PermutationImportance(sk_wrapper_model, random_state = SEED).fit(X,y)
        eli5.show_weights(perm, feature_names = X.columns.tolist())

    def get_model(self):
        model = tf.keras.Sequential()

        # Defining the Input layer and FIRST hidden layer, both are same!
        model.add(Dense(units=self.units/2, input_dim=self.input_dim, kernel_initializer=self.kernel_initializer, activation='relu'))
        
        model.add(Dense(units=self.units, kernel_initializer=self.kernel_initializer, activation='tanh'))
        model.add(Dense(units=self.units, kernel_initializer=self.kernel_initializer, activation='tanh'))
        model.add(Dense(units=self.units, kernel_initializer=self.kernel_initializer, activation='tanh'))
        model.add(Dense(units=self.units, kernel_initializer=self.kernel_initializer, activation='tanh'))
        model.add(Dense(units=self.units, kernel_initializer=self.kernel_initializer, activation='tanh'))
        model.add(Dense(units=self.units, kernel_initializer=self.kernel_initializer, activation='tanh'))
        model.add(Dense(units=self.units, kernel_initializer=self.kernel_initializer, activation='tanh'))
        model.add(Dense(units=self.units, kernel_initializer=self.kernel_initializer, activation='tanh'))

        # The output neuron is a single fully connected node 
        # Since we will be predicting a single number
        model.add(Dense(1, kernel_initializer='normal'))  

        return model
        

    def evaluate(self, X_test, y_test):
        self.model.summary()
        self.X_test, self.y_test = self.__init_data(X_test, y_test)
        mse = self.model.evaluate(self.X_test, self.y_test)
        # print(f' Model loss on the test set: {loss}')
        print(f' Model MSE on the test set: {mse}')
