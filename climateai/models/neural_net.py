"""
@version: 2023.01.10
@author: Tilman Kerl
"""

import math
import shap
import pickle

import pathlib

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

    def __init__(self, dim = 24, model_name="nn"):
        self.file_dir = str(pathlib.Path(__file__).parent.resolve())
        self.input_dim = dim # len(self.columns)
        self.units = self.input_dim * 2
        self.kernel_initializer = "normal"
        # self.activation_func = "relu"
        # This is also known as mini-batch gradient descent. 
        # the bigger the number -> underfitting, smaller -> overfitting
        self.batch_size = 15
        self.epochs = 60
        self.model_name = model_name
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

        if save: 
            self.model.save(f'{self.file_dir}/saved_models/{self.model_name}.model')


    def inspect_feature_importance(self, X, feautures, js=False, force=False):

        shap_values = None
        if not force:
            try:
                with open(f"{self.file_dir}/shap-values-{self.model_name}.pk", "rb") as fi:
                    shap_values = pickle.load(fi)
            except Exception as e:
                print(e)

        if not shap_values:
            idx = np.random.randint(X.shape[0], size=500)
            shap_value_subset = X[idx,:]
            explainer = shap.DeepExplainer(self.model, shap_value_subset)
            shap_values = explainer.shap_values(shap_value_subset)
            with open(f"{self.file_dir}/shap-values-{self.model_name}.pk", "wb") as fi:
                pickle.dump(shap_values, fi)
                    
        shap.summary_plot(shap_values, plot_type="bar", features = feautures)
      


    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(f'{self.file_dir}/saved_models/{self.model_name}.model')
            return True
        except Exception as e:
            print(e)
            return False

    def get_model(self):
        model = tf.keras.Sequential()

        # Defining the Input layer and FIRST hidden layer, both are same!
        model.add(Dense(units=self.units/2, input_dim=self.input_dim, kernel_initializer=self.kernel_initializer, activation='relu'))

        # Defining the Second layer of the model
        # after the first layer we don't have to specify input_dim as keras configure it automatically
        model.add(Dense(units=self.units, kernel_initializer=self.kernel_initializer, activation='tanh'))
        # model.add(Dense(units=self.units, kernel_initializer=self.kernel_initializer, activation='tanh'))

        # The output neuron is a single fully connected node 
        # Since we will be predicting a single number
        model.add(Dense(1, kernel_initializer='normal'))  

        return model
        

    def evaluate(self, X_test, y_test, full=False):        
        self.X_test, self.y_test = self.__init_data(X_test, y_test)
        result = self.model.evaluate(self.X_test, self.y_test)
        rmse_model = math.sqrt(result[0])
        print("RMSE: ", rmse_model)
        if full:
            self.model.summary()
            print(f'Model performance (MSE, PREC, AUC) on the test set: {result}')
            return rmse_model, result        
        return rmse_model
