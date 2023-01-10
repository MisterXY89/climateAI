"""
@version: 2022.10.30
@author: Tilman Kerl

https://github.com/joyfang1106/RLANet
https://stackoverflow.com/questions/22431676/neural-networks-in-realtime
"""

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
from climateai.models.neural_net import NeuralNet


class DeepNeuralNet(NeuralNet):

    def __init__(self, dim = 24):        
        # self.model_name = "dnn"
        super().__init__(model_name = "dnn")
        # self.epochs = 50
        # self.model = self.get_model()

    def get_model(self):
        model = tf.keras.Sequential()

        # Defining the Input layer and FIRST hidden layer, both are same!
        model.add(Dense(units=self.units/2, input_dim=self.input_dim, kernel_initializer=self.kernel_initializer, activation='relu'))

        # Defining the Second layer of the model
        # after the first layer we don't have to specify input_dim as keras configure it automatically
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