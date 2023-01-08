
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from climateai.config import *

class Process(object):

    def __init__(self):
        pass        

    @staticmethod
    def standardise_data(X_train, X_test, X_val=None):
        sc = StandardScaler()
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.transform(X_test)
        if isinstance(X_val, pd.DataFrame):
            X_val_scaled = sc.transform(X_val)
            return X_train_scaled, X_test_scaled, X_val_scaled

        return X_train_scaled, X_test_scaled


    @staticmethod
    def winsorise(X, drop_cols=[], plot=False):
        if isinstance(X, np.ndarray):
            X = X.to_frame()
        X_wins = X.copy()

        for col in X_wins.drop(columns=drop_cols).columns:    
            X_wins.loc[X_wins[col] >= X_wins[col].quantile(0.99), col] = X_wins[col].quantile(0.99)
        X_wins = X_wins.drop(columns=drop_cols)
        if plot:
            X_wins.hist(alpha=.8, color="lightgray", edgecolor="black", bins=20, figsize=(10,10))
            plot.show()
        
        return X_wins
       

    @staticmethod
    def split_data(df):
        X = df.drop(columns=["prec"])#.values
        y = df["prec"]#.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, 
            test_size = TEST_SIZE, 
            random_state = SEED
        ) 
        X_wins_train = Process.winsorise(X_train)
        X_wins_test = Process.winsorise(X_test)
        X_scaled_train, X_scaled_test = Process.standardise_data(X_train, X_test)
        return X_train, X_test, y_train, y_test, X_wins_train, X_wins_test, X_scaled_train, X_scaled_test