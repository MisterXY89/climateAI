
import os
import glob
import math
import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from climateai.config import *

class DataLoader(object):

    def __init__(self):        
        self.data_path = f"{pathlib.Path(__file__).parent.parent.resolve()}/data"
        self.file_num = 882

    def get_random_files(self, base=True):
        all_files = glob.glob(f"{self.data_path}/*.csv")
        if base:
            return all_files[:10]
        shuffled_files = np.random.shuffle(all_files)
        return shuffled_files
        
    def load(self):
        files = self.get_random_files()
        self.df_list = [
            pd.read_csv(file, sep=";") 
            for file in tqdm(files, total=len(files))
        ]        
        self.full_df = pd.concat(self.df_list)

        # self.full_df.hist(column = "prec")
        # plt.show()        
        
        return self.full_df
