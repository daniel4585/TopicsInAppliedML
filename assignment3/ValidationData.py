
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ValidationData(object):
    def __init__(self, dataframe, cat_mapping, numerical_mapping):
        self.df = dataframe

        numerical = dataframe.select_dtypes(['int64', 'float64'])
        for col, val in numerical.iteritems():
            dataframe[col] = val.fillna(numerical_mapping[col])

        self.cat_mapping = cat_mapping



