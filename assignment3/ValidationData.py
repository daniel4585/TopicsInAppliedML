
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ValidationData(object):
    def __init__(self, dataframe, cat_mapping, cat_mapping_avg, numerical_mapping):
        self.df = dataframe

        numerical = dataframe.select_dtypes(['int64', 'float64'])
        for col, val in numerical.iteritems():
            if col == 'Id':
                continue
            dataframe[col] = val.fillna(numerical_mapping[col])

        self.cat_mapping = cat_mapping
        self.cat_mapping_avg = cat_mapping_avg

        for col, val in self.df.select_dtypes(['object']).iteritems():
            if col == 'Id':
                continue

            self.df[col + "Rank"] = val.apply(lambda x: self.catMapping(col, x))
            self.df = self.df.drop(col, axis=1)

    def catMapping(self, col, x):
        if (col, x) in self.cat_mapping:
            return self.cat_mapping[(col, x)][0]
        else:
            return self.cat_mapping_avg[col]
