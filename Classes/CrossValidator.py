# -*- coding: utf-8 -*-
"""
Created on Mon May 19 22:41:50 2025

@author: clear
"""

from sklearn.model_selection import KFold

class CrossValidator:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits              # Number of folds for cross-validation
        self.metrics_per_fold = []            # List of metrics for each fold
        self.kfold = KFold(n_splits=n_splits) # KFold instance

    def split_data(self, data, labels, n_splits): pass
    def aggregate_metrics(self, results): pass
