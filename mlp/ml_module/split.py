# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split

def split_train_test(Config, dataset):
    X = dataset.drop(Config.target_features, axis=1)
    y = dataset.loc[:, Config.target_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = Config.test_size, random_state = Config.random_state)

    return X_train, X_test, y_train, y_test