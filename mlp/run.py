# Standard
import numpy as np
import pandas as pd

# Own modules
from config import Config_split, Config_encoder, Config_scaler, Config_mlmodel
from ml_module.eda_preprocessing import eda_preprocess
from ml_module.split import split_train_test
from ml_module.encode_normalise import encoder, scaler
from ml_module.mlmodel import compare_models, model_prediction

# Importing data and pre-processed based on EDA
dataset = eda_preprocess()

print('\nSplitting data into training and test set...')
# Split into X and y (training set and test set)
X_train, X_test, y_train, y_test = split_train_test(Config_split, dataset)
print('Done')

print('\nConducting data encoding...')
# Label encode each categorical feature
X_train, X_test = encoder(Config_encoder, X_train, X_test)
print('Done')

print('\nNormalize/Scale the data...')
# Scale the train and test set
X_train, X_test, _ = scaler(Config_scaler, X_train, X_test)
y_train, y_test, y_scaler = scaler(Config_scaler, y_train, y_test)
print('Done')

print('\nConducting model training on Training Set...')
# Train model based on configuration from Config.py
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')

model_performance, best_model = compare_models(Config_mlmodel, X_train, y_train)
print('Done')

print('\nPredicting results on Test Set...')
# Predict using trained model and display results
predictions = model_prediction(best_model, X_test, y_test, y_scaler)
print('Done')