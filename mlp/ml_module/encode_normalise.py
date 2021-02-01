# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Extract categorical features' header
def extract_categorical_header(dataset_train):
    data_type = ['object', 'category']
    extracted_header = []
    
    for i in data_type:
        header = dataset_train.dtypes[dataset_train.dtypes==i]
        header = list(header.index)
        extracted_header += header
    return extracted_header

# Label encode using categorical features' header
def label_encoder(dataset_train, dataset_test, categorical_header):
    le = LabelEncoder()
    for header in categorical_header:
        dataset_train.loc[:, header] = le.fit_transform(dataset_train.loc[:, header])
        dataset_test.loc[:, header] = le.transform(dataset_test.loc[:, header])
    return dataset_train, dataset_test

# Onehot encode
def onehot_encoder(dataset_train, dataset_test, categorical_header):
    oe = OneHotEncoder(drop='first')
    ct = ColumnTransformer([('one_hot_encoder', oe, categorical_header)],
                            remainder = 'passthrough')
    dataset_train = ct.fit_transform(dataset_train)
    dataset_test = ct.transform(dataset_test)
    return dataset_train, dataset_test

# Scale dataset with standard scaler
def standard_scaler(dataset_train, dataset_test):
    ss= StandardScaler()
    dataset_train = ss.fit_transform(dataset_train)
    dataset_test = ss.transform(dataset_test)
    return dataset_train, dataset_test, ss

def minmax_scaler(dataset_train, dataset_test):
    ms= MinMaxScaler()
    dataset_train = ms.fit_transform(dataset_train)
    dataset_test = ms.transform(dataset_test)
    return dataset_train, dataset_test, ms

def encoder(config, dataset_train, dataset_test):
    categorical_header = extract_categorical_header(dataset_train)
    
    if config.encoder == 'LabelEncoder':
        dataset_train, dataset_test = label_encoder(dataset_train, dataset_test, categorical_header)
    elif config.encoder == 'OneHotEncoder':
        dataset_train, dataset_test = onehot_encoder(dataset_train, dataset_test, categorical_header)
    else:
        print('Incorrect encoder name in Config, please check: ' + config.encoder)
    return dataset_train, dataset_test

def scaler(config, dataset_train, dataset_test):
    if config.scaler == 'StandardScaler':
        dataset_train, dataset_test, dataset_scaler = standard_scaler(dataset_train, dataset_test)
    elif config.scaler == 'MinMaxScaler':
        dataset_train, dataset_test, dataset_scaler = minmax_scaler(dataset_train, dataset_test)
    return dataset_train, dataset_test, dataset_scaler


