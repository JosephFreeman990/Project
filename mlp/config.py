# -*- coding: utf-8 -*-

class Config_split:
    target_features = ['registered-users', 'guest-users']
    test_size = 0.3 
    random_state = 0
    
class Config_encoder:
    '''
    possible encoders
    'LabelEncoder' (Range 0 to n)
    'OneHotEncoder' (Creates one column for each value)
    special thanks to: 'https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159'
    '''
    encoder = 'OneHotEncoder' #Select only 1 encoder

class Config_scaler: 
    '''
    possible scalers
    'MinMaxScaler' (Scale Range is 0 to 1, susceptible to outlier)
    'StandardScaler' (Mean 0, SD 1, susceptible to outlier)
    special thanks to: 'https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02'
    
    '''
    scaler = 'MinMaxScaler' #Select only 1 scaler

class Config_mlmodel:
    '''
    possible models:
    'XGBRegression', 'LinearRegression', 'LassoRegression', 'RidgeRegression'
    '''
    # Select your model
    models = ['LinearRegression', 'LassoRegression', 'RidgeRegression', 'XGBRegression']
    
    # Configre your general parameters here
    cv = 10
    n_jobs = -1
    scoring = 'r2' #Explained Variation/Total Variation
    
    # Configure your model specific parameters here  
    linearparameters = {'estimator__fit_intercept':[True, False], 
                       }
    #Regularization, prevents overfitting (significantly reduces the variance of the model, without substantial increase in its bias)
    lassoparameters = {'estimator__alpha':[0.01, 0.02, 0.03], # Increasing it increases regularization, using slope
                      }
    
    ridgeparameters = {'estimator__alpha':[2.1, 2.2, 2.3, 2.4] # Increasing it increases regularization, using slope + weight of all features except y-intercept
                      }
    # Gradient-boosted decision Tree
    # Obj Function: Training Loss (How well model fit training data)
    #             : Regularization
    # Tree boosting by additive training - Add Trees that optimise the Obj Function
    xgbparameters = {'estimator__objective':['reg:squarederror'],
                     'estimator__tree_method':['exact'], # construction algorithm, 'exact' for greedy, 'approx' for approximate
                     'estimator__early_stopping_rounds':[40], # prevents overfitting
                     'estimator__max_depth':[6], # Maximum depth of tree, increasing it increases model complexity
                     'estimator__learning_rate':[0.03], # learning rate
                     'estimator__n_estimators':[1500], # Number of decision trees to plot
                     'estimator__alpha':[0], # L1 Regularizaation, increase to make model more conservative
                     'estimator__min_child_weight':[3], # Min sum of instance needed in a child
                     'estimator__subsample':[0.8], # Randomly sample X of the training data prior to growing trees
                     }