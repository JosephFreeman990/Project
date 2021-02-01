# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# Machine Learning Models
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.multioutput import MultiOutputRegressor

# Model Evaluation Metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

def compare_models(Config, X_train, y_train):
    best_score = -9999999
    results = []
    for model in Config.models:
        if model == 'XGBRegression':
            xgb = GridSearchCV(estimator=MultiOutputRegressor(XGBRegressor()), 
                               param_grid=Config.xgbparameters,
                               cv=Config.cv,
                               n_jobs=Config.n_jobs,
                               scoring=Config.scoring,
                               refit=True)
            xgb.fit(X_train, y_train)
            results.append([model, xgb.best_score_, xgb.best_params_])
            if best_score < xgb.best_score_:
                best_score = xgb.best_score_
                best_model = xgb
                
        elif model == 'LinearRegression':
            linear = GridSearchCV(estimator=MultiOutputRegressor(LinearRegression()),
                                  param_grid=Config.linearparameters,
                                  cv=Config.cv,
                                  n_jobs=Config.n_jobs,
                                  scoring=Config.scoring,
                                  refit=True)
            linear.fit(X_train, y_train)
            results.append([model, linear.best_score_, linear.best_params_])
            if best_score < linear.best_score_:
                best_score = linear.best_score_
                best_model = linear
        
        elif model == 'LassoRegression':
            lasso = GridSearchCV(estimator=MultiOutputRegressor(Lasso()),
                                  param_grid=Config.lassoparameters,
                                  cv=Config.cv,
                                  n_jobs=Config.n_jobs,
                                  scoring=Config.scoring,
                                  refit=True)
            lasso.fit(X_train, y_train)
            results.append([model, lasso.best_score_, lasso.best_params_])
            if best_score < lasso.best_score_:
                best_score = lasso.best_score_
                best_model = lasso
                
        elif model == 'RidgeRegression':
            ridge = GridSearchCV(estimator=MultiOutputRegressor(Ridge()),
                                  param_grid=Config.ridgeparameters,
                                  cv=Config.cv,
                                  n_jobs=Config.n_jobs,
                                  scoring=Config.scoring,
                                  refit=True)
            ridge.fit(X_train, y_train)
            results.append([model, ridge.best_score_, ridge.best_params_]) 
            if best_score < ridge.best_score_:
                best_score = ridge.best_score_
                best_model = ridge
                
        else:
            print('Incorrect model name in Config, please check: ' + model)
        
    results = pd.DataFrame(results, columns=['model', 'mean_cv_r2', 'best_param'])
    
    print ("\n" + "\033[1m" + "Model Report" + "\033[0m")
    print(results.to_string(index=False))
        
    return results, best_model
    
def model_prediction(model, X_test, y_test, y_test_scaler):
    y_pred = model.predict(X_test)
    y_pred = y_test_scaler.inverse_transform(y_pred).astype(int)
    y_test = y_test_scaler.inverse_transform(y_test).astype(int)
    
    R_squared = r2_score(y_test, y_pred)
    RMSE = mean_squared_error(y_test, y_pred)
    
    print ("\n" + "\033[1m" + "Prediction Report" + "\033[0m")
    print("R-squared : " + str(R_squared))
    print("RMSE: " + str(RMSE))
    
    # Print predictions
    total_test = np.add(y_test[:,0],y_test[:,1])
    total_pred = np.add(y_pred[:,0],y_pred[:,1])
    predictions = np.concatenate([total_test.reshape(-1,1), total_pred.reshape(-1,1)], axis=1)
    predictions = pd.DataFrame(predictions, columns=['total_test', 'total_pred'])  

    print ("\n" + "\033[1m"+ "First 30 predictions" + "\033[0m")
    print(predictions.head(30))
    
    return predictions
