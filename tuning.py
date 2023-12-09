import time
import math
# import pickle
# import utils_ECAD_journal as utils_ECAD
# from scipy.stats import skew
# import seaborn as sns
# import PI_class_EnbPI_journal as EnbPI
# import PI_class_EnbPI_multi as EnbPI
# import matplotlib
# matplotlib.use('TkAgg',force=True)
# matplotlib.use('Agg')
import os
# if os.environ.get('DISPLAY','') == '':
#     print('No display found. Using non-interactive Agg backend')
#     matplotlib.use('Agg')
# else:
#     matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
# print("Switched to:",matplotlib.get_backend())
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
# from pyod.models.pca import PCA
# from pyod.models.ocsvm import OCSVM
# from pyod.models.iforest import IForest
# from pyod.models.hbos import HBOS
# from pyod.models.knn import KNN   # kNN detector
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV

# from sklearn import neighbors
# from sklearn.neural_network import MLPClassifier
# from sklearn import svm
# import utils_EnbPI_journal as util
# from matplotlib.lines import Line2D  # For legend handles
# import calendar
# import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import MinMaxScaler
import itertools
import importlib
import time
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error
# from reward import get_UCB_LCB_avg
# from reward import get_absolute_error
# from gap_b import gap_bandit 
import sys
# import tensorflow as tf
# warnings.filterwarnings("ignore")
# importlib.reload(sys.modules['PI_class_EnbPI_journal'])
# import multiprocessing
# import dill
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# import json

def tuning(X, y, model_name):
    if model_name == 'xgb':
        params = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1]}
        model = XGBRegressor()
    elif model_name == 'svr':
        params = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.1, 1, 10],
            'gamma': [0.1, 1, 10]}
        model = SVR()
    elif model_name == 'ridge':
        min_alpha = 0.0001
        max_alpha = 10
        # do not specify cv, it defaults to Generalized Cross-Validation (GCV), 
        # which is a form of efficient Leave-One-Out cross-validation.
        model = RidgeCV(alphas=np.linspace(min_alpha, max_alpha, 10))
    elif model_name == 'lasso':
        # For LassoCV, the default is 5-fold cross-validation, 
        # which is generally a good balance between computational demand and validation robustnes
        model = LassoCV(alphas=np.linspace(min_alpha, max_alpha, 10), cv=5)
    # For LinearRegression, the focus is typically more on feature selection 
    # and data preprocessing rather than on hyperparameter tuning. 
    # elif model_name == 'linear':
    #     model = LinearRegression()
    #     params = {}     
    # search= RandomizedSearchCV(model, params, n_iter=100, 
    # scoring='neg_mean_squared_error', cv=3, verbose=2, random_state=42, n_jobs=-1)   
    # Measure the execution time
    start_time = time.time()
    if model_name == 'ridge' or model_name == 'lasso': 
        model.fit(X, y)
        best_alpha = model.alpha_
        best_score = model.best_score_
    else:
        search = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
        search.fit(X, y)
        best_params = search.best_params_
        best_score = search.best_score_
    end_time = time.time()

    # Extract information

    execution_time = end_time - start_time
    print('Model Name:', model_name)
    if model_name == 'ridge' or model_name == 'lasso': 
        print("Best Alpha:", best_alpha)
        results_df = pd.DataFrame(model.cv_results_)
    else:
        print("Best Parameters:", best_params)
        results_df = pd.DataFrame(search.cv_results_)
    print("Best Score:", best_score)
    print("Execution Time:", execution_time, "seconds")

    
    if not os.path.exists('./tuning_results'):
        # os.mkdirs('./tuning_results')
        os.makedirs('./tuning_results')
    results_df.to_csv(f'./tuning_results/{model_name}_tuning_results.csv', index=False)
if __name__ == '__main__':
    # num_test_pat = 500
    num_train_sepsis_pat = 1000
    num_train_nosepsis_pat = 3000
    # start_test = 0
    start_nosepsis_train = 0
    start_sepsis_train = 0
    # test_set = np.load('./Data/test_set.npy')
    # test_set =  test_set[start_test:start_test+num_test_pat]
    train_sepsis = np.load('./Data/train_sepsis.npy')
    train_nosepsis = np.load('./Data/train_nosepsis.npy')
    train_sepsis = train_sepsis[start_sepsis_train:start_sepsis_train+num_train_sepsis_pat]
    train_nosepsis = train_nosepsis[start_nosepsis_train:start_nosepsis_train+num_train_nosepsis_pat]
    sepsis_full = pd.read_csv('./Data/fully_imputed.csv')
    sepsis_full.drop(['HospAdmTime'], axis=1, inplace=True)
    train_sepis_df = sepsis_full[sepsis_full['pat_id'].isin(train_sepsis)]
    train_nosepis_df = sepsis_full[sepsis_full['pat_id'].isin(train_nosepsis)]
    # test_set_df = sepsis_full[sepsis_full['pat_id'].isin(test_set)]
    train_set_df = pd.concat([train_sepis_df, train_nosepis_df], ignore_index=True)
    # test_set_df = test_set_df.reset_index(drop=True)     
    train_set_df_x = train_set_df.drop(columns = ['pat_id','hours2sepsis'])
    train_set_df_y = train_set_df['hours2sepsis']
            # do not use inplace =True, otherwise train_set_df_x/y and test_set_df_x/y will become nonetype
            # then we cannot use to_numpy()
            # the original training dataset before experts selection
    X_train = train_set_df_x.to_numpy(dtype='float', na_value=np.nan)
    Y_train = train_set_df_y.to_numpy(dtype='float', na_value=np.nan)

    for expert in [ 'xgb','svr']:
        tuning(X_train, Y_train, expert)