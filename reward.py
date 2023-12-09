import numpy as np, os, os.path, sys, warnings
import pandas as pd
from sklearn.metrics import mean_squared_error

'''
this is for the online framework
each time it calculate the normalized utility score for the current patient only
hours2sepsis: true label
Y_upper: a list of the upper bound calculated by conformal prediction algorithm
Y_lower: a list of the lower bound calculated by conformal prediction algorithm
'''
# labels： "SepsisLabel" 

def standardize_reward(hours2sepsis, Y_upper, Y_lower):

    # get the RMSE
    Y_rmse_upper = mean_squared_error(hours2sepsis, Y_upper, squared=False)
    Y_rmse_lower = mean_squared_error(hours2sepsis, Y_lower, squared=False)
    rmse_min = 0
    # mean_value = np.mean(hours2sepsis)
    # rmse_max =  mean_squared_error(hours2sepsis, np.full_like(hours2sepsis, mean_value))
    rmse_max = 500
    standardize_rmse_upper = (Y_rmse_upper-rmse_min)/(rmse_max)
    standardize_rmse_lower = (Y_rmse_lower-rmse_min)/(rmse_max)
    reward_upper = max(1-standardize_rmse_upper,1-standardize_rmse_lower)
    reward_lower = min(1-standardize_rmse_upper,1-standardize_rmse_lower)
    return reward_lower, reward_upper
def get_UCB_LCB_avg(curr_pat_df, Y_upper, Y_lower):
    # I force Y in [0, 500], confirmed with Rishi, this is fine
    Y_upper = [500 if y >500 else y for y in Y_upper]
    Y_upper = [0 if y<0 else y for y in Y_upper]
    Y_lower = [500 if y >500 else y for y in Y_lower]
    Y_lower = [0 if y<0 else y for y in Y_lower] 
    labels  = curr_pat_df['SepsisLabel']
    hours2sepsis = curr_pat_df['hours2sepsis']
    if len(hours2sepsis) != len(labels):
        raise Exception('@@@@Numbers of predictions and labels must be the same.')
    for label in labels:
        if not label in (0, 1):
            raise Exception('@@@@Labels must satisfy label == 0 or label == 1.')
    LCB, UCB = standardize_reward(hours2sepsis,Y_upper, Y_lower)
    print(F'[LCB, UCB]: [{LCB},{UCB}]')
 
    return UCB, LCB
 
def get_absolute_error(curr_pat_df, Y_upper, Y_lower):
    # I force Y in [0, 500], confirmed with Rishi, this is fine
    Y_upper = [500 if y >500 else y for y in Y_upper]
    Y_upper = [0 if y<0 else y for y in Y_upper]
    Y_lower = [500 if y >500 else y for y in Y_lower]
    Y_lower = [0 if y<0 else y for y in Y_lower] 
    labels  = curr_pat_df['SepsisLabel']
    hours2sepsis = curr_pat_df['hours2sepsis']
    if len(hours2sepsis) != len(labels):
        raise Exception('@@@@Numbers of predictions and labels must be the same.')
    for label in labels:
        if not label in (0, 1):
            raise Exception('@@@@Labels must satisfy label == 0 or label == 1.')
            
    upper_array = np.array(Y_upper)
    lower_array = np.array(Y_lower)
    hours_array = np.array(hours2sepsis)
    
    upper_ae = abs(upper_array - hours_array)/500
    lower_ae = abs(lower_array - hours_array)/500
 
    return upper_ae, lower_ae