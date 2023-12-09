import time
import math
# import pickle
import utils_ECAD_journal as utils_ECAD
from scipy.stats import skew
import seaborn as sns
 

import matplotlib
# matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
# print("Switched to:",matplotlib.get_backend())
import os
import matplotlib

if os.environ.get('DISPLAY','') == '':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN   # kNN detector
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import utils_EnbPI_journal as util
from matplotlib.lines import Line2D  # For legend handles
import calendar
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import itertools
import importlib

import time
import pandas as pd
import numpy as np
import os
import sys
# import keras
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
# importlib.reload(sys.modules['PI_class_EnbPI_journal'])


class gap_bandit(object):
    def __init__(self, UCB, LCB, K):
        self.K = K
        self.UCB = UCB   
        self.LCB = LCB 
        self.s_value = [0]*self.K
        self.B_value = [-float('inf')]*self.K
    '''
    input results from Thompson sampling, namely, estimated_mu for
    each arm, time t, N_i(t),
    pull arm with largest UCB,
    output: update hat_mu,  N_i(t), and ConfBound
    ouput : index of the arm pulled
    get the reward, let the reward = estimated mean + uncertainty
    '''

    '''
    Updated gap-based bandit:
    f_tk(LOO): the fitted leave-one-out algorithm based on historical data
    F-1(1-alpha+hat_beta(t,k)):  the inverse empirical quantile function (ICDF) [quantile of the residuals]
    U = f_tk(LOO) + F-1(1-alpha+hat_beta(t,k))
    L = f_tk(LOO) + F-1(hat_beta(t,k))
    '''

    def pull_arm(self):
        for k in range(self.K):
            for a in range(self.K):
                if a==k:
                    continue # skip the case where a==k
                self.B_value[k]=max(self.B_value[k], self.UCB[a]-self.LCB[k])
        # python does not support substractiopn between list
        # self.s_value = self.UCB - self.LCB
        self.s_value = [ucb-lcb for ucb, lcb in zip(self.UCB, self.LCB)]
        Jt = np.argmin(self.B_value)
        jt = np.argmax(self.s_value)

        if self.s_value[Jt]>=self.s_value[jt]:
            pulled_arm = Jt
        else:
            pulled_arm = jt
        return pulled_arm
 


