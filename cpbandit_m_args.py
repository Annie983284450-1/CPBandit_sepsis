import time
import math
import pickle
# import utils_ECAD_journal as utils_ECAD
from scipy.stats import skew
import seaborn as sns
# import PI_class_EnbPI_journal as EnbPI
import PI_Sepsysolcp as EnbPI
import matplotlib
# matplotlib.use('TkAgg',force=True)
# matplotlib.use('Agg')
import os
if os.environ.get('DISPLAY','') == '':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
print("Switched to:",matplotlib.get_backend())
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN   # kNN detector
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
# from sklearn import svm
import utils_Sepsysolcp as util
from matplotlib.lines import Line2D  # For legend handles
# import calendar
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
 
from sklearn.preprocessing import MinMaxScaler
import itertools
import importlib
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from reward import get_UCB_LCB_avg
from reward import get_absolute_error
from gap_b import gap_bandit 
import sys
# this will suppress all the error messages
# be cautious
# stderr = sys.stderr
# sys.stderr = open('logfile.log','w')
import tensorflow as tf
# sys.stderr = stderr
# tf.get_logger().setLevel('ERROR')
# # warnings.filterwarnings("ignore")
# # importlib.reload(sys.modules['PI_class_EnbPI_journal'])

import multiprocessing
import dill

multiprocessing.get_context().Process().Pickle = dill
# =============Read data and initialize parameters
class CPBandit:
    def __init__(self, experts):
        self.experts = experts
        self.k = len(experts)
        #maintain a list of upper bound value and a lower bound value for each arm
        self.UCB_avg = [0]*self.k
        self.LCB_avg = [0]*self.k
        # Reward is the utilty function, i.e., 1-RMSE
        self.rewards = []

    def _start_game(self):
        '''
        num_test_pat = 10  
        num_train_sepsis_pat = 50 
        num_train_nosepsis_pat = 150 
        Total excution time ---  5 seconds ---
I
        num_test_pat = 100  
        num_train_sepsis_pat = 500 
        num_train_nosepsis_pat = 1500 
        Total excution time --- 2472.87833571434 seconds ---
        
        the time complexity is too high. So we need to do the refitting less frequently
        '''
        # read sepsis dataset
        num_test_pat = 1000
        num_train_sepsis_pat = 2000
        num_train_nosepsis_pat = 4000

        start_test = 0
        start_nosepsis_train = 0
        start_sepsis_train = 0

        test_set = np.load('../cpbanditsepsis_experiements//Data/test_set.npy')
        test_set =  test_set[start_test:start_test+num_test_pat]

        train_sepsis = np.load('../cpbanditsepsis_experiements//Data/train_sepsis.npy')
        train_nosepsis = np.load('../cpbanditsepsis_experiements//Data/train_nosepsis.npy')
        train_sepsis = train_sepsis[start_sepsis_train:start_sepsis_train+num_train_sepsis_pat]
        train_nosepsis = train_nosepsis[start_nosepsis_train:start_nosepsis_train+num_train_nosepsis_pat]



        sepsis_full = pd.read_csv('../cpbanditsepsis_experiements/Data/fully_imputed.csv')
        sepsis_full.drop(['HospAdmTime'], axis=1, inplace=True)

        final_result_path='../cpbanditsepsis_experiements/hyperparameters_tuned'+'/Results'+'('+f'test{num_test_pat},train{num_train_sepsis_pat+num_train_nosepsis_pat}'+str(self.experts)+')'
        if not os.path.exists(final_result_path):
            os.makedirs(final_result_path)
        if start_test !=0:
            X_train = np.load(final_result_path +'./X_train_merged.npy', X_train_merged)
            Y_train = np.load(final_result_path + '/Y_train_merged.npy', Y_train_merged)
        else:
            train_sepis_df = sepsis_full[sepsis_full['pat_id'].isin(train_sepsis)]
            train_nosepis_df = sepsis_full[sepsis_full['pat_id'].isin(train_nosepsis)]
            test_set_df = sepsis_full[sepsis_full['pat_id'].isin(test_set)]
            train_set_df = pd.concat([train_sepis_df, train_nosepis_df], ignore_index=True)

            test_set_df = test_set_df.reset_index(drop=True)     
            train_set_df_x = train_set_df.drop(columns = ['pat_id','hours2sepsis'])
            train_set_df_y = train_set_df['hours2sepsis']
            # do not use inplace =True, otherwise train_set_df_x/y and test_set_df_x/y will become nonetype
            # then we cannot use to_numpy()
            # the original training dataset before experts selection
            X_train = train_set_df_x.to_numpy(dtype='float', na_value=np.nan)
            Y_train = train_set_df_y.to_numpy(dtype='float', na_value=np.nan)


        # # ==================Getting the conformal intervals......===================
        # initialze parameters
        data_name = 'physionet_sepsis'
        stride = 1
        miss_test_idx=[]
        tot_trial = 1
        # usually B=30 can make sure every sample have LOO residual
        B = 25 
        K = len(self.experts)
        alpha=0.1
        alpha_ls = np.linspace(0.05,0.25,5)
        # alpha_ls = [0.1]
        min_alpha = 0.0001
        max_alpha = 10

        if not os.path.exists(final_result_path+'/dats'):
            os.makedirs(final_result_path+'/dats')
        if not os.path.exists(final_result_path+'/imgs'):
            os.makedirs(final_result_path+'/imgs')
        f_name = ''
        for i, expert in enumerate(self.experts):
            if i==0:
                f_name = f_name+expert
            else:
                f_name = f_name+'_'+expert
        f_dat_path = os.path.join(final_result_path+'/dats',f_name)
        f_img_path = os.path.join(final_result_path+'/imgs', f_name)
        if not os.path.exists(f_dat_path):
            os.makedirs(f_dat_path)
        if not os.path.exists(f_img_path):
            os.makedirs(f_img_path)

        methods  = ['Ensemble'] # conformal prediction methods
        
        start_time = time.time()
 
        num_pat_tested = 0
        num_fitting = 0
        refit_step = 100
        X_size = {}
        
        expert_idx = list(range(K))
        expert_dict = dict(zip(self.experts, expert_idx))
        print(f'expert dict: {expert_dict}')
     
        interval_namelist = [x+'_interval' for x in self.experts]
        coverage_namelist = [x+'_coverage' for x in self.experts]
        regret_namelist = [x+'_regret' for x in self.experts]
        final_columns = ['patient_id', 'alpha', 'itrial', 'method']
        final_columns.extend(interval_namelist)
        final_columns.extend(coverage_namelist)
        final_columns.extend(regret_namelist)
        final_columns.append('winner')
        print(f'final columns: {final_columns}')

        final_all_results_avg = pd.DataFrame(columns=final_columns)
        
        rmse_min = 0
        rmse_max = 500
                                                                
         
        predictions_namelist = [x+'_predictions' for x in self.experts]
        predictions_col = ['pat_id','itrial','hours2sepsis']
        predictions_col.extend(predictions_namelist)
        # predictions_col =['pat_id','itrial', 'ridge_predictions', 'rf_predictions', 'nn_predictions',  'hours2sepsis']        
        print(f'predictions_col: {predictions_col}')

        UCB_namelist = ['UCB_'+x for x in self.experts]
        LCB_namelist = ['LCB_'+x  for x in self.experts]
        upper_namelist = ['upper_'+x for x in self.experts]
        lower_namelist = ['lower_'+x for x in self.experts]

        cp_col = ['patient_id', 'alpha', 'itrial', 'method']
        cp_col.extend(UCB_namelist)
        cp_col.extend(LCB_namelist)
        cp_col.extend(predictions_namelist)
        cp_col.extend(upper_namelist)
        cp_col.extend(lower_namelist)
        cp_col.append('winner')
        cp_col.append('pulled_idx')
        print(f'cp_col: {cp_col}')

        if 'ridge' in self.experts:
            ridge_f = RidgeCV(alphas=np.linspace(min_alpha, max_alpha, 10))
        if 'lasso' in self.experts:
            lasso_f = LassoCV(alphas=np.linspace(min_alpha, max_alpha, 10))
        if 'rf' in self.experts:
            rf_hyperparams = {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 10}
            # rf_hyperparams = {'n_estimators':10, 'criterion':'mse','bootstrap': False, 'max_depth': 2}

            rf_f = RandomForestRegressor(n_estimators=100, min_samples_split = 2, min_samples_leaf = 4, criterion='mse',  max_features = 'sqrt' , max_depth=10, n_jobs=-1)
        # if 'svr' in self.experts:
        #     svr_f = SVR()
        if 'xgb' in self.experts:
            xgb_hyperparams = {'subsample': 0.8, 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
            xgb_f = XGBRegressor(subsample=xgb_hyperparams['subsample'],
                                                n_estimators=xgb_hyperparams['n_estimators'],
                                                max_depth=xgb_hyperparams['max_depth'],
                                                learning_rate=xgb_hyperparams['learning_rate'])
        for patient_id in test_set:
            start_curr_pat_time =  time.time()
            print('\n\n')
            print(f'=======Processing patient {num_pat_tested}th patient: {patient_id}====================')
            # Note: If there are fewer than 48 rows with the desired value, the result will contain all rows with that value.
            # only take the data from the first 48 hours
            curr_pat_df = sepsis_full[sepsis_full['pat_id']==patient_id].head(48)
            X_predict = curr_pat_df.drop(columns=['pat_id','hours2sepsis'])
            Y_predict = curr_pat_df['hours2sepsis']
            # curr_pat_histories['hours2sepsis'] = Y_predict                   
            if num_pat_tested % refit_step==0:
                Isrefit = True
                if num_pat_tested!=0:
                    X_size['Old_X_Size'] = X_train.shape[0]
                    X_train = X_train_merged
                    Y_train = Y_train_merged
                    X_size['New_X_Size'] = X_train.shape[0]
                    print(f'Training Dataset Updated!!!!!!!!!!!!!!!!!!!!!')
                    print(f'{X_size}')
                if start_test!=0:
                    Isrefit = False
            else:
                Isrefit = False            
            for itrial in range(tot_trial):
                curr_pat_predictions = pd.DataFrame(columns=predictions_col)
                np.random.seed(98765+itrial)
                # we will only do refitting every 100 patients
                # if 'nn' in self.experts:
                #     nnet = util.keras_mod()
                # nnet is not picklable
                # try:
                #     pickle.dumps(nnet)
                #     print("nnet is picklable")
                # except Exception as e:
                #     print("nnet is not picklable:", str(e))  
                    # nn_cp = EnbPI.prediction_interval(
                    #     nnet,  X_train, X_predict, Y_train, Y_predict)
                    # nn_cp.fit_bootstrap_models_online_multi(B, miss_test_idx,  Isrefit, model_name = 'nn')
                    # nn_predictions = nn_cp.Ensemble_pred_interval_centers
                    # curr_pat_predictions['nn_predictions'] = nn_predictions
                    # nn_rmse = mean_squared_error(Y_predict, nn_predictions, squared=False)
                    # standardize_nn_rmse = (nn_rmse-rmse_min)/(rmse_max)
                    
                        # rnnet = util.keras_rnn()
                        # fit_bootstrap_models_xx() contructs f^b and LOO
                standardize_rmse_dict = {}
                cp_EnbPI_dict = {}
                for expert in self.experts:
                    cp_EnbPI = EnbPI.prediction_interval(locals()[f'{expert}_f'], X_train, X_predict, Y_train, Y_predict,final_result_path)
                    cp_EnbPI.fit_bootstrap_models_online_multi(B, miss_test_idx, Isrefit, model_name = expert)
                    predictions = cp_EnbPI.Ensemble_pred_interval_centers
                    curr_pat_predictions[f'{expert}_predictions'] = predictions 
                    rmse = mean_squared_error(Y_predict, predictions, squared=False)
                    standardize_rmse_dict[f'{expert}'] = (rmse-rmse_min)/(rmse_max-rmse_min)
                    cp_EnbPI_dict[f'{expert}'] = cp_EnbPI
                    
                curr_pat_predictions['itrial'] = itrial
                curr_pat_predictions['pat_id'] = patient_id
                curr_pat_predictions['hours2sepsis'] = Y_predict
                histories_dat_path = f_dat_path + '/all_histories/' + patient_id  +'/itrial#'+str(itrial)
                if not os.path.exists(histories_dat_path):
                    os.makedirs(histories_dat_path)
                curr_pat_predictions.to_csv(f'{histories_dat_path}/predictions.csv')
                # cp_col =['itrial','alpha', 'method', 'UCB_ridge', 'LCB_ridge','UCB_rf', 'LCB_rf','UCB_nn', 'LCB_nn',
                #                                          'upper_ridge', 'lower_ridge','upper_rf', 'lower_rf','upper_nn', 'lower_nn',
                #                                          'ridge_predictions', 'rf_predictions', 'nn_predictions',
                #                                          'winner','pulled_idx']
                for alpha in alpha_ls:

                    print(f'~~~~~~~~~~At trial # {itrial} and alpha={alpha}~~~~~~~~~~~~~~')
                    for method in methods:
                        cp_dat_path = histories_dat_path + '/' + method
                        if not os.path.exists(cp_dat_path):
                            os.makedirs(cp_dat_path)
                        curr_pat_cp = pd.DataFrame(columns=cp_col)
                        LCB_dict = {}
                        UCB_dict = {}
                        coverage_dict = {}
                        regret_dict = {}
                        upper_ae_dict = {}
                        lower_ae_dict = {}
                        new_row_all_avg = {'patient_id': patient_id, 'alpha': alpha, 'itrial': itrial, 'method': method}
                        if method == 'Ensemble':
                            for expert in self.experts:
                                curr_pat_cp[f'{expert}_predictions'] = curr_pat_predictions[f'{expert}_predictions']
                                PIs_df, results = cp_EnbPI_dict[f'{expert}'].run_experiments(alpha, stride, data_name, itrial,
                                                    true_Y_predict=[], get_plots=False, none_CP=False, methods=methods)
                                print(results)
                                coverage_dict[expert] = results.mean_coverage.values[0]
                                Y_upper = PIs_df['upper']
                                Y_lower = PIs_df['lower']
                                curr_pat_cp[f'upper_{expert}'] = Y_upper
                                curr_pat_cp[f'lower_{expert}'] = Y_lower

                                upper_ae, lower_ae = get_absolute_error(curr_pat_df, Y_upper, Y_lower)
                                upper_ae_dict[f'{expert}'] = upper_ae
                                lower_ae_dict[f'{expert}'] = lower_ae

                                UCB_avg, LCB_avg = get_UCB_LCB_avg(curr_pat_df, Y_upper, Y_lower)
                                UCB_dict[f'{expert}'] = UCB_avg
                                LCB_dict[f'{expert}'] = LCB_avg
                                print(f'[LCB_{expert}_avg, UCB_{expert}_avg]: [{LCB_avg}, {UCB_avg}]')
                                k_idx = expert_dict[f'{expert}']
                                self.UCB_avg[k_idx] = UCB_avg
                                self.LCB_avg[k_idx] = LCB_avg
                        curr_pat_cp['patient_id'] = patient_id
                        curr_pat_cp['itrial'] = itrial
                        curr_pat_cp['method'] = method
                        curr_pat_cp['alpha'] = alpha
                        print(f'Selecting the best expert on average @@@@~~~')
                        pulled_arm_idx_avg = gap_bandit(self.UCB_avg, self.LCB_avg, self.k).pull_arm()
                        for expert in self.experts:
                            #the larger the standardized_rmse, the larger the regret
                            # regret_dict[f'{expert}'] = 1 - standardize_rmse_dict[f'{expert}']
                            regret_dict[f'{expert}'] = standardize_rmse_dict[f'{expert}']
                            new_row_all_avg[f'{expert}_interval'] = (LCB_dict[expert], UCB_dict[expert])
                            new_row_all_avg[f'{expert}_coverage'] = coverage_dict[expert]
                            new_row_all_avg[f'{expert}_regret'] = regret_dict[expert]

                        new_row_all_avg['winner_avg'] = list(expert_dict.keys())[pulled_arm_idx_avg]
                        final_all_results_avg = final_all_results_avg.append(new_row_all_avg, ignore_index=True)
                        print(f'Selecting the best expert on hourly basis @@@@~~~')
                        UCB_tmp = [0]*self.k
                        LCB_tmp = [0]*self.k
                        pulled_arms = []
                        selected_experts = []           
                        for m in range(len(Y_predict)):
                            for expert in self.experts:
                                k_idx = expert_dict[expert]
                                upper_ae_k = upper_ae_dict[f'{expert}'] 
                                lower_ae_k = lower_ae_dict[f'{expert}']
                                UCB_tmp[k_idx] = max(1-upper_ae_k[m],1-upper_ae_k[m])
                                LCB_tmp[k_idx] = min(1-upper_ae_k[m],1-upper_ae_k[m])
                                curr_pat_cp.loc[m, f'UCB_{expert}'] = UCB_tmp[k_idx]
                                curr_pat_cp.loc[m, f'LCB_{expert}'] = LCB_tmp[k_idx]
                            pulled_arm_idx = gap_bandit(UCB_tmp, LCB_tmp, self.k).pull_arm() 
                            selected_expert = list(expert_dict.keys())[pulled_arm_idx]
                            pulled_arms.append(pulled_arm_idx)
                            selected_experts.append(selected_expert)
                        curr_pat_cp['winner'] = selected_experts
                        curr_pat_cp['pulled_idx'] = pulled_arms
                        curr_pat_cp.to_csv(f'{cp_dat_path}/alpha={alpha}.csv')
            if Isrefit:
                num_fitting = num_fitting+1            
            # updating dataset
            print(f'# {num_pat_tested} patients already tested! ......')
            print(f'\n')
            print(f'-------------------------------------------')
            print(f'Updating the training dataset ..........')
            X_train_new_df = curr_pat_df.drop(columns = ['pat_id','hours2sepsis'])
            # train_set_df_x.append(X_train_new_df, ignore_index = True)
            Y_train_new_df = curr_pat_df['hours2sepsis']
            X_train_new = X_train_new_df.to_numpy(dtype='float', na_value=np.nan)
            Y_train_new = Y_train_new_df.to_numpy(dtype='float', na_value=np.nan)
            if num_pat_tested ==0:
                X_train_merged = np.append(X_train,X_train_new,axis=0)
                Y_train_merged = np.append(Y_train,Y_train_new,axis=0)
            else:
                X_train_merged = np.append(X_train_merged,X_train_new,axis=0)
                Y_train_merged = np.append(Y_train_merged,Y_train_new,axis=0)
            # dataset updated
            print(f'~~~~~~Excution time for # {patient_id}: {time.time()-start_curr_pat_time} seconds~~~~~~')
            print('\n\n')
            print('========================================================')
            num_pat_tested = num_pat_tested + 1
         
        np.save(final_result_path+'/X_train_merged.npy', X_train_merged)
        np.save(final_result_path+'/Y_train_merged.npy', Y_train_merged)
        final_all_results_avg.to_csv(final_result_path+'/final_all_results_avg.csv')
        print('========================================================')
        print('========================================================')
        print(f'Test size: {len(test_set)}')
      
        print(f'Total excution time: {(time.time() - start_time)} seconds~~~~~~' )
        machine = 'ece-kl2313-01.ece.gatech.edu'
        with open(final_result_path+'/execution_info.txt', 'w') as file:
            file.write(f'Total excution time: {(time.time() - start_time)} seconds\n')
            file.write(f'num_test_pat = {num_test_pat}\n')
            file.write(f'num_train_sepsis_pat = {num_train_sepsis_pat}\n')
            file.write(f'num_train_nosepsis_pat = {num_train_nosepsis_pat}\n')
            file.write(f'start_test = {start_test}\n')
            file.write(f'start_nosepsis_train = {start_nosepsis_train}\n')
            file.write(f'start_sepsis_train = {start_sepsis_train}\n')
            file.write(f'tot_trial = {tot_trial}\n')
            file.write(f'refit_step = {refit_step}\n') 
            file.write(f'No of experts = {self.k}\n')
            file.write(f'Experts = {list(expert_dict.keys())}\n')
            file.write(f'The models have been fitted for {num_fitting} times.\n')
            file.write(f'Machine: {machine}\n')
            file.write(f'Multiprocessor: True\n')
            if 'xgb' in self.experts:
                file.write(f'xgb hyperparameters: {xgb_hyperparams}\n')
            if 'rf' in self.experts:
                file.write(f'rf hyperparameters: {rf_hyperparams}\n')
             
        print(f'The models have been fitted for {num_fitting} times.')
        print('========================================================')

import argparse

def main():

     
    parser = argparse.ArgumentParser(description='Run CPBandit with a list of experts.')
    parser.add_argument('experts_list', nargs='+', help='List of expert names')
    args = parser.parse_args()
    cpbanit_player = CPBandit(experts=args.experts_list)
    cpbanit_player._start_game()

if __name__=='__main__':
    main()
 