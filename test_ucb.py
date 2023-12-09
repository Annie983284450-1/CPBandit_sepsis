
import numpy as np
import argparse
import math
import random
import matplotlib.pyplot as plt
import time
import sys
import os
from arm import arm
from gap_b import UCB
#from posterior import posterior
import math
import random
import pandas as pd



## add the relative directory to path to make module load easier
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curPath, "Candidate_rf_svc_lr/", "code/"))
sys.path.append(os.path.join(curPath, "Candidate_xgb/"))

## load self defined modules
#from Candidate_xgb import test_expert1
import test_xgb
import test_random_forest_svc

'''
init and start game for one round
'''
class MAB:
    #def __init__(self,expert_list, alpha, T):
    def __init__(self , alpha, T, experts_name_list ):
        '''
        parameters required by class UCB
        '''
        self.alpha = alpha  # UCB confidence interval parameter
        self.experts_name_list  = experts_name_list

        ### five experts
        #self.experts_name_list  = ['RandomGuess',   'XGB',          'RandomForest', 'LR' ,   'SVC' ]

        ### four experts
        #self.experts_name_list  = ['RandomGuess',   'XGB',          'RandomForest', 'LR'  ]
        #self.experts_name_list  = ['RandomGuess',   'XGB',          'RandomForest', 'SVC' ]
        #self.experts_name_list  = ['RandomGuess',   'XGB',          'LR' ,          'SVC' ]
        #self.experts_name_list  = ['RandomGuess',   'RandomForest', 'LR' ,          'SVC' ]
        #self.experts_name_list  = ['XGB',           'RandomForest', 'LR' ,          'SVC' ]

        ### three experts
        #self.experts_name_list  = ['RandomForest',  'LR' ,          'SVC' ]
        #self.experts_name_list  = ['XGB',           'LR' ,          'SVC' ]
        #self.experts_name_list  = ['XGB',           'RandomForest', 'SVC' ]
        #self.experts_name_list  = ['XGB',           'RandomForest', 'LR'   ]
        #self.experts_name_list  = ['RandomGuess',   'LR' ,          'SVC' ]
        #self.experts_name_list  = ['RandomGuess',   'RandomForest', 'SVC' ]
        #self.experts_name_list  = ['RandomGuess' ,  'RandomForest', 'LR'   ]
        #self.experts_name_list  = ['RandomGuess',   'XGB' ,         'SVC' ]
        #self.experts_name_list  = ['RandomGuess',   'XGB',          'LR'   ]
        #self.experts_name_list  = ['RandomGuess',   'XGB',          'RandomForest'  ]

        ## two experts

        #self.experts_name_list  = ['RandomGuess',  'XGB'  ]
        #self.experts_name_list  = ['RandomGuess',  'RandomForest' ]
        #self.experts_name_list  = ['RandomGuess',  'LR'  ]
        #self.experts_name_list  = ['RandomGuess',  'SVC' ]
        #self.experts_name_list  = ['XGB',          'RandomForest' ]
        #self.experts_name_list  = ['XGB',          'LR'   ]
        #self.experts_name_list  = ['XGB',          'SVC' ]
        #self.experts_name_list  = ['RandomForest', 'LR'  ]
        #self.experts_name_list  = ['RandomForest', 'SVC' ]
        #self.experts_name_list  = ['LR' ,          'SVC' ]


        ### four experts
        #self.experts_name_list  = ['RandomGuess','XGB', 'RandomForest', 'SVC'  ]

        ### three experts
        #self.experts_name_list  = ['RandomGuess','XGB', 'SVC'  ]
        #self.experts_name_list = ['RandomGuess','RandomForest', 'SVC']
        #self.experts_name_list = ['RandomGuess','XGB', 'RandomForest'  ]
        #self.experts_name_list  = ['XGB', 'RandomForest', 'SVC'  ]


        ## two experts

        #self.experts_name_list  = ['RandomGuess','XGB'   ]
        #self.experts_name_list  = ['RandomGuess', 'RandomForest'  ]
        #self.experts_name_list  = ['RandomGuess' , 'SVC'  ]
        #self.experts_name_list  = ['XGB', 'RandomForest'  ]
        #self.experts_name_list  = [ 'XGB', 'SVC'  ]
        #self.experts_name_list  = ['RandomForest', 'SVC'  ]
        # one experts

        self.experts_name_list  = ['XGB']

        self.k = len(self.experts_name_list)
        # each arm needs to keep track of the estimated value at each time step t
        #self.hat_mu_list = [0]*self.k  # list of currentl round of hat_mu
        self.hat_mu_list = np.zeros(self.k)  # list of currentl round of hat_mu
        self.N = [0]*self.k  # N_i(t) (i=1,2,3,...k), cumulated # times arm i got pulled
        self.UCB = [0]*self.k # maintain a list of UCB values of all arms at time t

        # parameters added for class MAB
        self.T = T # theno. of rounds will be input by the user

        self.rewards = list()    #[0]*self.T

        self.mu_best = 1
        # regret is recorded at each round, stored in a list
        # so that we could add them together in the end
        self.regrets = list()  #[0]*self.T  # total regrets after each round
        self.cul_regrets = list()  #[0]*self.T  # total regrets after each round
        # a (T X k) matrix, at each round t, we need to have k N_i(t)
        # to store the number of times the arm has been selected
        self.N_matrix=np.zeros((self.T,self.k))
        self.arm_history = np.zeros( self.T )
        self.all_utilities =   np.zeros( self.T )
        self.expert_name_history = list()

    def _start_game(self ):
    # each round t, we have one coming patient
        f_name = ''
        if not os.path.exists('./dat'):
            os.mkdir('./dat')
        if not os.path.exists('imgs'):
            os.mkdir('./imgs')
        for i, expert_name in enumerate(self.experts_name_list ):
            if i==0:
                f_name = f_name + expert_name
            else:
                f_name = f_name +'_'+ expert_name
        f_dat_path = os.path.join('./dat/',f_name )
        f_imgs_path = os.path.join('./imgs/',f_name )

        if not os.path.exists(f_dat_path):
                os.mkdir(f_dat_path)
        if not os.path.exists(f_imgs_path):
            os.mkdir( f_imgs_path)
        test = pd.read_csv('test_set_all_filled.csv')
        psv_names =  np.sort(np.load('./Candidate_xgb/data/test_set.npy'))
        expert_name_history =list()
        hat_mu_list = np.zeros(self.k)
        arm_history=  np.zeros(self.T)
        all_utilities = np.zeros(self.T)
        current_reward = np.zeros(self.k)
        psv_list = []

        for t in np.arange(self.T):
            psv_list.append(psv_names[t])
            # hat_mu_list = list() # for all estimated mus
            patient_i = t
            # var_list = list()
            print("Round ****************",t,"******************")

            ## fill in the NaN values
            X_testi = test.loc[test['Patient_id'] == patient_i]
            X  = X_testi.drop(['Patient_id','SepsisLabel'] , axis=1)
            Y  = X_testi.SepsisLabel

            #for i in np.arange(self.k):
            for i, expert_name in enumerate(self.experts_name_list):
                '''
                draw hat_mu according to the utility function defined by Physionet for the 2019 sepsis challenge
                c.f. https://physionet.org/content/challenge-2019/1.0.0/
                (see the evaluation - scoring part)
                '''
                ##
                # suppose that each time there is only one patient arriving at the system
                #if i == 0: # ecpert0: random guess
                if expert_name == "RandomGuess":
                ## suppose the second arm is a random guess in [0,1]
                    r = random.uniform(0, 1)
                elif expert_name == "XGB":
                    r = test_xgb.get_reward_for_current_patient(patient_i, self.experts_name_list)
                elif expert_name == "RandomForest":
                    # hat_mu = test_expert1.get_reward_for_current_patient(patient_i)
                    r = test_random_forest_svc.get_reward_for_current_patient_rf(X ,Y , patient_i,self.experts_name_list)
                    # if (r>=0.95) or (r<=0.1):
                    #     if patient_i%100==0:
                    #         test_random_forest_svc.lime_explainer(expert_name, X, patient_i)
                elif expert_name == "SVC":
                    r = test_random_forest_svc.get_reward_for_current_patient_SVC(X ,Y , patient_i, self.experts_name_list)
                    # if (r>=0.95) or (r<=0.1):
                    #     if patient_i%100==0:
                    #         test_random_forest_svc.lime_explainer(expert_name, X, patient_i)
                elif expert_name == "LR":
                    r = test_random_forest_svc.get_reward_for_current_patient_LR(X ,Y , patient_i, self.experts_name_list)
                    # if (r>=0.95) or (r<=0.1):
                    #     if patient_i%100==0:
                    #         test_random_forest_svc.lime_explainer(expert_name, X, patient_i)
                current_reward[i] = r
                if t ==0:
                    hat_mu = r
                else:
                    hat_mu = (r +   hat_mu_list[i]*(t-1))/t
                hat_mu_list[i] =  hat_mu


            self.hat_mu_list = hat_mu_list

            '''
            Then, get UCB values of each arm and get max arm index. Specifically,
	for arm i at time/round t,
	UCB_i^(t) = hat_mu_i^(t) +  sqrt( [ alpha * log (t) ] /  N_i(t)  )
	where hat_mu_i^(t) is the estimated value of arm i at time t
	N_i(t) : up to time t, the # of times arm i was picked
	select argmax (UBC_i(t)) over K arms, observe reward x_i(t)
            '''
            pulled_arm  =  UCB(t, hat_mu_list, self.N, self.alpha).pull_max_arm()
            pulled_arm = int(pulled_arm)
            reward = current_reward[pulled_arm]
            arm_history[t] = pulled_arm
            expert_name_history.append(self.experts_name_list[ pulled_arm ])
            all_utilities[t] = reward
            UCB_values = UCB(t, hat_mu_list, self.N, self.alpha).get_UCB()
            reward = float(reward)
            self.rewards.append(reward)
            print("\nSelecting arms.............")
            print("*** Selected arm:",pulled_arm)
            print("*** Reward:",reward)

            # get regret
            avg_regret = self.get_average_regret(t)
            cul_regret = self.get_cu_regret(t)
            print("*** Average regret:",avg_regret)
            self.regrets.append(avg_regret)
            self.cul_regrets.append(cul_regret)

            print("\n\n\n")
            # update N_i,t,N_matrix [T,k]

            self.N[pulled_arm]+=1
            self.N_matrix[t,:]=self.N

        ## save historical utility, selected arm
        self.arm_history = arm_history
        self.all_utilities = all_utilities
        self.expert_name_history =  expert_name_history


        ### save dat data

        ttt = np.arange(self.T)
        header = "t, psv ID, Selected arm ID, Expert Name, Utility Score"
        data = np.column_stack((ttt, psv_list , self.arm_history,self.expert_name_history, self.all_utilities ))
        np.savetxt(os.path.join( f_dat_path, 'arm_utility_history.dat'),  data, fmt="%s", delimiter='|', header=header)




        self.plot_regret(f_dat_path,f_imgs_path)





    def get_average_regret(self,t):

        regret = ((t+1)*self.mu_best - np.sum(self.rewards))/float(t+1)
        if self.mu_best<self.rewards[-1]:
            print("the reward of round ", t , " exceeds 4, which is ", self.rewards[-1], "!!!!!!")
        return regret

    def get_cu_regret(self,t):

        regret = (t+1)*self.mu_best - np.sum(self.rewards)
        if self.mu_best<self.rewards[-1]:
            print("the reward of round ", t , " exceeds 4, which is ", self.rewards[-1], "!!!!!!")
        return regret


    def plot_regret(self,f_dat_path,f_imgs_path):


        x = np.arange(self.T)
        y = self.regrets
        y1 = self.cul_regrets
        header = "t, Average Regret, Cumulative Regret"
        data = np.column_stack((x, y, y1))
        np.savetxt(os.path.join(f_dat_path,'average_cumulative_regret.dat'),  data ,delimiter='|', header=header)

       ##########Average regret plot ########################
        plt.figure(1)
        plt.plot(x,y)
        plt.title('average regret VS time')
        plt.xlabel('time')
        plt.ylabel('average regret')
        img_name = 'avg_regret_t_'+str(self.T)+'.png'
        filename=os.path.join(f_imgs_path,img_name)
        plt.savefig(filename)



        ##########Cumulative regret plot########################
        plt.figure(2)
        plt.plot(x,y1)
        plt.title('Cumulative regret VS time')
        plt.xlabel('time')
        plt.ylabel('cumulative regret')
        img_name = 'cumulative_regret_t_'+str(self.T)+'.png'
        filename=os.path.join(f_imgs_path,img_name)
        plt.savefig(filename)





def main():

    alpha = 2
    test_set = np.load('./Candidate_xgb/data/test_set.npy')
    T = int(math.floor(np.size(test_set)/500 ))
    T = 2500

  # run for a single XGB for fair comparison
    mab = MAB(alpha, T, ['XGB'])
    mab._start_game( )

    experts_name_list = []
    experts_name_list.append(['RandomGuess',   'XGB',          'RandomForest', 'LR' ,   'SVC' ])

    experts_name_list.append(['RandomGuess',   'XGB',          'RandomForest', 'LR'  ])
    experts_name_list.append(['RandomGuess',   'XGB',          'RandomForest', 'SVC' ])
    experts_name_list.append(['RandomGuess',   'XGB',          'LR' ,          'SVC' ])
    experts_name_list.append(['RandomGuess',   'RandomForest', 'LR' ,          'SVC' ])
    experts_name_list.append(['XGB',           'RandomForest', 'LR' ,          'SVC' ])




    experts_name_list.append(['RandomForest',  'LR' ,          'SVC' ])
    experts_name_list.append(['XGB',           'LR' ,          'SVC' ])
    experts_name_list.append(['XGB',           'RandomForest', 'SVC' ])
    experts_name_list.append(['XGB',           'RandomForest', 'LR'   ])
    experts_name_list.append(['RandomGuess',   'LR' ,          'SVC' ])
    experts_name_list.append(['RandomGuess',   'RandomForest', 'SVC' ])
    experts_name_list.append(['RandomGuess' ,  'RandomForest', 'LR'   ])
    experts_name_list.append(['RandomGuess',   'XGB' ,         'SVC' ])
    experts_name_list.append(['RandomGuess',   'XGB',          'LR'   ])
    experts_name_list.append(['RandomGuess',   'XGB',          'RandomForest'  ])


    experts_name_list.append(['RandomGuess',  'XGB'  ])
    experts_name_list.append(['RandomGuess',  'RandomForest' ])
    experts_name_list.append(['RandomGuess',  'LR'  ])
    experts_name_list.append(['RandomGuess',  'SVC' ])
    experts_name_list.append(['XGB',          'RandomForest' ])
    experts_name_list.append(['XGB',          'LR'   ])
    experts_name_list.append(['XGB',          'SVC' ])
    experts_name_list.append(['RandomForest', 'LR'  ])
    experts_name_list.append(['RandomForest', 'SVC' ])
    experts_name_list.append(['LR' ,          'SVC' ])
    # for i in range(len(experts_name_list)):
    #     mab = MAB(  alpha, T,experts_name_list[i] )
    #     mab._start_game( )

if __name__=='__main__':
    main()
