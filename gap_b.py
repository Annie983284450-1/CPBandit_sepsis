import numpy as np

'''
    Updated gap-based bandit:
    f_tk(LOO): the fitted leave-one-out algorithm based on historical data
    F-1(1-alpha+hat_beta(t,k)):  the inverse empirical quantile function (ICDF) [quantile of the residuals]
    U = f_tk(LOO) + F-1(1-alpha+hat_beta(t,k))
    L = f_tk(LOO) + F-1(hat_beta(t,k))
'''

class gap_bandit(object):
    def __init__(self, UCB, LCB, K):
        self.K = K
        self.UCB = UCB   
        self.LCB = LCB 
        self.s_value = [0]*self.K
        self.B_value = [-float('inf')]*self.K

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
 


