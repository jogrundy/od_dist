"""
refactoring outlier score code
"""

import numpy as np

from .reg_os import get_VAR_os, get_OLS_os, get_ridge_os, get_LASSO_os
from .dens_os import get_OCSVM_os, get_IF_os, get_GMM_os, get_DBSCAN_os
from .gru import get_GRU_os, get_LSTM_os
from .outlier_pursuit import get_OP_os
from .GOP import get_GOP_os
from .ae import get_AE_os
from .vae import get_VAE_os
from .utils import norm_score

def rand_os(X):
    """
    to give a true baseline for the algorithms.
    generates a random os score for each sample.
    returns random uniform score between 0 and 1.
    """
    n,p = X.shape
    return np.random.rand(n)

#super class for all algoritms to inherit from.
class OD():
    algo_dict = {'VAR':get_VAR_os, 'FRO':get_OLS_os, 'FRL':get_LASSO_os, 'FRR':get_ridge_os,
    'GMM': get_GMM_os, 'OCSVM': get_OCSVM_os, 'DBSCAN':get_DBSCAN_os,
    'IF': get_IF_os,
    'AE': get_AE_os, 'VAE': get_VAE_os, 'GRU':get_GRU_os, 'LSTM':get_LSTM_os,
    'OP': get_OP_os, 'GOP': get_GOP_os, 'RAND':rand_os}
    def __init__(self, algo, params=[]):
        #algo is string representing algo.
        # params is optional, if you want to change the parameters from default
        # all parameters for the algo in question must be specified at that point.
        self.params = params
        self.algo_str = algo

    def get_os(self, data, norm=False):
        """
        data must be in form n samples by p features
        will return a score for each sample, array length n.
        """
        self.algo = OD.algo_dict[self.algo_str]
        os = self.algo(data, *self.params)
        if norm:
            os = norm_score(os)
        # os = sanitise_scores(os)
        return os
