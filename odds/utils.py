# utils for outlier detection.

# needs:
import numpy as np
from sklearn import metrics

class TimeoutException(Exception):
    def __init__(self, time):
        Exception.__init__(self, 'timeout after {}s'.format(time))

def ese(pred, target):
    """
    takes in predicted values and actual values, returns elementwise squared error
    via (x-y)^2
    """
    # print(pred.shape, target.shape)
    errs = (pred - target)**2
    # errs = np.sum((pred - target)**2, axis=0)
    # print(errs.shape)
    return errs

def sanitise_scores(est_out_scores):
    """
    takes in array of scores. they may contain infs and nans. need to be sanitised.
    must preserve order.
    """
    n = len(est_out_scores)
    est_out_scores = np.array(est_out_scores).reshape(-1)
    inf_inds = np.isinf(est_out_scores)+ np.isnan(est_out_scores)
    # print(inf_inds)
    clean_scores = np.delete(est_out_scores, inf_inds)
    # print(clean_scores)

    # print(np.nanmax(clean_scores))
    if len(clean_scores)==0:
        print('all scores are inf or nan')
        return np.ones_like(est_out_scores)
    est_out_scores = est_out_scores/(0.1*np.nanmax(np.abs(clean_scores))) # max gives inf.
    #sanitise results to between -10 and 10

    for ind in range(len(inf_inds)): # deals with both -inf and +inf.
        if inf_inds[ind]:
            # print(est_out_scores[ind])
            if est_out_scores[ind] == -np.inf:
                est_out_scores[ind]=-11 #turn it up to 11 :-)
            else: # will also put np.nan as 11
                est_out_scores[ind]=11


    # print(est_out_scores)
    return est_out_scores


def auc(est_out_scores, outs, debug=False):
    """
    measures how good the separation is between outliers and inliers
    uses auc on the estimated outlier score from each algorithm, compared to
    the actual where 1 is for outlier and 0 is for not.
    """
    n = len(est_out_scores)
    # est_out_scores = est_out_scores.reshape(-1)
    # print(est_out_scores.shape)
    # est_out_scores = sanitise_scores(est_out_scores)
    actual_os = [1 if i in outs else 0 for i in range(n)]
    fpr, tpr, thresholds = metrics.roc_curve(actual_os, est_out_scores)
    return metrics.auc(fpr, tpr)


def fps(est_out_scores, outs):
    """
    measures how good the separation is between outliers and inliers
    uses number of false positives found after finding all outliers
    uses the estimated outlier score from each algorithm.
    higher score = more outlier
    """
    inds = np.flip(np.argsort(est_out_scores)) #gives indices in size order
    n = len(est_out_scores)
    for i in range(n):
        if len(np.setdiff1d(outs,inds[:i]))==0: #everything in outs is also in inds
            fps = len(np.setdiff1d(inds[:i], outs)) #count the things in inds not in outs
            return fps/i
    return 1

def normalise(X):
    """
    here I mean make max 1 and min 0 and fit everything else in.
    should be by feature.
    X is np array
    returns X shame shape but normalised by feature.
    """
    n,p = X.shape
    for column in range(p):
        X[:,column] = ((X[:,column] - np.max(X[:,column])) / (np.max(X[:,column]) - np.min(X[:,column])))+1
    return X

def norm_score(X):
    """
    here I mean make max 1 and min 0 and fit everything else in.
    1d for the score
    """
    X = ((X - np.max(X)) / (np.max(X) - np.min(X)))+1
    return X
