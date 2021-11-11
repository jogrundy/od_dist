# for density based algos.
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np

def get_OCSVM_os(X, kernel='rbf', nu=0.5):
    clf = OneClassSVM(kernel=kernel, nu=nu,gamma='scale')
    clf.fit(X)
    dists = clf.decision_function(X)*-1
    return dists

def get_IF_os(X, n_estimators=100):
    clf = IsolationForest(n_estimators=n_estimators, contamination='auto')
    clf.fit(X)
    os = clf.decision_function(X)*-1 # average number splits to isolation. small is outlier.
    return os

def get_GMM_os(X, k=3):
    clf = GaussianMixture(n_components=k)
    clf.fit(X)
    scores = clf.score_samples(X)*-1 # returns log probs for data
    return scores

def get_DBSCAN_os(X, eps=0.5, min_samples=5):
    n,p = X.shape
    clf = DBSCAN(eps=eps, min_samples=min_samples)
    classes = clf.fit_predict(X)

    #returns only in class or out of class binary classification
    i = -1
    n_found = 0
    cl_sizes = {}
    while n_found <n:
        n_found_inds = len(np.where(classes == i)[0])
        n_found += n_found_inds
        # print(i, n_found_inds)
        cl_sizes[i] = n_found_inds
        i+=1
    # print(cl_sizes)
    cl_lst = [i[0] for i in sorted(cl_sizes.items(), key=lambda k:k[1], reverse=True)]
    # print(cl_lst)
    n_classes = len(cl_lst)

    # most populous group get score zero, then 1, 2, etc..
    os = [n_classes if x<0 else x for x in classes]

    return np.array(os)
