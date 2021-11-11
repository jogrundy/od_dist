
"""
translated from matlab
https://github.com/omarshetta/Manuscript_Royal_Society/blob/master/utils/admm_algo_OP_on_graphs.m
by Omar Shetta
"""


import numpy as np
import numpy.linalg as la
from scipy.sparse import issparse
from scipy.sparse.linalg import svds

def graph_reg_OP(X, lamb, gamma, phi):
    eta=0.000001
    p, n = X.shape
    L_k = np.random.randn(p,n)
    W_k = np.random.randn(p,n)
    S_k = np.random.randn(p,n)
    S_kmin1 = np.random.randn(p,n)
    W_kmin1 = np.random.randn(p,n)

    Z1_k = X - L_k - S_k
    Z2_k = W_k - L_k
    P1_k = nuclear_norm(L_k)
    P2_k = lamb*sum(np.sqrt(np.sum(S_k**2, axis=0)))
    P3_k =  gamma*np.trace(np.dot(L_k, np.dot(phi, L_k.T)))
    converged=False
    count=0
    r1_k = 1
    r2_k = 1

    maxiter=1000
    obj_func_kp1 = []
    while not converged:
        count=count+1;
        # print('GOP iteration =', count)
        H_1 = X - S_k + (Z1_k/r1_k)
        H_2 = W_k + (Z2_k/r2_k)
        A = (r1_k * H_1 + r2_k * H_2) / (r1_k + r2_k)
        r_k = (r1_k + r2_k)/2

        L_kp1 = prox_nuc_norm(A, 1/r_k)
        S_kp1 = column_thresh((X - L_kp1 + (Z1_k/r1_k)), lamb/r1_k)

        #matrix inversion causes slowdown. Tweak
        W_kp1 = GOP_tweak(gamma, phi, r2_k, n, L_kp1, Z2_k)

        Z1_kp1 = Z1_k + r1_k * (X - L_kp1 - S_kp1)
        Z2_kp1 = Z2_k + r2_k * (W_kp1 - L_kp1)

        P1_kp1 = nuclear_norm(L_kp1)
        P2_kp1 = lamb*sum(np.sqrt(sum(S_kp1**2)))
        P3_kp1 = gamma*np.trace(np.dot(L_kp1, np.dot(phi, L_kp1.T)))

        obj_func_kp1.append(P1_kp1 + P2_kp1 + P3_kp1)

        # % checking convergence
        too_small = 1e-15
        if P1_k >too_small: #preventing division by zero error
            rel_err_1 = (P1_kp1 - P1_k)**2 / P1_k**2
        else:
            rel_err_1 = 0
        if P2_k >too_small:
            rel_err_2 = (P2_kp1 - P2_k)**2 / P2_k**2
        else:
            rel_err_2 = 0
        if P3_k >too_small:
            rel_err_3 = (P3_kp1 - P3_k)**2 / P3_k**2
        else:
            rel_err_3 =0
        if la.norm(Z1_kp1 -Z1_k,'fro') >too_small:
            rel_err_z1 = la.norm(Z1_kp1 - Z1_k,'fro')**2 / (la.norm(Z1_k,'fro')**2)
        else:
            rel_err_z1 = 0
        if la.norm(Z2_kp1 -Z2_k,'fro') >too_small:
            rel_err_z2 = la.norm(Z2_kp1 - Z2_k,'fro')**2 / (la.norm(Z2_k,'fro')**2)
        else:
            rel_err_z2 = 0
        within_tol = rel_err_1<eta and rel_err_2<eta and rel_err_3<eta and rel_err_z1<eta and rel_err_z2<eta
        if within_tol or count==maxiter:
            converged=True
        else:
            # % get ready for new iteration
            S_k = S_kp1
            W_k = W_kp1
            Z1_k = Z1_kp1
            Z2_k = Z2_kp1
            P1_k = P1_kp1
            P2_k = P2_kp1
            P3_k = P3_kp1

    L_hat = L_kp1
    S_hat = S_kp1
    # print('GOP converged in {} iterations'.format(count))
    return L_hat, S_hat, obj_func_kp1

def GOP_tweak(gamma, phi, r2_k, n, L_kp1, Z2_k):
    A = (gamma * phi + (r2_k* np.identity(n)) ).T
    B = (L_kp1 - (Z2_k/r2_k)).T
    XT = la.solve(A, B)
    return XT.T

def prox_nuc_norm(X, eps):

    U,S,V=la.svd(X, full_matrices=False)
    n1 = len(S) #in python S is vector
    # Diagonal soft thresholding
    for i in range(n1):
        if abs(S[i]) <= eps:
            S[i]=0
        else:
            S[i]=S[i]-eps*np.sign(S[i])

    s_mat=np.diag(S)
    X=np.dot(U,np.dot(s_mat,V))
    return X

def column_thresh(C, eps):
    n1 = C.shape[1]
    for i in range(n1):
        if la.norm(C[:,i], 2) < eps: # norm here defaults to 2 norm for vector
            C[:,i]=0
        else:
            C[:,i]=C[:,i]-eps*C[:,i]/la.norm(C[:,i],2)
    return C

def nuclear_norm(x):
    """
    sum of singular values
    Nuclear norm of x
       Usage: norm_nuclear(x)
       Input parameters
           x       : a matrix
       Output parameters
          n       : nuclear norm of x
    """
    if issparse(x):
        u, s, vt = svds(x, np.min(size(x)))
    else:
        u, s, vt = la.svd(x)
    return np.sum(s)

def knn_graph(X,k):
    """
    from matlab version:
    This function computes the K-Nearest Neighbour graph.
    Inputs:
    X, is the data matrix with dimension (n x m) n is the number of samples and m is the number of features.
    k, is a scalar that defines the number of nearest neighbours.

    Outputs:
    Lap, is the graph Laplacian matrix of the k-nearest neighbour graph. It is an (n x n) symmetric positive semi-definite matrix.
    W, is the Weight matrix. It is an (n x n) symmetric matrix. It holds the weight of each edge of the k-nearest neighbour graph.
    """
    K = k+1 #as don't want to include self in neighbours
    n,p = X.shape
    # print('Building KNN graph for', n, 'samples.')
    W = np.zeros((n,n))
    D = np.zeros((n,n))
    dists = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            d = la.norm(X[i,:] - X[j,:],2)
            dists[i,j] = d

        inds = np.argpartition(dists[i,:], K)
        dists[i,inds[K:]] = 0

    # calculate weights for nonzero values
    nz = np.nonzero(dists)
    const = (np.sum(dists[:])/(K*n))**2
    W[nz] = np.exp(-(dists[nz]**2)/const)
    # symmetrise
    W = (W+W.T)/2
    # calculate laplacian matrix
    d = np.sum(W, axis=1)
    D = np.diag(d)
    laplacian = np.array(D - W)
    return laplacian, W


def GOP(M, lamb, gamma):
    k = min(M.shape[1]-2, 5)
    phi, W = knn_graph(M.T, k)
    L_hat, S_hat, ob_fn = graph_reg_OP(M, lamb, gamma, phi)
    return S_hat

def get_GOP_os(X, lamb=0.5, alpha=0.1):
    """
    to tie in to outlier detection object
    """
    M = X.T
    S_hat = GOP(M, lamb, alpha)
    return np.sum(S_hat, axis=0)
