"""
Translated from matlab,
https://github.com/omarshetta/Manuscript_Royal_Society/blob/master/utils/OUTLIER_PERSUIT.m

Implementing Xu et al Robust PCA via outlier Pursuit
IEEE Trans Inf Theory, 58, pg 3047-3064 2012
"""

import numpy as np
import numpy.linalg as la


def outlier_pursuit(M,lamb):
    mu = 0.99*la.norm(M,2)
    m,n = M.shape
    delta=10e-5
    eta=0.9
    # print('mu = ', mu)
    mu_bar=delta* mu

    L_min0=np.zeros((m,n))
    L_min1=np.zeros((m,n))
    C_min0=np.zeros((m,n))
    C_min1=np.zeros((m,n))
    t_min0=1
    t_min1=1

    converged=False
    count=0
    tol=0.000001*la.norm(M) #norm defaults to frobenius norm for matrix
    while not converged:

        Y_L=L_min0+((t_min1-1)/t_min0)*(L_min0-L_min1)
        Y_C=C_min0+((t_min1-1)/t_min0)*(C_min0-C_min1)
        G_L=Y_L-0.5*(Y_L+Y_C-M)
        G_C=Y_C-0.5*(Y_L+Y_C-M)

        L_new=prox_nuc_norm(G_L,mu/2)
        C_new=column_thresh(G_C,lamb*mu/2)

        t_new=(1+np.sqrt(4*t_min0*t_min0+1))/2
        mu_new=np.max([eta*mu,mu_bar])
        # % check if converged %
        S_L=2*(Y_L-L_new)+(L_new+C_new-Y_L-Y_C)
        S_C=2*(Y_C-C_new)+(L_new+C_new-Y_L-Y_C)
        if la.norm(S_L)**2 + la.norm(S_C)**2 < tol*tol:
            converged=True

        L_min1=L_min0
        L_min0=L_new
        C_min1=C_min0
        C_min0=C_new
        t_min1=t_min0
        t_min0=t_new
        mu=mu_new

        if count ==1000:
            print('Failed to converge')
            break

        count+=1
    L_hat=L_new
    C_hat=C_new
    return L_hat, C_hat, count

def prox_nuc_norm(X, eps):
    """
    to solve arg min_x 1\2 ||X - Y||^2_F + \tau||X||_*

    nuclear norm ||A||* = trace(sqrt(A*A))
    also known as trace norm.
    (latex) ||A||_*  =  \sum_{i=1}^{min\{m,n\}} \sigma_i (A)

    solution is soft thresholding operator.
    """
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
    """
    cleans out C, removes all values below eps.
    otherwise
    """
    n1 = C.shape[1]
    for i in range(n1):
        if la.norm(C[:,i], 2) < eps: # norm here defaults to 2 norm for vector
            C[:,i]=0
        else:
            C[:,i]=C[:,i]-eps*C[:,i]/la.norm(C[:,i],2)
    return C

def get_OP_os(X, lamb=0.5):
    """
    takes in only data 'X', in samples as rows format
    returns only list of outlier scores for each sample
    outlier score is sum of comlumn sparse matrix C
    higher score = more outlier
    """
    M = X.T #should be p,n
    L_hat, C_hat, count = outlier_pursuit(M, lamb)
    return np.sum(C_hat, axis=0)

if __name__ == '__main__':
    """
    testing code for OP
    """

    import matplotlib.pyplot as plt

    def get_lambda(n,mu, r, f):
        c = 9/121
        gamma = mu*r/(1+mu*r/c)
        lamb = 3/(7*np.sqrt(gamma*n))
        lamb = lamb *f
        print('Lambda = ', lamb)
        return lamb

    def test_op(n, p,r, lamb, gamma, mode,f):
        """
        for testing outlier pursuit algorithm
        """
        L_o, C_o, M = get_test_M_Xu(n,p,r, gamma,mode)
        L_hat, C_hat, count = outlier_pursuit(M,lamb)
        print("rank L_o = {0}, rank L_hat = {1}".format(r, la.matrix_rank(L_hat)))
        print("Norm diff ratio = {}".format(la.norm(L_o - L_hat)/la.norm(L_o))) # norm here defaults to 2 norm for vector
        plot_op(L_o, C_o, L_hat, C_hat, n, p, r)
        return L_hat, C_hat, count

    def plot_op(L_o, C_o, L_hat, C_hat, n, p, r):
        plt.figure(figsize=(6,8))
        plt.subplot(2,3,1)
        plt.imshow(L_o)
        plt.title("Original L_o rank {}".format(la.matrix_rank(L_o)))
        plt.colorbar()
        plt.subplot(2,3,2)
        plt.imshow(C_o)
        plt.title('Original C_o, density {}'.format(np.count_nonzero(np.sum(C_o, axis=0))))
        plt.colorbar()
        plt.subplot(2,3,3)
        plt.imshow(L_o+C_o)
        plt.title('Original M rank {}'.format(la.matrix_rank(L_o+C_o)))
        plt.colorbar()
        plt.subplot(2,3,4)
        plt.imshow(L_hat)
        plt.title("Estimated L_hat rank {}".format(la.matrix_rank(L_hat)))
        plt.colorbar()
        plt.subplot(2,3,5)
        plt.imshow(C_hat)
        plt.title('Estimated C_hat {} non zero cols'.format(np.count_nonzero(np.sum(C_hat, axis=0))))
        plt.colorbar()
        plt.subplot(2,3,6)
        plt.imshow(L_hat+C_hat)
        plt.title('Reconstructed matrix, rank {}'.format(la.matrix_rank(L_hat+C_hat)))
        plt.colorbar()
        fname = 'OP_n_{}_p_{}_r_{}_plot.eps'.format(n,p,r)
        plt.savefig(fname)
        plt.show()

    def get_test_M_Xu(n,p,r, gamma,mode='neutral'):
        """
        test M generated according to Xu et al Robust PCA via outlier Pursuit
        IEEE Trans Inf Theory, 58, pg 3047-3064 2012
        """
        A = np.random.rand(p,r)
        B = np.random.rand(n,r) # changed from int(n-gamma*n) to n
        L_o = np.dot(A,B.T) #the clean part of M
        L_o = (L_o - np.mean(L_o[:]))/np.std(L_o[:]) # normalised.. not sure if this is a good idea

        # chose at random columns to be outliers
        nz_cols = np.random.choice(np.arange(n), size=(int(gamma*n)), replace=False)
        C_o = np.zeros((n,p))
        M_o = np.copy(L_o)
        if mode=='neutral':
            for col in nz_cols:
                L_o[:,col] = np.zeros(p)
                C_o[:,col] = np.random.rand(p)
                M_o[:,col] = np.random.rand(p)
        if mode == 'adversarial':
            rv = np.random.rand(p)
            for col in nz_cols:
                L_o[:,col] = np.zeros(p)
                C_o[:,col] = rv
                M_o[:,col] = rv

        return L_o, C_o, M_o
