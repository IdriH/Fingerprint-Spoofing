

import numpy

import matplotlib.pyplot as plt 

import scipy.linalg
from load import load

def load_iris():
    
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def vcol(x): 
    return x.reshape((x.size, 1))

def vrow(x): 
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in numpy.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_geig(D, L, m):
    
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]

def compute_lda_JointDiag(D, L, m):

    Sb, Sw = compute_Sb_Sw(D, L)

    U, s, _ = numpy.linalg.svd(Sw)
    P = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)

    Sb2 = numpy.dot(P, numpy.dot(Sb, P.T))
    U2, s2, _ = numpy.linalg.svd(Sb2)

    P2 = U2[:, 0:m]
    return numpy.dot(P2.T, P).T

def apply_lda(U, D):
    return U.T @ D

def plot_lda_histogram(D_LDA, L):
    D0_LDA = D_LDA[:, L == 0]
    D1_LDA = D_LDA[:, L == 1]

    plt.figure()
    plt.hist(D0_LDA[0, :], bins=10, alpha=0.5, label='Spoofed')
    plt.hist(D1_LDA[0, :], bins=10, alpha=0.5, label='Authentic')
    plt.title('LDA Component')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig('../results/plots/lda/hist_lda.png' )
    plt.show()


if __name__ == '__main__':

    D , L  = load('../data/raw/trainData.txt')

    print(D)


    U = compute_lda_geig(D, L, m = 1)
    print(U)
    print(compute_lda_JointDiag(D, L, m=1)) # May have different signs for the different directions
    
    D_LDA = apply_lda(U,D)

    # Check if the authentic class samples are, on average, on the right of the spoofed samples , as for convention
    if D_LDA[0, L==0].mean() > D_LDA[0, L==1].mean():
        U = -U
        D_LDA = apply_lda(U, D)



    ##

    print(D_LDA)
    # Plot histogram of the LDA-projected data
    plot_lda_histogram(D_LDA, L)