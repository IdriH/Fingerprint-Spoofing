import numpy

from load import load
import matplotlib.pyplot as plt 


def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_pca(D, m):

    mu, C = compute_mu_C(D)
    U, s, Vh = numpy.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D
    
def plot_histograms(D_pca, L, num_components):
    D0_pca = D_pca[:, L == 0]
    D1_pca = D_pca[:, L == 1]

    for i in range(num_components):
        plt.figure()
        plt.hist(D0_pca[i, :], bins=10, alpha=0.5, label='Spoofed')
        plt.hist(D1_pca[i, :], bins=10, alpha=0.5, label='Authentic')
        plt.title(f'PCA Component {i+1}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.savefig('../results/plots/pca_component_analysis/hist_%d.png' % i)
        plt.show()

if __name__ == '__main__':

    
    D , L  = load('../data/raw/trainData.txt')


    print(D)
    print(D.shape)


    ## Apply PCA 
    m=6 # n directions
    P_pca = compute_pca(D,m)
    D_pca = apply_pca(P_pca,D)


    print(D_pca.shape)

    print(D_pca)
    
    plot_histograms(D_pca,L ,m)
    
