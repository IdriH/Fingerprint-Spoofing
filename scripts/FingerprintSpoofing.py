import numpy
import matplotlib 
import matplotlib.pyplot as plt 

import feature_analysis as fa

from load import load

import pca as pca

import basic_statistics as bSt

if __name__ == '__main__':
   
    D , L  = load('../data/raw/trainData.txt')


    print(D)
    print(D.shape)

    #mu,C = bSt.compute_mu_C(D)

    #bSt.print_basicStatistics(D,L,mu,C)

    #fa.plot_hist(bSt.D,bSt.L)
    #fa.plot_scatter(bSt.D,bSt.L)

    ## Apply PCA 
    m=6 # n directions
    P_pca = pca.compute_pca(D,m)
    D_pca = pca.apply_pca(P_pca,D)


    print(D_pca.shape)

    print(D_pca)
    
    pca.plot_histograms(D_pca,L ,m)