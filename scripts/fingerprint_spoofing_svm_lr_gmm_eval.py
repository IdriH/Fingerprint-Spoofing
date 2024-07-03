import numpy as np
import matplotlib.pyplot as plt
from load import load
from logistic_regression import *
from gmm import *
from svm import *

def plot_dcf_vs_log_odds(sValLLR, LVAL, pT_range, log_odds_range, color, label):
    actual_DCFs = []
    minimum_DCFs = []
    for pT in pT_range:
        # Compute actual DCF
        act_DCF = compute_actDCF_binary_fast(sValLLR, LVAL, pT, Cfn=1.0, Cfp=1.0)
        # Compute minimum DCF
        min_DCF = compute_minDCF_binary_fast(sValLLR, LVAL, pT, Cfn=1.0, Cfp=1.0)
        actual_DCFs.append(act_DCF)
        minimum_DCFs.append(min_DCF)
    
    plt.plot(log_odds_range, actual_DCFs, label=f'Actual DCF - {label}', color=color, linestyle='-')
    plt.plot(log_odds_range, minimum_DCFs, label=f'Minimum DCF - {label}', color=color, linestyle='--')

if __name__ == '__main__':
    D, L = load('../data/raw/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)    

    

    # Quadratic logistic regression for lambda = 0.005
    DTR_exp = expand_features_quadratic(DTR)
    DVAL_exp = expand_features_quadratic(DVAL)

    # Train and evaluate the model for lambda = 0.005
    lamb = 0.005
    log_odds_range = np.linspace(-4, 4, 100)
    pT_range = 1 / (1 + np.exp(-log_odds_range))  # Convert log-odds to prior probabilities

    color = 'r'  # Color for the plot
    plt.figure()

    # Train the model
    w, b = trainLogRegBinary(DTR_exp, LTR, lamb)
    sVal = np.dot(w.T, DVAL_exp) + b  # Compute validation scores
    pEmp = (LTR == 1).sum() / LTR.size  # Compute empirical prior
    sValLLR = sVal - np.log(pEmp / (1 - pEmp))  # Compute LLR-like scores

    # Plot DCF vs. log-odds for the selected lambda
    plot_dcf_vs_log_odds(sValLLR, LVAL, pT_range, log_odds_range, lamb, color, f'Lambda={lamb}')
    
    plt.xlabel('Log-Odds')
    plt.ylabel('DCF value')
    plt.title('DCF vs Log-Odds for Quadratic Logistic Regression (Lambda=0.005)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/plots/dcf_vs_log_odds_quadratic_lambda_0.005.png')
    plt.show()


  # SVM with rbf kernel and C = 30
    C = 30
    gamma = 0.1353   # You can adjust gamma for the RBF kernel
    kernelFunc = rbfKernel(gamma)
    
    # Train the SVM model
    fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc)

    # Compute validation scores
    sVal = fScore(DVAL)
    
    # Define log-odds range
    log_odds_range = np.linspace(-4, 4, 100)
    pT_range = 1 / (1 + np.exp(-log_odds_range))  # Convert log-odds to prior probabilities
    
    # Color for the plot
    color = 'r'
    plt.figure()

    # Compute empirical prior
    pEmp = (LTR == 1).sum() / LTR.size
    sValLLR = sVal - np.log(pEmp / (1 - pEmp))  # Compute LLR-like scores

    # Plot DCF vs. log-odds for the selected lambda
    plot_dcf_vs_log_odds(sValLLR, LVAL, pT_range, log_odds_range, C, color, f'C={C}')
    
    plt.xlabel('Log-Odds')
    plt.ylabel('DCF value')
    plt.title('DCF vs Log-Odds for SVM with RBF Kernel (C=30)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/plots/dcf_vs_log_odds_svm_rbf_c1.png')
    plt.show()


    
     # GMM with full covariance matrix and 16 components
    numComponents = 16
    gmm_0 = train_GMM_LBG_EM(DTR[:, LTR == 0], numComponents, covType='Full')
    gmm_1 = train_GMM_LBG_EM(DTR[:, LTR == 1], numComponents, covType='Full')

    # Compute validation scores
    logS0 = logpdf_GMM(DVAL, gmm_0)
    logS1 = logpdf_GMM(DVAL, gmm_1)
    sValLLR = logS1 - logS0
    
    # Define log-odds range
    log_odds_range = np.linspace(-4, 4, 100)
    pT_range = 1 / (1 + np.exp(-log_odds_range))  # Convert log-odds to prior probabilities
    
    # Color for the plot
    color = 'b'
    plt.figure()

    # Plot DCF vs. log-odds for GMM
    plot_dcf_vs_log_odds(sValLLR, LVAL, pT_range, log_odds_range, color, f'GMM {numComponents} Components')

    
    plt.xlabel('Log-Odds')
    plt.ylabel('DCF value')
    plt.title(f'DCF vs Log-Odds for GMM with {numComponents} Components')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../results/plots/dcf_vs_log_odds_gmm_{numComponents}_components.png')
    plt.show()