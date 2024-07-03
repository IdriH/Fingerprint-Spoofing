import numpy as np
import matplotlib.pyplot as plt
from load import load
from logistic_regression import *
from gmm import train_GMM_LBG_EM, logpdf_GMM
from svm import rbfKernel, train_dual_SVM_kernel
from evaluation import *
from calibration1 import *

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def bayesPlot(S, L, left=-4, right=4, npts=21):
    effPriorLogOdds = np.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(compute_actDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
    return effPriorLogOdds, actDCF, minDCF

def compute_and_plot_DCF(S, L, model_name):
    effPriorLogOdds, actDCF, minDCF = bayesPlot(S, L)
    plt.plot(effPriorLogOdds, actDCF, label=f'Actual DCF {model_name}')
    plt.plot(effPriorLogOdds, minDCF, label=f'Minimum DCF {model_name}')
    plt.xlabel('Effective Prior Log-Odds')
    plt.ylabel('DCF')
    plt.legend()
    plt.title(f'Bayes Error Plot for {model_name}')
    plt.grid(True)
    plt.savefig(f'../results/plots/final_results/{model_name}.png')
    plt.show()

    #print(f'Actual DCF for {model_name}:', actDCF)
    #print(f'Minimum DCF for {model_name}:', minDCF)

if __name__ == '__main__':
    # Load evaluation data
    D_eval, L_eval = load('../data/raw/evalData.txt')

    # Preprocess evaluation data
    D_eval_exp = expand_features_quadratic(D_eval)

    # Load trained model parameters
    w = np.load('../results/final_models/lr/lr_w.npy')
    b = np.load('../results/final_models/lr/lr_b.npy')


    ##Quadratic Logistic Regression evaluation

    sVal_lr = np.dot(w.T, D_eval_exp) + b  # Compute validation scores
    pEmp_lr = (L_eval == 1).sum() / L_eval.size  # Compute empirical prior
    sValLLR_lr = sVal_lr - np.log(pEmp_lr / (1 - pEmp_lr))  # Compute LLR-like scores
    scores_lr = sVal_lr

    print('Logistic Regression: minDCF (0.1) = %.3f - actDCF (0.1) = %.3f' % (
        compute_minDCF_binary_fast(scores_lr, L_eval, 0.1, 1.0, 1.0),
        compute_actDCF_binary_fast(scores_lr, L_eval, 0.1, 1.0, 1.0)))

    compute_and_plot_DCF(scores_lr,L_eval,"QuadraticLogistic Regression")
    

    


    #np.save is giving issues saving gaussian mixture classifiers so we train in this 
    # script using traindata and evaluate it on ecaluation data
    

    D, L = load('../data/raw/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)


    #GMM clasiffier 
    #train on training data evaluate on evaluation data
    ## changing to 8 and diagonal to test original is 16 , full 
    numComponents = 16
    gmm_0 = train_GMM_LBG_EM(DTR[:, LTR == 0], numComponents, covType='full')
    gmm_1 = train_GMM_LBG_EM(DTR[:, LTR == 1], numComponents, covType='full')

    logS0 = logpdf_GMM(D_eval, gmm_0)
    logS1 = logpdf_GMM(D_eval, gmm_1)
    sEvalLLR_gmm = logS1 - logS0
    scores_gmm = sEvalLLR_gmm

    # Compute performance metrics
    print('GMM: minDCF (0.1) = %.3f - actDCF (0.1) = %.3f' % (
        compute_minDCF_binary_fast(scores_gmm, L_eval, 0.1, 1.0, 1.0),
        compute_actDCF_binary_fast(scores_gmm, L_eval, 0.1, 1.0, 1.0)))
    compute_and_plot_DCF(scores_gmm,L_eval,"Gaussian Mixture Model nComponents = 16")
    
    


    # SVM with RBF kernel and C = 30: 

    # SVM with RBF kernel and C = 30
    C = 30
    gamma = 0.1353
    kernelFunc = rbfKernel(gamma)
    fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc)


    ##now evaluation on evaluation data
    
    sVal_svm = fScore(D_eval)
    pEmp_svm = (L_eval == 1).sum() / L_eval.size
    sValLLR_svm = sVal_svm - np.log(pEmp_svm / (1 - pEmp_svm))  # Compute LLR-like scores
    scores_svm = sValLLR_svm

    print('SVM: minDCF (0.1) = %.3f - actDCF (0.1) = %.3f' % (
        compute_minDCF_binary_fast(scores_svm, L_eval, 0.1, 1.0, 1.0),
        compute_actDCF_binary_fast(scores_svm, L_eval, 0.1, 1.0, 1.0)))
    

    compute_and_plot_DCF(scores_svm,L_eval,"SVM with RBF kernel and C = 30 gamma = 0.135")
     # Compute performance metrics
    
    # Fusion #
    fusedScores = []
    fusedLabels = []

    pT = 0.1

    for foldIdx in range(KFOLD):
        SCAL1, SVAL1 = extract_train_val_folds_from_ary(scores_lr, foldIdx)
        SCAL2, SVAL2 = extract_train_val_folds_from_ary(scores_svm, foldIdx)
        SCAL3, SVAL3 = extract_train_val_folds_from_ary(scores_gmm, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(L_eval, foldIdx)

        SCAL = np.vstack([SCAL1, SCAL2, SCAL3])
        w, b = trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)

        SVAL = np.vstack([SVAL1, SVAL2, SVAL3])
        fused_SVAL = (w.T @ SVAL + b - np.log(pT / (1-pT))).ravel()

        fusedScores.append(fused_SVAL)
        fusedLabels.append(LVAL)

    fusedScores = np.hstack(fusedScores)
    #np.save('../results/scores/fused_scores.npy', fusedScores)
    fusedLabels = np.hstack(fusedLabels)

    # Evaluate the performance on pooled scores
    print('Fusion:')
    print('minDCF (0.1) = %.3f' % compute_minDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0))
    print('actDCF (0.1) = %.3f' % compute_actDCF_binary_fast(fusedScores, fusedLabels, 0.1, 1.0, 1.0))

    # Plot fused Bayes error plots
    logOdds, actDCF, minDCF = bayesPlot(fusedScores, fusedLabels)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(logOdds, minDCF, color='C3', linestyle='--', label='minDCF')
    ax.plot(logOdds, actDCF, color='C3', linestyle='-', label='actDCF')
    ax.set_ylim(0, 0.8)
    ax.legend()
    ax.set_title('Fusion - validation - fused scores')
    plt.savefig('../results/plots/final_results/fusion_calibrates.png')
    plt.show()