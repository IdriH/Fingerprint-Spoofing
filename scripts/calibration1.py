import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from load import load
from logistic_regression import *
from gmm import train_GMM_LBG_EM, logpdf_GMM
from svm import * 
from evaluation import *

#Global 
gmm_0 = None
gmm_1 = None





def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def bayesPlot(S, L, left = -3, right = 3, npts = 21):
    
    effPriorLogOdds = np.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(compute_actDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
    return effPriorLogOdds, actDCF, minDCF



def save_scores(DTR, LTR, DVAL, LVAL):
    # Quadratic Logistic Regression for lambda = 0.005
    DTR_exp = expand_features_quadratic(DTR)
    DVAL_exp = expand_features_quadratic(DVAL)
    lamb = 0.005
    w, b = trainLogRegBinary(DTR_exp, LTR, lamb)
    np.save('../results/final_models/lr_w.npy',w)
    np.save('../results/final_models/lr_b.npy',b)
    sVal_lr = np.dot(w.T, DVAL_exp) + b  # Compute validation scores
    pEmp_lr = (LTR == 1).sum() / LTR.size  # Compute empirical prior
    sValLLR_lr = sVal_lr - np.log(pEmp_lr / (1 - pEmp_lr))  # Compute LLR-like scores
    
    # SVM with RBF kernel and C = 30
    C = 30
    gamma = 0.1353
    kernelFunc = rbfKernel(gamma)
    fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc)
    np.save('../results/final_models/svm_fcscore.npy',fScore,allow_pickle=True)
    sVal_svm = fScore(DVAL)
    pEmp_svm = (LTR == 1).sum() / LTR.size
    sValLLR_svm = sVal_svm - np.log(pEmp_svm / (1 - pEmp_svm))  # Compute LLR-like scores
    
    # GMM with full covariance matrix and 16 components
    numComponents = 16
    gmm_0 = train_GMM_LBG_EM(DTR[:, LTR == 0], numComponents, covType='Full')
    gmm_1 = train_GMM_LBG_EM(DTR[:, LTR == 1], numComponents, covType='Full')
    #np.save('../results/final_models/gmm_0.npy',gmm_0)
    #np.save('../results/final_models/gmm_1.npy',gmm_1)
    logS0 = logpdf_GMM(DVAL, gmm_0)
    logS1 = logpdf_GMM(DVAL, gmm_1)
    sValLLR_gmm = logS1 - logS0

    # Save scores
    #np.save('../results/data/scores_lr.npy', sValLLR_lr)
    #np.save('../results/data/scores_svm.npy', sValLLR_svm)
    #np.save('../results/data/scores_gmm.npy', sValLLR_gmm)
    #np.save('../results/data/labels_val.npy', LVAL)
    
    sValLLR_svm = 0 # just for now

    return sValLLR_lr, sValLLR_svm, sValLLR_gmm 


KFOLD = 5

# Extract i-th fold from a 1-D numpy array (as for the single fold case, we do not need to shuffle scores in this case, but it may be necessary if samples are sorted in peculiar ways to ensure that validation and calibration sets are independent and with similar characteristics   
def extract_train_val_folds_from_ary(X, idx):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]


if __name__ == '__main__':
 

    D, L = load('../data/raw/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    
    scores_lr, scores_svm, scores_gmm= save_scores(DTR, LTR, DVAL, LVAL)

    print("Scores for Logistic Regression, SVM, and GMM saved.")

    scores_gmm = np.load('../results/data/scores_gmm.npy')
    scores_lr = np.load('../results/data/scores_lr.npy')
    scores_svm = np.load('../results/data/scores_svm.npy')
    labels = np.load('../results/data/labels_val.npy')

   # Compute performance metrics
    print('GMM: minDCF (0.1) = %.3f - actDCF (0.1) = %.3f' % (
        compute_minDCF_binary_fast(scores_gmm, labels, 0.1, 1.0, 1.0),
        compute_actDCF_binary_fast(scores_gmm, labels, 0.1, 1.0, 1.0)))

    print('Logistic Regression: minDCF (0.1) = %.3f - actDCF (0.1) = %.3f' % (
        compute_minDCF_binary_fast(scores_lr, labels, 0.1, 1.0, 1.0),
        compute_actDCF_binary_fast(scores_lr, labels, 0.1, 1.0, 1.0)))

    print('SVM: minDCF (0.1) = %.3f - actDCF (0.1) = %.3f' % (
        compute_minDCF_binary_fast(scores_svm, labels, 0.1, 1.0, 1.0),
        compute_actDCF_binary_fast(scores_svm, labels, 0.1, 1.0, 1.0)))

    # Plot Bayes error plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot for GMM
    logOdds, actDCF, minDCF = bayesPlot(scores_gmm, labels)
    axes[0].plot(logOdds, minDCF, color='C0', linestyle='--', label='minDCF')
    axes[0].plot(logOdds, actDCF, color='C0', linestyle='-', label='actDCF')
    axes[0].set_ylim(0, 0.8)
    axes[0].legend()
    axes[0].set_title('GMM - validation - non-calibrated scores')

    # Plot for Logistic Regression
    logOdds, actDCF, minDCF = bayesPlot(scores_lr, labels)
    axes[1].plot(logOdds, minDCF, color='C1', linestyle='--', label='minDCF')
    axes[1].plot(logOdds, actDCF, color='C1', linestyle='-', label='actDCF')
    axes[1].set_ylim(0, 0.8)
    axes[1].legend()
    axes[1].set_title('Logistic Regression - validation - non-calibrated scores')

    # Plot for SVM
    logOdds, actDCF, minDCF = bayesPlot(scores_svm, labels)
    axes[2].plot(logOdds, minDCF, color='C2', linestyle='--', label='minDCF')
    axes[2].plot(logOdds, actDCF, color='C2', linestyle='-', label='actDCF')
    axes[2].set_ylim(0, 0.8)
    axes[2].legend()
    axes[2].set_title('SVM - validation - non-calibrated scores')

    plt.tight_layout()
    plt.savefig('../results/plots/calibration/allmodels_no_calibrates.png')
    plt.show()

    # Calibration for Logistic Regression
    calibrated_scores_lr = []
    labels_lr = []

    # We train the calibration model for the prior pT = 0.1
    pT = 0.1

    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_lr, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
        # Apply the model to the validation fold
        calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the scores list
        calibrated_scores_lr.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        labels_lr.append(LVAL)

    # Re-build the score and label arrays (pooling)
    calibrated_scores_lr = np.hstack(calibrated_scores_lr)
    np.save('../results/scores/calibrated_scores_lr.npy', calibrated_scores_lr)

    labels_lr = np.hstack(labels_lr)

    # Evaluate the performance on pooled scores
    print('Logistic Regression - Calibrated:')
    print('minDCF (0.1) = %.3f' % compute_minDCF_binary_fast(calibrated_scores_lr, labels_lr, 0.1, 1.0, 1.0))
    print('actDCF (0.1) = %.3f' % compute_actDCF_binary_fast(calibrated_scores_lr, labels_lr, 0.1, 1.0, 1.0))

    # Plot calibrated Bayes error plots for Logistic Regression
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_lr, labels_lr)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(logOdds, minDCF, color='C1', linestyle='--', label='minDCF')
    ax.plot(logOdds, actDCF, color='C1', linestyle='-', label='actDCF')
    ax.set_ylim(0, 0.8)
    ax.legend()
    ax.set_title('Logistic Regression - validation - calibrated scores')
    plt.savefig('../results/plots/calibration/lr_calibrates.png')
    plt.show()


    # Calibration for SVM
    calibrated_scores_svm = []
    labels_svm = []

    # We train the calibration model for the prior pT = 0.1
    pT = 0.1

    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_svm, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
        # Apply the model to the validation fold
        calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the scores list
        calibrated_scores_svm.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        labels_svm.append(LVAL)

    # Re-build the score and label arrays (pooling)
    calibrated_scores_svm = np.hstack(calibrated_scores_svm)
    np.save('../results/scores/calibrated_scores_svm.npy', calibrated_scores_svm)

    labels_svm = np.hstack(labels_svm)

    # Evaluate the performance on pooled scores
    print('SVM - Calibrated:')
    print('minDCF (0.1) = %.3f' % compute_minDCF_binary_fast(calibrated_scores_svm, labels_svm, 0.1, 1.0, 1.0))
    print('actDCF (0.1) = %.3f' % compute_actDCF_binary_fast(calibrated_scores_svm, labels_svm, 0.1, 1.0, 1.0))

    # Plot calibrated Bayes error plots for SVM
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_svm, labels_svm)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(logOdds, minDCF, color='C2', linestyle='--', label='minDCF')
    ax.plot(logOdds, actDCF, color='C2', linestyle='-', label='actDCF')
    ax.set_ylim(0, 0.8)
    ax.legend()
    ax.set_title('SVM - validation - calibrated scores')
    plt.savefig('../results/plots/calibration/svm_calibrates.png')
    plt.show()

    # Calibration for GMM
    calibrated_scores_gmm = []
    labels_gmm = []

    # We train the calibration model for the prior pT = 0.1
    pT = 0.1

    for foldIdx in range(KFOLD):
        # keep 1 fold for validation, use the remaining ones for training
        SCAL, SVAL = extract_train_val_folds_from_ary(scores_gmm, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)
        # Train the model on the KFOLD - 1 training folds
        w, b = trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
        # Apply the model to the validation fold
        calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(pT / (1-pT))).ravel()
        # Add the scores of this validation fold to the scores list
        calibrated_scores_gmm.append(calibrated_SVAL)
        # Add the corresponding labels to preserve alignment between scores and labels
        labels_gmm.append(LVAL)

    # Re-build the score and label arrays (pooling)
    calibrated_scores_gmm = np.hstack(calibrated_scores_gmm)
    np.save('../results/scores/calibrated_scores_gmm.npy', calibrated_scores_gmm)
    labels_gmm = np.hstack(labels_gmm)

    # Evaluate the performance on pooled scores
    print('GMM - Calibrated:')
    print('minDCF (0.1) = %.3f' % compute_minDCF_binary_fast(calibrated_scores_gmm, labels_gmm, 0.1, 1.0, 1.0))
    print('actDCF (0.1) = %.3f' % compute_actDCF_binary_fast(calibrated_scores_gmm, labels_gmm, 0.1, 1.0, 1.0))

    # Plot calibrated Bayes error plots for GMM
    logOdds, actDCF, minDCF = bayesPlot(calibrated_scores_gmm, labels_gmm)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(logOdds, minDCF, color='C0', linestyle='--', label='minDCF')
    ax.plot(logOdds, actDCF, color='C0', linestyle='-', label='actDCF')
    ax.set_ylim(0, 0.8)
    ax.legend()
    ax.set_title('GMM - validation - calibrated scores')
    plt.savefig('../results/plots/calibration/gmm_calibrates.png')
    plt.show()

     # Fusion #
    fusedScores = []
    fusedLabels = []

    pT = 0.1

    for foldIdx in range(KFOLD):
        SCAL1, SVAL1 = extract_train_val_folds_from_ary(scores_lr, foldIdx)
        SCAL2, SVAL2 = extract_train_val_folds_from_ary(scores_svm, foldIdx)
        SCAL3, SVAL3 = extract_train_val_folds_from_ary(scores_gmm, foldIdx)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx)

        SCAL = np.vstack([SCAL1, SCAL2, SCAL3])
        w, b = trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)

        SVAL = np.vstack([SVAL1, SVAL2, SVAL3])
        fused_SVAL = (w.T @ SVAL + b - np.log(pT / (1-pT))).ravel()

        fusedScores.append(fused_SVAL)
        fusedLabels.append(LVAL)

    fusedScores = np.hstack(fusedScores)
    np.save('../results/scores/fused_scores.npy', fusedScores)
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
    plt.savefig('../results/plots/calibration/fusion_calibrates.png')
    plt.show()
    