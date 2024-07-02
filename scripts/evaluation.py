import matplotlib.pyplot as plt
from gaussian_classifiers import *
import numpy as np
import scipy.special
from load import load

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def effective_prior(pi, Cfn, Cfp):
    return pi * Cfn / (pi * Cfn + (1 - pi) * Cfp)

def compute_posteriors(log_clas_conditional_ll, prior_array):
    logJoint = log_clas_conditional_ll + vcol(np.log(prior_array))
    logPost = logJoint - scipy.special.logsumexp(logJoint, 0)
    return np.exp(logPost)

def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = np.zeros((nClasses, nClasses), dtype=np.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

def compute_optimal_Bayes(posterior, costMatrix):
    expectedCosts = costMatrix @ posterior
    return np.argmin(expectedCosts, 0)

def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -np.log((prior * Cfn) / ((1 - prior) * Cfp))
    return np.int32(llr > th)

def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = np.argsort(llr)
    llrSorted = llr[llrSorter]
    classLabelsSorted = classLabels[llrSorter]

    Pfp = []
    Pfn = []
    
    nTrue = (classLabelsSorted == 1).sum()
    nFalse = (classLabelsSorted == 0).sum()
    nFalseNegative = 0
    nFalsePositive = nFalse
    
    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)
    
    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    llrSorted = np.concatenate([-np.array([np.inf]), llrSorted])

    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx+1] != llrSorted[idx]:
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])
            
    return np.array(PfnOut), np.array(PfpOut), np.array(thresholdsOut)

def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels)
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp
    if normalize:
        return bayesError / np.minimum(prior * Cfn, (1 - prior) * Cfp)
    return bayesError

def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / np.minimum(prior * Cfn, (1 - prior) * Cfp)
    idx = np.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]

def compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, classLabels, prior, Cfn, Cfp, normalize=True):
    predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
    return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=normalize)

compute_actDCF_binary_fast = compute_empirical_Bayes_risk_binary_llr_optimal_decisions

def compute_pca(D, m):
    mu = D.mean(axis=1, keepdims=True)
    DC = D - mu
    C = np.dot(DC, DC.T) / D.shape[1]
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, :m]
    return P

def apply_pca(D, P):
    return np.dot(P.T, D)

if __name__ == '__main__':
    D, L = load('../data/raw/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    """
    applications = [
        (0.5, 1.0, 1.0),
        (0.9, 1.0, 1.0),
        (0.1, 1.0, 1.0),
        (0.5, 1.0, 9.0),
        (0.5, 9.0, 1.0)
    ]

    effective_priors = [effective_prior(pi, Cfn, Cfp) for pi, Cfn, Cfp in applications]
    print("Effective priors:", effective_priors)

    # MVG Model
    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_MVG)
    LLR = S_logLikelihood[1] - S_logLikelihood[0]

    # Focus on the three applications with effective priors 0.1, 0.5, and 0.9
    effective_applications = [
        (effective_priors[0], 1.0, 1.0),
        (effective_priors[2], 1.0, 1.0),
        (effective_priors[1], 1.0, 1.0)
    ]

    for pi, Cfn, Cfp in effective_applications:
        print(f"\nApplication (pi={pi}, Cfn={Cfn}, Cfp={Cfp})")
        PRED = compute_optimal_Bayes_binary_llr(LLR, pi, Cfn, Cfp)
        error_rate_MVG = compute_error_rate(PRED, LVAL)
        DCF = compute_empirical_Bayes_risk_binary(PRED, LVAL, pi, Cfn, Cfp)
        minDCF, _ = compute_minDCF_binary_fast(LLR, LVAL, pi, Cfn, Cfp, returnThreshold=True)
        print(f"MVG - Error rate: {error_rate_MVG:.1f}%")
        print(f"MVG - DCF: {DCF:.3f}")
        print(f"MVG - minDCF: {minDCF:.3f}")

    # Tied Gaussian Model
    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)
    S_logLikelihood_Tied = compute_log_likelihood_Gau(DVAL, hParams_Tied)
    LLR_Tied = S_logLikelihood_Tied[1] - S_logLikelihood_Tied[0]

    for pi, Cfn, Cfp in effective_applications:
        print(f"\nApplication (pi={pi}, Cfn={Cfn}, Cfp={Cfp})")
        PRED_Tied = compute_optimal_Bayes_binary_llr(LLR_Tied, pi, Cfn, Cfp)
        error_rate_Tied = compute_error_rate(PRED_Tied, LVAL)
        DCF_Tied = compute_empirical_Bayes_risk_binary(PRED_Tied, LVAL, pi, Cfn, Cfp)
        minDCF_Tied, _ = compute_minDCF_binary_fast(LLR_Tied, LVAL, pi, Cfn, Cfp, returnThreshold=True)
        print(f"Tied Gaussian - Error rate: {error_rate_Tied:.1f}%")
        print(f"Tied Gaussian - DCF: {DCF_Tied:.3f}")
        print(f"Tied Gaussian - minDCF: {minDCF_Tied:.3f}")

    # Naive Bayes Model
    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)
    S_logLikelihood_Naive = compute_log_likelihood_Gau(DVAL, hParams_Naive)
    LLR_Naive = S_logLikelihood_Naive[1] - S_logLikelihood_Naive[0]

    for pi, Cfn, Cfp in effective_applications:
        print(f"\nApplication (pi={pi}, Cfn={Cfn}, Cfp={Cfp})")
        PRED_Naive = compute_optimal_Bayes_binary_llr(LLR_Naive, pi, Cfn, Cfp)
        error_rate_Naive = compute_error_rate(PRED_Naive, LVAL)
        DCF_Naive = compute_empirical_Bayes_risk_binary(PRED_Naive, LVAL, pi, Cfn, Cfp)
        minDCF_Naive, _ = compute_minDCF_binary_fast(LLR_Naive, LVAL, pi, Cfn, Cfp, returnThreshold=True)
        print(f"Naive Bayes - Error rate: {error_rate_Naive:.1f}%")
        print(f"Naive Bayes - DCF: {DCF_Naive:.3f}")
        print(f"Naive Bayes - minDCF: {minDCF_Naive:.3f}")

    # Repeat classification with PCA
    m_values = [2, 3, 4, 5]
    for m in m_values:
        print(f"\nPCA with m={m}")
        P = compute_pca(DTR, m)
        DTR_pca = apply_pca(DTR, P)
        DVAL_pca = apply_pca(DVAL, P)

        # MVG Model with PCA
        hParams_MVG_pca = Gau_MVG_ML_estimates(DTR_pca, LTR)
        S_logLikelihood_pca = compute_log_likelihood_Gau(DVAL_pca, hParams_MVG_pca)
        LLR_pca = S_logLikelihood_pca[1] - S_logLikelihood_pca[0]

        for pi, Cfn, Cfp in effective_applications:
            print(f"\nApplication (pi={pi}, Cfn={Cfn}, Cfp={Cfp})")
            PRED_pca = compute_optimal_Bayes_binary_llr(LLR_pca, pi, Cfn, Cfp)
            error_rate_MVG_pca = compute_error_rate(PRED_pca, LVAL)
            DCF_pca = compute_empirical_Bayes_risk_binary(PRED_pca, LVAL, pi, Cfn, Cfp)
            minDCF_pca, _ = compute_minDCF_binary_fast(LLR_pca, LVAL, pi, Cfn, Cfp, returnThreshold=True)
            print(f"MVG with PCA (m={m}) - Error rate: {error_rate_MVG_pca:.1f}%")
            print(f"MVG with PCA (m={m}) - DCF: {DCF_pca:.3f}")
            print(f"MVG with PCA (m={m}) - minDCF: {minDCF_pca:.3f}")

        # Tied Gaussian Model with PCA
        hParams_Tied_pca = Gau_Tied_ML_estimates(DTR_pca, LTR)
        S_logLikelihood_Tied_pca = compute_log_likelihood_Gau(DVAL_pca, hParams_Tied_pca)
        LLR_Tied_pca = S_logLikelihood_Tied_pca[1] - S_logLikelihood_Tied_pca[0]

        for pi, Cfn, Cfp in effective_applications:
            print(f"\nApplication (pi={pi}, Cfn={Cfn}, Cfp={Cfp})")
            PRED_Tied_pca = compute_optimal_Bayes_binary_llr(LLR_Tied_pca, pi, Cfn, Cfp)
            error_rate_Tied_pca = compute_error_rate(PRED_Tied_pca, LVAL)
            DCF_Tied_pca = compute_empirical_Bayes_risk_binary(PRED_Tied_pca, LVAL, pi, Cfn, Cfp)
            minDCF_Tied_pca, _ = compute_minDCF_binary_fast(LLR_Tied_pca, LVAL, pi, Cfn, Cfp, returnThreshold=True)
            print(f"Tied Gaussian with PCA (m={m}) - Error rate: {error_rate_Tied_pca:.1f}%")
            print(f"Tied Gaussian with PCA (m={m}) - DCF: {DCF_Tied_pca:.3f}")
            print(f"Tied Gaussian with PCA (m={m}) - minDCF: {minDCF_Tied_pca:.3f}")

        # Naive Bayes Model with PCA
        hParams_Naive_pca = Gau_Naive_ML_estimates(DTR_pca, LTR)
        S_logLikelihood_Naive_pca = compute_log_likelihood_Gau(DVAL_pca, hParams_Naive_pca)
        LLR_Naive_pca = S_logLikelihood_Naive_pca[1] - S_logLikelihood_Naive_pca[0]

        for pi, Cfn, Cfp in effective_applications:
            print(f"\nApplication (pi={pi}, Cfn={Cfn}, Cfp={Cfp})")
            PRED_Naive_pca = compute_optimal_Bayes_binary_llr(LLR_Naive_pca, pi, Cfn, Cfp)
            error_rate_Naive_pca = compute_error_rate(PRED_Naive_pca, LVAL)
            DCF_Naive_pca = compute_empirical_Bayes_risk_binary(PRED_Naive_pca, LVAL, pi, Cfn, Cfp)
            minDCF_Naive_pca, _ = compute_minDCF_binary_fast(LLR_Naive_pca, LVAL, pi, Cfn, Cfp, returnThreshold=True)
            print(f"Naive Bayes with PCA (m={m}) - Error rate: {error_rate_Naive_pca:.1f}%")
            print(f"Naive Bayes with PCA (m={m}) - DCF: {DCF_Naive_pca:.3f}")
            print(f"Naive Bayes with PCA (m={m}) - minDCF: {minDCF_Naive_pca:.3f}")
    """

    print("Pca m = 5 and logprioris")
    m = 5
    P = compute_pca(DTR, m)
    DTR_pca = apply_pca(DTR, P)
    DVAL_pca = apply_pca(DVAL, P)

   # Bayes error plot
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    Cfn =1.0
    Cfp = 1.0
    #normalized

    actDCF = []
    minDCF = []

    # MVG Model with PCA
    hParams_MVG_pca = Gau_MVG_ML_estimates(DTR_pca, LTR)
    S_logLikelihood_pca = compute_log_likelihood_Gau(DVAL_pca, hParams_MVG_pca)
    LLR_pca = S_logLikelihood_pca[1] - S_logLikelihood_pca[0]

    for effPrior in effPriors:

        PRED_pca = compute_optimal_Bayes_binary_llr(LLR_pca, effPrior, Cfn, Cfp)
        
        DCF_pca = compute_empirical_Bayes_risk_binary(PRED_pca, LVAL, effPrior, Cfn, Cfp)
        minDCF_pca, _ = compute_minDCF_binary_fast(LLR_pca, LVAL, effPrior, Cfn, Cfp, returnThreshold=True)
        
        actDCF.append(DCF_pca)
        minDCF.append(minDCF_pca)

    plt.figure(1)
    plt.plot(effPriorLogOdds, actDCF, label='Actual DCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='Minimum DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlabel("Prior log-odds")
    plt.ylabel("DCF value")
    plt.title("Bayes Plot for MVG Classifier with PCA (m=5)")
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/plots/evaulation/bayes_error_plots/MVG_bayes_plot.png')
    plt.show()

    ##Tied 
    actDCF = []
    minDCF = []

    hParams_Tied_pca = Gau_Tied_ML_estimates(DTR_pca, LTR)
    S_logLikelihood_Tied_pca = compute_log_likelihood_Gau(DVAL_pca, hParams_Tied_pca)
    LLR_Tied_pca = S_logLikelihood_Tied_pca[1] - S_logLikelihood_Tied_pca[0]

    for effPrior in effPriors:

        
        PRED_pca = compute_optimal_Bayes_binary_llr(LLR_Tied_pca, effPrior, Cfn, Cfp)
        
        DCF_pca = compute_empirical_Bayes_risk_binary(PRED_pca, LVAL, effPrior, Cfn, Cfp)
        minDCF_pca, _ = compute_minDCF_binary_fast(LLR_Tied_pca, LVAL, effPrior, Cfn, Cfp, returnThreshold=True)
        
        actDCF.append(DCF_pca)
        minDCF.append(minDCF_pca)

    plt.figure(1)
    plt.plot(effPriorLogOdds, actDCF, label='Actual DCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='Minimum DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlabel("Prior log-odds")
    plt.ylabel("DCF value")
    plt.title("Bayes Plot for Tied Classifier with PCA (m=5)")
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/plots/evaulation/bayes_error_plots/Tied_bayes_plot.png')
    plt.show()

    ## Naive 
    actDCF = []
    minDCF = []

    # Naive Bayes Model with PCA
    hParams_Naive_pca = Gau_Naive_ML_estimates(DTR_pca, LTR)
    S_logLikelihood_Naive_pca = compute_log_likelihood_Gau(DVAL_pca, hParams_Naive_pca)
    LLR_Naive_pca = S_logLikelihood_Naive_pca[1] - S_logLikelihood_Naive_pca[0]

    for effPrior in effPriors:

        
        PRED_pca = compute_optimal_Bayes_binary_llr(LLR_Naive_pca, effPrior, Cfn, Cfp)
        
        DCF_pca = compute_empirical_Bayes_risk_binary(PRED_pca, LVAL, effPrior, Cfn, Cfp)
        minDCF_pca, _ = compute_minDCF_binary_fast(LLR_Naive_pca, LVAL, effPrior, Cfn, Cfp, returnThreshold=True)
        
        actDCF.append(DCF_pca)
        minDCF.append(minDCF_pca)

    plt.figure(1)
    plt.plot(effPriorLogOdds, actDCF, label='Actual DCF', color='r')
    plt.plot(effPriorLogOdds, minDCF, label='Minimum DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlabel("Prior log-odds")
    plt.ylabel("DCF value")
    plt.title("Bayes  Plot for Naive Classifier with PCA (m=5)")
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/plots/evaulation/bayes_error_plots/Naive_bayes_plot.png')
    plt.show()