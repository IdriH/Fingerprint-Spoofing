import numpy as np
import scipy.special
from load import load  # Assuming you have a function to load the project data

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

# MVG model
def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        hParams[lab] = compute_mu_C(DX)
    return hParams

# Naive Bayes model
def Gau_Naive_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, C * np.eye(D.shape[0]))
    return hParams

# Tied Gaussian model
def Gau_Tied_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    hMeans = {}
    CGlobal = 0
    for lab in labelSet:
        DX = D[:, L==lab]
        mu, C_class = compute_mu_C(DX)
        CGlobal += C_class * DX.shape[1]
        hMeans[lab] = mu
    CGlobal = CGlobal / D.shape[1]
    for lab in labelSet:
        hParams[lab] = (hMeans[lab], CGlobal)
    return hParams

# Compute per-class log-densities
def compute_log_likelihood_Gau(D, hParams):
    S = np.zeros((len(hParams), D.shape[1]))
    for lab in range(S.shape[0]):
        S[lab, :] = logpdf_GAU_ND(D, hParams[lab][0], hParams[lab][1])
    return S

# Compute log-posterior matrix
def compute_logPosterior(S_logLikelihood, v_prior):
    SJoint = S_logLikelihood + vcol(np.log(v_prior))
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    SPost = SJoint - SMarginal
    return SPost

# Compute predictions from LLRs
def compute_predictions(LLR, threshold=0):
    PVAL = np.zeros(LLR.shape, dtype=np.int32)
    PVAL[LLR >= threshold] = 1
    return PVAL

# Compute error rate from predictions
def compute_error_rate(PRED, L):
    return (PRED != L).sum() / float(L.size) * 100

if __name__ == '__main__':
    D, L = load('../data/raw/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # MVG Model
    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_MVG)
    LLR = S_logLikelihood[1] - S_logLikelihood[0]
    PRED = compute_predictions(LLR)
    error_rate_MVG = compute_error_rate(PRED, LVAL)
    print("MVG - Error rate: %.1f%%" % error_rate_MVG)

    # Tied Gaussian Model
    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_Tied)
    LLR = S_logLikelihood[1] - S_logLikelihood[0]
    PRED = compute_predictions(LLR)
    error_rate_Tied = compute_error_rate(PRED, LVAL)
    print("Tied Gaussian - Error rate: %.1f%%" % error_rate_Tied)

    # Naive Bayes Model
    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_Naive)
    LLR = S_logLikelihood[1] - S_logLikelihood[0]
    PRED = compute_predictions(LLR)
    error_rate_Naive = compute_error_rate(PRED, LVAL)
    print("Naive Bayes - Error rate: %.1f%%" % error_rate_Naive)

    # Analyze covariance and correlation matrices
    for lab in [0, 1]:
        print(f'Class {lab} covariance matrix:')
        print(hParams_MVG[lab][1])
        Corr = hParams_MVG[lab][1] / (vcol(hParams_MVG[lab][1].diagonal()**0.5) * vrow(hParams_MVG[lab][1].diagonal()**0.5))
        print(f'Class {lab} correlation matrix:')
        print(Corr)
        print()

    # Repeat classification using only features 1-4
    DTR_reduced = DTR[:4, :]
    DVAL_reduced = DVAL[:4, :]

    #MVG
    hParams_MVG_reduced = Gau_MVG_ML_estimates(DTR_reduced, LTR)
    S_logLikelihood_reduced = compute_log_likelihood_Gau(DVAL_reduced, hParams_MVG_reduced)
    LLR_reduced = S_logLikelihood_reduced[1] - S_logLikelihood_reduced[0]
    PRED_reduced = compute_predictions(LLR_reduced)
    error_rate_MVG_reduced = compute_error_rate(PRED_reduced, LVAL)
    print("MVG (features 1-4) - Error rate: %.1f%%" % error_rate_MVG_reduced)

    # Tied Gaussian Model
    hParams_Tied = Gau_Tied_ML_estimates(DTR_reduced, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL_reduced, hParams_Tied)
    LLR = S_logLikelihood[1] - S_logLikelihood[0]
    PRED = compute_predictions(LLR)
    error_rate_Tied = compute_error_rate(PRED, LVAL)
    print("Tied Gaussian (features 1-4)  - Error rate: %.1f%%" % error_rate_Tied)

    # Naive Bayes Model
    hParams_Naive = Gau_Naive_ML_estimates(DTR_reduced, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL_reduced, hParams_Naive)
    LLR = S_logLikelihood[1] - S_logLikelihood[0]
    PRED = compute_predictions(LLR)
    error_rate_Naive = compute_error_rate(PRED, LVAL)
    print("Naive Bayes (features 1-4)  - Error rate: %.1f%%" % error_rate_Naive)


      # Repeat classification using only features 1-2
    DTR_1_2 = DTR[:2, :]
    DVAL_1_2 = DVAL[:2, :]

     #MVG with features 1-2
    hParams_MVG_1_2 = Gau_MVG_ML_estimates(DTR_1_2, LTR)
    S_logLikelihood_1_2 = compute_log_likelihood_Gau(DVAL_1_2, hParams_MVG_1_2)
    LLR_1_2 = S_logLikelihood_1_2[1] - S_logLikelihood_1_2[0]
    PRED_1_2 = compute_predictions(LLR_1_2)
    error_rate_MVG_1_2 = compute_error_rate(PRED_1_2, LVAL)
    print("MVG (features 1-2) - Error rate: %.1f%%" % error_rate_MVG_1_2)

    # Tied Gaussian Model with features 1-2
    hParams_Tied_1_2 = Gau_Tied_ML_estimates(DTR_1_2, LTR)
    S_logLikelihood_1_2 = compute_log_likelihood_Gau(DVAL_1_2, hParams_Tied_1_2)
    LLR_1_2 = S_logLikelihood_1_2[1] - S_logLikelihood_1_2[0]
    PRED_1_2 = compute_predictions(LLR_1_2)
    error_rate_Tied_1_2 = compute_error_rate(PRED_1_2, LVAL)
    print("Tied Gaussian (features 1-2) - Error rate: %.1f%%" % error_rate_Tied_1_2)

    # Repeat classification using only features 3-4
    DTR_3_4 = DTR[2:4, :]
    DVAL_3_4 = DVAL[2:4, :]

    #MVG with features 3-4
    hParams_MVG_3_4 = Gau_MVG_ML_estimates(DTR_3_4, LTR)
    S_logLikelihood_3_4 = compute_log_likelihood_Gau(DVAL_3_4, hParams_MVG_3_4)
    LLR_3_4 = S_logLikelihood_3_4[1] - S_logLikelihood_3_4[0]
    PRED_3_4 = compute_predictions(LLR_3_4)
    error_rate_MVG_3_4 = compute_error_rate(PRED_3_4, LVAL)
    print("MVG (features 3-4) - Error rate: %.1f%%" % error_rate_MVG_3_4)

    # Tied Gaussian Model with features 3-4
    hParams_Tied_3_4 = Gau_Tied_ML_estimates(DTR_3_4, LTR)
    S_logLikelihood_3_4 = compute_log_likelihood_Gau(DVAL_3_4, hParams_Tied_3_4)
    LLR_3_4 = S_logLikelihood_3_4[1] - S_logLikelihood_3_4[0]
    PRED_3_4 = compute_predictions(LLR_3_4)
    error_rate_Tied_3_4 = compute_error_rate(PRED_3_4, LVAL)
    print("Tied Gaussian (features 3-4) - Error rate: %.1f%%" % error_rate_Tied_3_4)

    # PCA and Classification
    import pca
    m = 4  # Number of principal components
    UPCA = pca.compute_pca(DTR, m=m)
    DTR_pca = pca.apply_pca(UPCA, DTR)
    DVAL_pca = pca.apply_pca(UPCA, DVAL)

    # MVG Model with PCA
    hParams_MVG_pca = Gau_MVG_ML_estimates(DTR_pca, LTR)
    S_logLikelihood_pca = compute_log_likelihood_Gau(DVAL_pca, hParams_MVG_pca)
    LLR_pca = S_logLikelihood_pca[1] - S_logLikelihood_pca[0]
    PRED_pca = compute_predictions(LLR_pca)
    error_rate_MVG_pca = compute_error_rate(PRED_pca, LVAL)
    print("MVG with PCA - Error rate: %.1f%%" % error_rate_MVG_pca)

    # Tied Gaussian Model
    hParams_Tied = Gau_Tied_ML_estimates(DTR_pca, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL_pca, hParams_Tied)
    LLR = S_logLikelihood[1] - S_logLikelihood[0]
    PRED = compute_predictions(LLR)
    error_rate_Tied = compute_error_rate(PRED, LVAL)
    print("Tied Gaussian with PCA  - Error rate: %.1f%%" % error_rate_Tied)

    # Naive Bayes Model
    hParams_Naive = Gau_Naive_ML_estimates(DTR_pca, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL_pca, hParams_Naive)
    LLR = S_logLikelihood[1] - S_logLikelihood[0]
    PRED = compute_predictions(LLR)
    error_rate_Naive = compute_error_rate(PRED, LVAL)
    print("Naive Bayes with PCA  - Error rate: %.1f%%" % error_rate_Naive)