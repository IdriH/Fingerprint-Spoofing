import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from load import load  # Assuming you have a function to load the project data
from evaluation import *

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5 * x.shape[0] * np.log(np.pi * 2) - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * ((x - mu) * (P @ (x - mu))).sum(0)

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GMM(X, gmm):
    S = []
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + np.log(w)
        S.append(logpdf_joint)
    S = np.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens

def smooth_covariance_matrix(C, psi):
    U, s, Vh = np.linalg.svd(C)
    s[s < psi] = psi
    CUpd = U @ (vcol(s) * U.T)
    return CUpd

def train_GMM_EM_Iteration(X, gmm, covType='Full', psiEig=None):
    assert (covType.lower() in ['full', 'diagonal', 'tied'])
    S = []
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + np.log(w)
        S.append(logpdf_joint)
    S = np.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    gammaAllComponents = np.exp(S - logdens)
    gmmUpd = []
    for gIdx in range(len(gmm)):
        gamma = gammaAllComponents[gIdx]
        Z = gamma.sum()
        F = vcol((vrow(gamma) * X).sum(1))
        S = (vrow(gamma) * X) @ X.T
        muUpd = F / Z
        CUpd = S / Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if covType.lower() == 'diagonal':
            CUpd = CUpd * np.eye(X.shape[0])
        if psiEig is not None:
            CUpd = smooth_covariance_matrix(CUpd, psiEig)
        gmmUpd.append((wUpd, muUpd, CUpd))
    if covType.lower() == 'tied':
        CTied = sum(w * C for w, mu, C in gmmUpd)
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]
    return gmmUpd

def train_GMM_EM(X, gmm, covType='Full', psiEig=None, epsLLAverage=1e-6, verbose=True):
    llOld = logpdf_GMM(X, gmm).mean()
    llDelta = None
    if verbose:
        print('GMM - it %3d - average ll %.8e' % (0, llOld))
    it = 1
    while (llDelta is None or llDelta > epsLLAverage):
        gmmUpd = train_GMM_EM_Iteration(X, gmm, covType=covType, psiEig=psiEig)
        llUpd = logpdf_GMM(X, gmmUpd).mean()
        llDelta = llUpd - llOld
        if verbose:
            print('GMM - it %3d - average ll %.8e' % (it, llUpd))
        gmm = gmmUpd
        llOld = llUpd
        it += 1
    if verbose:
        print('GMM - it %3d - average ll %.8e (eps = %e)' % (it, llUpd, epsLLAverage))
    return gmm

def split_GMM_LBG(gmm, alpha=0.1, verbose=True):
    gmmOut = []
    if verbose:
        print('LBG - going from %d to %d components' % (len(gmm), len(gmm) * 2))
    for (w, mu, C) in gmm:
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut

def train_GMM_LBG_EM(X, numComponents, covType='Full', psiEig=None, epsLLAverage=1e-6, lbgAlpha=0.1, verbose=True):
    mu, C = compute_mu_C(X)
    if covType.lower() == 'diagonal':
        C = C * np.eye(X.shape[0])
    if psiEig is not None:
        gmm = [(1.0, mu, smooth_covariance_matrix(C, psiEig))]
    else:
        gmm = [(1.0, mu, C)]
    while len(gmm) < numComponents:
        if verbose:
            print('Average ll before LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        gmm = split_GMM_LBG(gmm, lbgAlpha, verbose=verbose)
        if verbose:
            print('Average ll after LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        gmm = train_GMM_EM(X, gmm, covType=covType, psiEig=psiEig, verbose=verbose, epsLLAverage=epsLLAverage)
    return gmm

def evaluate_GMM_components(DTR, LTR, DVAL, LVAL, component_range, covType='Full', psiEig=None, prior=0.1, Cfn=1.0, Cfp=1.0):
    minDCF_values = []
    actDCF_values = []
    for numComponents in component_range:
        gmm_0 = train_GMM_LBG_EM(DTR[:, LTR == 0], numComponents, covType=covType, psiEig=psiEig)
        gmm_1 = train_GMM_LBG_EM(DTR[:, LTR == 1], numComponents, covType=covType, psiEig=psiEig)
        logS0 = logpdf_GMM(DVAL, gmm_0)
        logS1 = logpdf_GMM(DVAL, gmm_1)
        llr = logS1 - logS0
        minDCF = compute_minDCF_binary_fast(llr, LVAL, prior, Cfn, Cfp)
        actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, LVAL, prior, Cfn, Cfp)
        minDCF_values.append(minDCF)
        actDCF_values.append(actDCF)
        
    return actDCF_values, minDCF_values

if __name__ == '__main__':
    # Load the project data
    D, L = load('../data/raw/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    component_range = [2, 4, 8, 16, 32]
    covTypes = ['Full', 'Diagonal']
    prior = 0.1
    Cfn = 1.0
    Cfp = 1.0
    psiEig = 1e-2  # Small value to ensure positive definiteness

    for covType in covTypes:
        actDCF_values, minDCF_values = evaluate_GMM_components(DTR, LTR, DVAL, LVAL, component_range, covType=covType, psiEig=psiEig, prior=prior, Cfn=Cfn, Cfp=Cfp)

        # Plot the results
        plt.figure(figsize=(8, 6))
        plt.plot(component_range, minDCF_values, label='Minimum DCF', color='b')
        plt.plot(component_range, actDCF_values, label='Actual DCF', color='r')
        plt.xscale('log', base=2)
        plt.xlabel('Number of Components')
        plt.ylabel('DCF value')
        plt.title(f'DCF vs Number of Components for GMM ({covType} Covariance)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'../results/plots/evaulation/GMM_{covType}_covariance.png')
        plt.show()
