import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from load import load
from evaluation import *

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

def train_dual_SVM_linear(DTR, LTR, C, K=1):
    ZTR = LTR * 2.0 - 1.0  # Convert labels to +1/-1
    DTR_EXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR_EXT.shape[1]), bounds=[(0, C) for i in LTR], factr=1.0)
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K
    return w, b

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

# We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
# The inner function will be able to access the arguments of the outer function
def polyKernel(degree, c):
    
    def polyKernelFunc(D1, D2):
        return (np.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)

    return rbfKernelFunc

# kernelFunc: function that computes the kernel matrix from two data matrices
def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):

    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

    print ('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0]))

    # Function to compute the scores for samples in DTE
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return fScore # we directly return the function to score a matrix of test samples

if __name__ == '__main__':

    D, L = load('../data/raw/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    """
    # Linear SVM Analysis
    C_values = np.logspace(-5, 0, 11)
    minDCFs = []
    actDCFs = []
    pi_T = 0.1

    for C in C_values:
        w, b = train_dual_SVM_linear(DTR, LTR, C, K=1.0)
        SVAL = (vrow(w) @ DVAL + b).ravel()
        minDCF = compute_minDCF_binary_fast(SVAL, LVAL, pi_T, 1.0, 1.0)
        actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(SVAL, LVAL, pi_T, 1.0, 1.0)
        minDCFs.append(minDCF)
        actDCFs.append(actDCF)

    plt.figure()
    plt.plot(C_values, actDCFs, label='Actual DCF', color='r')
    plt.plot(C_values, minDCFs, label='Minimum DCF', color='b')
    plt.xscale('log')
    plt.ylim([0, 1.1])
    plt.xlabel("C")
    plt.ylabel("DCF value")
    plt.title("DCF vs Lambda for Linear SVM")
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/plots/svm/linear_Svm_rawdata.png')
    plt.show()

    # Centered Data Analysis
    DTR_mean = np.mean(DTR, axis=1).reshape(-1, 1)
    DTR_centered = DTR - DTR_mean
    DVAL_centered = DVAL - DTR_mean

    minDCFs_centered = []
    actDCFs_centered = []

    for C in C_values:
        w, b = train_dual_SVM_linear(DTR_centered, LTR, C, K=1.0)
        SVAL = (vrow(w) @ DVAL_centered + b).ravel()
        minDCF = compute_minDCF_binary_fast(SVAL, LVAL, pi_T, 1.0, 1.0)
        actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(SVAL, LVAL, pi_T, 1.0, 1.0)
        minDCFs_centered.append(minDCF)
        actDCFs_centered.append(actDCF)

    plt.figure()
    plt.plot(C_values, actDCFs_centered, label='Actual DCF', color='r')
    plt.plot(C_values, minDCFs_centered, label='Minimum DCF', color='b')
    plt.xscale('log')
    plt.ylim([0, 1.1])
    plt.xlabel("C")
    plt.ylabel("DCF value")
    plt.title("DCF vs Lambda for Centered Linear SVM")
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/plots/svm/linear_Svm_centered_data.png')
    plt.show()

    C_values = np.logspace(-5, 0, 11)
    minDCF_values = []
    actDCF_values = []

    kernelFunc = polyKernel(2, 1)

    for C in C_values:
        fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=0.0)
        SVAL = fScore(DVAL)
        
        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size
        
        # Compute LLR-like scores
        SVAL_LLR = SVAL - np.log(pEmp / (1 - pEmp))
        
        # Compute minDCF and actDCF for pi_T = 0.1
        minDCF = compute_minDCF_binary_fast(SVAL_LLR, LVAL, 0.1, 1.0, 1.0)
        actDCF = compute_empirical_Bayes_risk_binary_llr_optimal_decisions(SVAL_LLR, LVAL, 0.1, 1.0, 1.0)
        
        minDCF_values.append(minDCF)
        actDCF_values.append(actDCF)

    plt.figure(figsize=(8, 6))
    plt.plot(C_values, minDCF_values, label='Minimum DCF', color='b')
    plt.plot(C_values, actDCF_values, label='Actual DCF', color='r')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('DCF value')
    plt.title('DCF vs C for Quadratic Polynomial Kernel SVM')
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/plots/svm/quadratic_polynomial_kernel_Svm.png')
    plt.show()
    """
    
    # Grid search values
    gamma_values = np.exp(np.arange(-4, 0, 1))
    C_values = np.logspace(-3, 2, 11)

    # Storage for results
    minDCF_results = {gamma: [] for gamma in gamma_values}
    actDCF_results = {gamma: [] for gamma in gamma_values}

    # Perform grid search
    for gamma in gamma_values:
        for C in C_values:
            kernelFunc = rbfKernel(gamma)
            fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps=1.0)
            SVAL = fScore(DVAL)
            
            # Compute minDCF and actDCF
            minDCF = compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
            actDCF = compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
            
            # Store results
            minDCF_results[gamma].append(minDCF)
            actDCF_results[gamma].append(actDCF)

    # Plot results
    plt.figure(figsize=(10, 8))

    for gamma in gamma_values:
        plt.plot(C_values, minDCF_results[gamma], label=f'γ={gamma} - minDCF', linestyle='-', marker='o')
        plt.plot(C_values, actDCF_results[gamma], label=f'γ={gamma} - actDCF', linestyle='--', marker='x')

    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('DCF value')
    plt.title('DCF vs C for RBF Kernel SVM')
    plt.legend()
    plt.grid(True)
    plt.show()