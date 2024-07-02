

import numpy
import scipy.special
import sklearn.datasets
from evaluation import *
from load import load

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

# Optimize the logistic regression loss
def trainLogRegBinary(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

# Optimize the weighted logistic regression loss
def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    
    wTrue = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wFalse = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTrue # Apply the weights to the loss computations
        loss[ZTR<0] *= wFalse

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wTrue # Apply the weights to the gradient computations
        G[ZTR < 0] *= wFalse
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

def expand_features_quadratic(D):
    n = D.shape[0]
    new_features = []
    for i in range(n):
        for j in range(i, n):
            new_features.append(D[i] * D[j])
    return np.vstack([D] + new_features)

def center_data(DTR, DVAL):
    mean_DTR = np.mean(DTR, axis=1, keepdims=True)
    DTR_centered = DTR - mean_DTR
    DVAL_centered = DVAL - mean_DTR
    return DTR_centered, DVAL_centered




if __name__ == '__main__':
    D, L = load('../data/raw/trainData.txt')  # Assuming you have a function to load the data
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    
    lambdas = np.logspace(-4, 2, 13)
    pi_T = 0.1
    Cfn = 1.0
    Cfp = 1.0

    actDCFs = []
    minDCFs = []

    for lamb in lambdas:
        w, b = trainLogRegBinary(DTR, LTR, lamb)  # Train model
        sVal = np.dot(w.T, DVAL) + b  # Compute validation scores
        PVAL = (sVal > 0) * 1  # Predict validation labels
        err = (PVAL != LVAL).sum() / float(LVAL.size)
        print('Error rate: %.1f' % (err*100))
        
        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size
        
        # Compute LLR-like scores
        sValLLR = sVal - np.log(pEmp / (1 - pEmp))
        
        # Compute actual and minimum DCF for the primary application pi_T = 0.1
        actDCF = compute_actDCF_binary_fast(sValLLR, LVAL, pi_T, Cfn, Cfp)
        minDCF = compute_minDCF_binary_fast(sValLLR, LVAL, pi_T, Cfn, Cfp)

        
        actDCFs.append(actDCF)
        minDCFs.append(minDCF)
  
    # Plot the results
    plt.figure()
    plt.plot(lambdas, actDCFs, label='Actual DCF', color='r')
    plt.plot(lambdas, minDCFs, label='Minimum DCF', color='b')
    plt.xscale('log', base=10)
    plt.xlabel("Lambda")
    plt.ylabel("DCF value")
    plt.title("DCF vs Lambda for Logistic Regression")
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/plots/logistic_regression/unweighted_model_train1.png')
    plt.show()


    ##downsampled trainingset 

# Downsample the training data
DTR_downsampled = DTR[:, ::50]
LTR_downsampled = LTR[::50]

# Define the range of lambda values to test
lambda_values = np.logspace(-4, 2, 13)

# Initialize lists to store DCF values
actual_DCFs = []
minimum_DCFs = []

# Compute empirical prior for the downsampled training set
pEmp = (LTR_downsampled == 1).sum() / LTR_downsampled.size

# Loop over lambda values
for lamb in lambda_values:
    # Train the logistic regression model on downsampled training data
    w, b = trainLogRegBinary(DTR_downsampled, LTR_downsampled, lamb)
    
    # Compute validation scores
    sVal = np.dot(w.T, DVAL) + b
    
    # Compute LLR-like scores
    sValLLR = sVal - np.log(pEmp / (1 - pEmp))
    
    # Compute actual and minimum DCF for πT = 0.1
    actual_DCF = compute_actDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0)
    minimum_DCF = compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, 1.0, 1.0)
   
    # Store the DCF values
    actual_DCFs.append(actual_DCF)
    minimum_DCFs.append(minimum_DCF)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(lambda_values, actual_DCFs, label='Actual DCF', color='r')
plt.plot(lambda_values, minimum_DCFs, label='Minimum DCF', color='b')
plt.xscale('log', base=10)
plt.ylim([0, 1.1])
plt.xlabel("Lambda")
plt.ylabel("DCF value")
plt.title("DCF vs Lambda for Logistic Regression with Downsampled Training Data")
plt.legend()
plt.grid(True)
plt.savefig('../results/plots/logistic_regression/unweighted_model_train_downsampled.png')
plt.show()


##weighted logistic regression

lambdas = np.logspace(-4, 2, 13)
pT = 0.1  # Target prior

actDCF_values = []
minDCF_values = []

for lamb in lambdas:
    w, b = trainWeightedLogRegBinary(DTR, LTR, lamb, pT)
    sVal = np.dot(w.T, DVAL) + b
    sValLLR = sVal - np.log(pT / (1 - pT))

    actDCF = compute_actDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)
    minDCF = compute_minDCF_binary_fast(sValLLR, LVAL, pT, 1.0, 1.0)

    actDCF_values.append(actDCF)
    minDCF_values.append(minDCF)

    

plt.figure()
plt.plot(lambdas, actDCF_values, label='Actual DCF', color='r')
plt.plot(lambdas, minDCF_values, label='Minimum DCF', color='b')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('DCF value')
plt.title('DCF vs Lambda for Prior-Weighted Logistic Regression')
plt.legend()
plt.grid(True)
plt.savefig('../results/plots/logistic_regression/weighted_model_train.png')
plt.show()


DTR_exp = expand_features_quadratic(DTR)
DVAL_exp = expand_features_quadratic(DVAL)

# 2. Train and evaluate the model for different values of lambda
lambdas = np.logspace(-4, 2, 13)
actual_DCFs = []
minimum_DCFs = []

for lamb in lambdas:
    w, b = trainLogRegBinary(DTR_exp, LTR, lamb)  # Train model
    sVal = np.dot(w.T, DVAL_exp) + b  # Compute validation scores
    pEmp = (LTR == 1).sum() / LTR.size  # Compute empirical prior
    sValLLR = sVal - np.log(pEmp / (1 - pEmp))  # Compute LLR-like scores
    
    # Compute actual DCF
    act_DCF = compute_actDCF_binary_fast(sValLLR, LVAL, pT, Cfn=1.0, Cfp=1.0)
    # Compute minimum DCF
    min_DCF = compute_minDCF_binary_fast(sValLLR, LVAL, pT, Cfn=1.0, Cfp=1.0)
    
    actual_DCFs.append(act_DCF)
    minimum_DCFs.append(min_DCF)
    


# 3. Plot the results
plt.figure()
plt.plot(lambdas, actual_DCFs, label='Actual DCF', color='r')
plt.plot(lambdas, minimum_DCFs, label='Minimum DCF', color='b')
plt.xscale('log', base=10)
plt.xlabel('Lambda')
plt.ylabel('DCF value')
plt.title('DCF vs Lambda for Quadratic Logistic Regression')
plt.legend()
plt.grid(True)
plt.ylim([0, 1.1])
plt.savefig('../results/plots/logistic_regression/unweighterd_qudratic.png')
plt.show()

DTR_centered, DVAL_centered = center_data(DTR, DVAL)

# Train and evaluate the model for different values of lambda using centered data
lambdas = np.logspace(-4, 2, 13)
actual_DCFs_centered = []
minimum_DCFs_centered = []

for lamb in lambdas:
    w, b = trainLogRegBinary(DTR_centered, LTR, lamb)  # Train model
    sVal = np.dot(w.T, DVAL_centered) + b  # Compute validation scores
    pEmp = (LTR == 1).sum() / LTR.size  # Compute empirical prior
    sValLLR = sVal - np.log(pEmp / (1 - pEmp))  # Compute LLR-like scores
    
    # Compute actual DCF
    act_DCF = compute_actDCF_binary_fast(sValLLR, LVAL, 0.1, Cfn=1.0, Cfp=1.0)
    # Compute minimum DCF
    min_DCF = compute_minDCF_binary_fast(sValLLR, LVAL, 0.1, Cfn=1.0, Cfp=1.0)
    
    actual_DCFs_centered.append(act_DCF)
    minimum_DCFs_centered.append(min_DCF)

# Plot the results
plt.figure()
plt.plot(lambdas, actual_DCFs_centered, label='Actual DCF (Centered)', color='r')
plt.plot(lambdas, minimum_DCFs_centered, label='Minimum DCF (Centered)', color='b')
plt.xscale('log', base=10)
plt.xlabel('Lambda')
plt.ylabel('DCF value')
plt.title('DCF vs Lambda for Centered Logistic Regression')
plt.legend()
plt.grid(True)
plt.ylim([0, 1.1])
plt.savefig('../results/plots/logistic_regression/centered_dataset_logisticRegression.png')
plt.show()