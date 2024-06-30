import numpy as np
import matplotlib.pyplot as plt
from load import load
import pca as pca
import lda as lda

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

if __name__ == '__main__':
    D, L = load('../data/raw/trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # LDA without PCA
    ULDA = lda.compute_lda_JointDiag(DTR, LTR, m=1)
    DTR_lda = lda.apply_lda(ULDA, DTR)
    if DTR_lda[0, LTR == 0].mean() > DTR_lda[0, LTR == 1].mean():
        ULDA = -ULDA
        DTR_lda = lda.apply_lda(ULDA, DTR)
    DVAL_lda = lda.apply_lda(ULDA, DVAL)
    #threshold = 0
    threshold = (DTR_lda[0, LTR == 0].mean() + DTR_lda[0, LTR == 1].mean()) / 2.0
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0
    
    print('LDA without PCA:')
    print('Labels:     ', LVAL)
    print('Predictions:', PVAL)
    print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    print('Error rate: %.1f%%' % ((PVAL != LVAL).sum() / float(LVAL.size) * 100))

    # LDA with PCA
    for m in range(1, DTR.shape[0] + 1):
        UPCA = pca.compute_pca(DTR, m=m)
        DTR_pca = pca.apply_pca(UPCA, DTR)
        DVAL_pca = pca.apply_pca(UPCA, DVAL)
        ULDA = lda.compute_lda_JointDiag(DTR_pca, LTR, m=1)
        DTR_lda = lda.apply_lda(ULDA, DTR_pca)
        if DTR_lda[0, LTR == 0].mean() > DTR_lda[0, LTR == 1].mean():
            ULDA = -ULDA
            DTR_lda = lda.apply_lda(ULDA, DTR_pca)
        DVAL_lda = lda.apply_lda(ULDA, DVAL_pca)
        threshold = (DTR_lda[0, LTR == 0].mean() + DTR_lda[0, LTR == 1].mean()) / 2.0
        PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
        PVAL[DVAL_lda[0] >= threshold] = 1
        PVAL[DVAL_lda[0] < threshold] = 0
        error_rate = (PVAL != LVAL).sum() / float(LVAL.size) * 100
        print(f'LDA with PCA (m={m}):')
        print('Labels:     ', LVAL)
        print('Predictions:', PVAL)
        print('Number of errors:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
        print('Error rate: %.1f%%' % error_rate)

        
    plt.show()
