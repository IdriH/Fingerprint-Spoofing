import numpy

def vcol(x): # Same as in pca script
    return x.reshape((x.size, 1))

def vrow(x): # Same as in pca script
    return x.reshape((1, x.size))



def compute_mu_C(D): # Same as in pca script
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C



def print_basicStatistics(D,L,mu,C):
    print(D)
    print(L)

    print('Mean:')
    print(mu)
    print()

    print('Covariance:')
    print(C)
    print()

    var = D.var(1)
    std = D.std(1)
    print('Variance:', var)
    print('Std. dev.:', std)
    print()
    for cls in [0, 1]:
        print('Class', cls)
        DCls = D[:, L == cls]
        mu_cls = DCls.mean(1).reshape(DCls.shape[0], 1)
        print('Mean:')
        print(mu_cls)
        C_cls = ((DCls - mu_cls) @ (DCls - mu_cls).T) / float(DCls.shape[1])
        print('Covariance:')
        print(C_cls)
        var_cls = DCls.var(1)
        std_cls = DCls.std(1)
        print('Variance:', var_cls)
        print('Std. dev.:', std_cls)
        print()