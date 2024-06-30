import numpy as np
import matplotlib.pyplot as plt
from load import load

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GAU_ND(X, mu, C):
    P = np.linalg.inv(C)
    log_det_C = np.linalg.slogdet(C)[1]
    M = X.shape[0]
    const_term = -0.5 * M * np.log(2 * np.pi)
    diff = X - mu
    log_density = const_term - 0.5 * log_det_C - 0.5 * (diff*(P @ diff)).sum(0)
    return log_density

def plot_gaussian_fit(D, L, feature_idx, class_label):
    D_class = D[:, L == class_label]
    feature_data = D_class[feature_idx, :]
    
    # Compute ML estimates
    mu = np.mean(feature_data)
    sigma = np.var(feature_data)
    
    # Plot histogram
    plt.figure()
    plt.hist(feature_data, bins=50, density=True, alpha=0.6, color='g')
    
    # Plot Gaussian density
    XPlot = np.linspace(feature_data.min(), feature_data.max(), 1000)
    gaussian_density = np.exp(logpdf_GAU_ND(vrow(XPlot), np.array([[mu]]), np.array([[sigma]])))
    plt.plot(XPlot, gaussian_density, color='r')
    plt.title(f'Feature {feature_idx + 1} - Class {class_label}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(f'../results/plots/density_estimation/Feature {feature_idx + 1} - Class {class_label}.png' )
    plt.show()

if __name__ == '__main__':
    D, L = load('../data/raw/trainData.txt')
    
    num_features = D.shape[0]
    class_labels = np.unique(L)
    
    for class_label in class_labels:
        for feature_idx in range(num_features):
            plot_gaussian_fit(D, L, feature_idx, class_label)

