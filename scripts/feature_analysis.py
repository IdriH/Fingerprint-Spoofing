import numpy
import matplotlib 
import matplotlib.pyplot as plt 



def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
        0: 'feature 1',
        1: 'feature 2',
        2: 'feature 3',
        3: 'feature 4',
        4: 'feature 5',
        5: 'feature 6'
        }

    for dIdx in range(6):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'False')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'True')
      
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('../results/plots/feature_analysis/hist_%d.png' % dIdx)
    plt.show()

def plot_scatter(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
        0: 'feature 1',
        1: 'feature 2',
        2: 'feature 3',
        3: 'feature 4',
        4: 'feature 5',
        5: 'feature 6'
        }

    for dIdx1 in range(6):
        for dIdx2 in range(6):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Authentic')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Spoofed')
            
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('../results/plots/feature_analysis/scatter_%d_%d.png' % (dIdx1, dIdx2))
        plt.show()