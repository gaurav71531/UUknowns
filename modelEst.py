import numpy as np
from scipy.io import loadmat, savemat
from fracModel import fracOrdUU
import os

def simMain(X):
#  
    meanX = np.mean(X, axis=1)
    X = X.T - meanX
    X = X.T
    fModel = fracOrdUU(verbose=1)
    fModel.fit(X)
    
    # correlation matrix (A) is obtained as:
    # fModel._AMat

    # unknown input matrix (B) is obtained as:
    # fModel._BMat

    # unknowns (u) are obtained as:
    # fModel._u

    # fractional orders are obtained as:
    print(fModel._order)
    return 1


if __name__ == '__main__':
    # sample input is prvided in the data directory
    data = loadmat(os.path.join('data', 'S001R03_edfm.mat'))
    K = 400 # number of samples
    sampleID = np.arange(0,K) + 5000-1 # taking the starting time to be after 5000 samples
    X = data['record'][:64,sampleID] # first 64 sensors
    #
    simMain(X)
    

