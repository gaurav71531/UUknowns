import numpy as np
from scipy.special import gamma
import scipy.linalg as LA
import scipy.sparse as spSparse
import scipy.sparse.linalg as sLA


import time

class HaarWaveletTransform(object):
    def __init__(self, X):
        self._N = np.shape(X)
        self.X = np.array(X)
        try:
            if np.size(self._N)==1:
                self._N = self._N[0]
            elif np.size(self._N)>1:
                if self._N[0] == 1 or self._N[1]==1:
                    self.X = np.squeeze(X)
                    self._N = np.size(self.X)     
                else:
                    raise Exception('dimErr')
        except Exception as err:
            errStatus = err.args[0]
            if errStatus == 'dimErr':
                print('Only single dimensional arrays are acceptable')

    def normalize(self):
        mean = np.mean(self.X)
        self.X -= mean

    def _dwthaar(self, Signal):
        NUse = int(np.floor(np.size(Signal)/2))
        C = (Signal[:2*NUse:2] + Signal[1:2*NUse:2])/2
        S = Signal[:2*NUse:2] - C

        C = 2 * C / np.sqrt(2)
        S = -2 * S / np.sqrt(2)
        return C, S

    def transform(self):
        Nby2 = int(np.floor(self._N/2))
        W = np.zeros((Nby2,Nby2))
        D = np.zeros((Nby2,Nby2))
        j = self._N
        Signal = self.X
        for i in range(int(np.floor(np.log2(self._N)))):
            j = int(np.floor(j/2))
            w, d = self._dwthaar(Signal)
            W[i,:j] = w
            D[i,:j] = d
            Signal = w
        return W, D


class fracOrdUU(object):
    def __init__(self, numInp=[], numFract = 20, niter = 10, B = [], lambdaUse=0.5, verbose=0):
        self.verbose=verbose
        self._order = []
        self._numCh = []
        self._K = []
        self._numInp = numInp
        self._numFract = numFract
        self._lambdaUse = lambdaUse
        self._niter = niter
        self._BMat = B
        self._zVec = []
        self._AMat = []
        self._u = []
        self._performSparseComputation = False
        self._preComputedVars = []

        # if np.size(B)>0:
        #     if numInp > 0:
        #         if numInp != np.shape(B)[1] :
        #             print('size of B should be consistent with the number of unknown inputs')


    def _getFractionalOrder(self, x):
        numScales = int(np.floor(np.log2(self._K)))
        log_wavelet_scales = np.zeros((numScales,))
        scale = np.arange(1,numScales+1)

        Wt = HaarWaveletTransform(x)
        Wt.normalize()
        _, W = Wt.transform()
        j = int(np.floor(self._K/2))
        for i in range(numScales-1):
            y = W[i,:j]
            variance = np.var(y, ddof=1) # for unbiased estimate
            log_wavelet_scales[i] = np.log2(variance)
            j = int(np.floor(j/2))
        p = np.polyfit(scale[:numScales-1], log_wavelet_scales[:numScales-1], 1)
        return p[0]/2


    def _estimateOrder(self, X):
        self._order = np.empty((self._numCh,))

        for i in range(self._numCh):
            self._order[i] = self._getFractionalOrder(X[i,:])

    def _updateZVec(self, X):
        self._zVec = np.empty((self._numCh, self._K))
        j = np.arange(0,self._numFract+1)
        for i in range(self._numCh):
            preFactVec = gamma(-self._order[i]+j)/gamma(-self._order[i]) / gamma(j+1)
            y = np.convolve(X[i,:], preFactVec)
            self._zVec[i,:] = y[:self._K]

    def _setHeuristicBMat(self, A):
        B = np.zeros((self._numCh, self._numCh))
        B[np.abs(A)>0.01] = A[np.abs(A)>0.01]
        _, r = LA.qr(B)
        colInd = np.where(np.abs(np.diag(r))>1e-7)
        if np.size(colInd[0])<self._numInp:
            self._BMat = np.vstack((np.eye(self._numInp), 
                        np.zeros((self._numCh-self._numInp, self._numInp))))
        else:
            colInd = colInd[0][:self._numInp]
            self._BMat = B[:,colInd]
        if np.linalg.matrix_rank(B) < self._numInp:
            raise Exception('rank deficient B')


    def _performLeastSq(self, Y, X):
        # X and Y are shape of (K,numCh)
        # A = [a1, a2,...,an]
        # Y = X*A.T + E
        # ai = Sigma_X^-1 * E[Xyi.T]

        XUse = np.vstack((np.zeros((1,self._numCh)), X[:-1,:]))
        A = np.matmul(np.matmul(Y.T, XUse),  LA.inv(np.matmul(XUse.T, XUse)))
        mse = LA.norm(Y - np.matmul(XUse, A.T),axis=0)**2 / self._K
        return A, np.mean(mse)

    def _factor(self, A, rho):
        m, n = np.shape(A)
        if self._performSparseComputation:
            if m >= n:
                L = LA.cholesky(np.matmul(A.T, A) + rho*spSparse.eye(n), lower=True)
            else:
                L = LA.cholesky(spSparse.eye(m) + 1/rho * np.matmul(A, A.T), lower=True)
            
            L = spSparse.csc_matrix(L)
            U = spSparse.csc_matrix(L.T)
        else:
            if m >= n:
                L = LA.cholesky(np.matmul(A.T, A) + rho*np.eye(n), lower=True)
            else:
                L = LA.cholesky(np.eye(m) + 1/rho * np.matmul(A, A.T), lower=True)
            U = L.T
        return L, U
    
    def _shrinkage(self, x, kappa):
        return np.maximum(0, x-kappa) - np.maximum(0, -x - kappa)

    def _objective(self, A, b, lambdaUse, x, z):
        return 0.5 * np.sum((np.matmul(A, x)-b)**2) + lambdaUse*LA.norm(z,ord=1)

    class _history(object):
        def __init__(self, N):
            self._objval = np.empty((N,))
            self._r_norm = np.empty((N,))
            self._s_norm = np.empty((N,))
            self._eps_pri = np.empty((N,))
            self._eps_dual = np.empty((N,))

    
    class _preComputedVars_(object):
        def __init__(self):
            self._lasso_L = []
            self._lasso_U = []
            self._lasso_LInv = []
            self._lasso_UInv = []

        def _updateLassoLUMat(self, A, rho):
            self._lasso_L, self._lasso_U = fracOrdUU()._factor(A, rho)
            self._lasso_LInv = LA.inv(self._lasso_L)
            self._lasso_UInv = LA.inv(self._lasso_U)


    def _getLassoSoln(self, b, lambdaUse):

        # code borrowed from 
        # https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html 
        A = self._BMat
        b = np.reshape(b, (np.size(b),1))
        MAX_ITER = 100
        ABSTOL = 1e-4
        RELTOL = 1e-2

        m, n = np.shape(A)
        Atb = np.matmul(A.T, b)
        rho = 1/lambdaUse
        alpha = 1

        z = np.zeros((n,1))
        u = np.zeros((n,1))
        # L, U = self._factor(A, rho)
        LInv = self._preComputedVars._lasso_LInv
        UInv = self._preComputedVars._lasso_UInv

        history = self._history(MAX_ITER)
        for k in range(MAX_ITER):
            # x-update
            q = Atb + rho * (z-u)
            if self._performSparseComputation:
                if m >= n: # is skinny
                    x = sLA.inv(U) *  (sLA.inv(L) * q)
                else: # if fat
                    x = q/rho - np.matmul(A.T, sLA.inv(U) *  
                        (sLA.inv(L) * np.matmul(A, q)))/rho**2
            else:
                if m >= n: # is skinny
                    x = np.matmul(UInv, np.matmul(LInv, q))
                    # x = LA.solve(U, LA.solve(L, q))
                else: # if fat
                    x = q/rho - np.matmul(A.T, np.matmul(LA.inv(U),   
                        np.matmul(LA.inv(L), np.matmul(A, q))))/rho**2

            zold = np.array(z)
            x_hat = alpha*x + (1-alpha)*zold
            z = self._shrinkage(x_hat + u, lambdaUse/rho)

            # u-update
            u += x_hat - z

            history._objval[k] = self._objective(A, b, lambdaUse, x, z)
            history._r_norm[k] = LA.norm(x-z)
            history._s_norm[k] = LA.norm(-rho*(z-zold))
            history._eps_pri[k] = (np.sqrt(n)*ABSTOL 
                + RELTOL*np.max((LA.norm(x), LA.norm(-z))))
            history._eps_dual[k] = np.sqrt(n)*ABSTOL + RELTOL*LA.norm(rho*u)

            if (history._r_norm[k] < history._eps_pri[k] and 
                history._s_norm[k] < history._eps_dual[k]):
                break
        return np.squeeze(z)

    def fit(self, X):
        # X must be data in the shape of (sensors, time)
        X = np.array(X,dtype='float')
        self._numCh, self._K = np.shape(X)
        if np.size(self._numInp) == 0:
            self._numInp = int(np.floor(self._numCh/2))
        self._AMat = np.empty((self._niter+1, self._numCh, self._numCh))
        self._u = np.zeros((self._numInp,self._K))
        try:
            if self._numCh == 1:
                raise Exception('oneSensor')
            if self._K < self._numCh:
                raise Exception('lessData')
            if np.size(self._BMat)>0:
                if np.shape(self._BMat) != (self._numCh, self._numInp):
                    raise Exception('BMatDim')
            
            self._estimateOrder(X)
            self._updateZVec(X)
            self._AMat[0,:,:], mse = self._performLeastSq(self._zVec.T, X.T)

            if np.size(self._BMat) == 0:
                self._setHeuristicBMat(self._AMat[0,:,:])

#           initiate precomputed variables process, 
#           compute all variable need to be computed exactly, again and again.
            self._preComputedVars = self._preComputedVars_()
            self._preComputedVars._updateLassoLUMat(self._BMat, 1/self._lambdaUse)

            t0 = time.time()
            if self.verbose > 0:
                print('beginning mse = %f'%(mse))
            mseIter = np.empty((self._niter+1,))
            mseIter[0] = mse
            for iterInd in range(self._niter):
                for kInd in range(1,self._K):
                    yUse = self._zVec[:,kInd] - np.matmul(self._AMat[iterInd,:,:],
                            X[:,kInd-1])
                    self._u[:,kInd] = self._getLassoSoln(yUse, self._lambdaUse)
                    # clf = linear_model.Lasso(alpha=self._lambdaUse)
                    # clf.fit(self._BMat * np.sqrt(self._numCh), yUse* np.sqrt(self._numCh))
                    # self._u[:,kInd] = clf.coef_

                self._AMat[iterInd+1,:,:],mseIter[iterInd+1] = self._performLeastSq(
                    (self._zVec - np.matmul(self._BMat, self._u)).T, X.T)
                if self.verbose>0:
                    print('iter ind = %d, mse = %f'%(iterInd, mseIter[iterInd+1]))     
            print('time taken = %f'%(time.time()-t0))

        except Exception as err:
            errStatus = err.args[0]
            if errStatus == 'oneSensor':
                print('The number of sensors must be > 1, retry...')
            elif errStatus == 'lessData':
                print('The number of data points are less than number of sensors, retry...')
            elif errStatus == 'BMatDim':
                print('size of B should be consistent with the number of channels and number of inputs')
            else:
                print('some different error')
        
        
        