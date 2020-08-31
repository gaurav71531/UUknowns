import numpy as np
from scipy.special import gamma
from copy import deepcopy

class arfima(object):
    def __init__(self, Ap = [], Aq = [], d = 0, infit=100):
        self.Ap = Ap
        self.Aq = Aq
        self.order = d
        self.infit = infit
        
        if not isinstance(Ap, list):
            raise TypeError('Ap has to be entered in a single list form')
        if not isinstance(Aq, list):
            raise TypeError('Aq has to be entered in a single list form')
        if d < -0.5 or d>0.5:
            raise Exception('d has to be in the range of [-0.5, 0.5]')
        
    def gen(self, T = 1000, noise_var = 1):
        p = len(self.Ap)
        q = len(self.Aq)
        
        et = np.random.randn(T,) * np.sqrt(noise_var)
        if self.order == 0:
            if p == 0 and q == 0:
                return et
            if p == 0:
                Aq = np.array([1] + self.Aq)
                moving_average_ = np.convolve(et, Aq)[:T]
                return moving_average_
            preFactVec = []
        else:
            j = np.arange(0,self.infit+1)
            preFactVec = gamma(-self.order+j)/gamma(-self.order) / gamma(j+1)
                
        if q == 0:
             moving_average_ = et
        else:
            Aq = np.array([1] + self.Aq)
            moving_average_ = np.convolve(et, Aq)
        
        if p == 0:
            preFactVec = preFactVec[1:]
            x = np.zeros((T,))
            x[0] = moving_average_[0]
            for t in range(1, T):
                x[t] += - self.getFilteredStride(x, preFactVec, t) + moving_average_[t]
        else:
            y = np.zeros((T,))
            y[0] = moving_average_[0]
            Ap = np.array(self.Ap)
            for t in range(1, T):
                y[t] += -self.getFilteredStride(y, Ap, t) + moving_average_[t]
            if len(preFactVec) == 0:
                return y
            
            preFactVec = preFactVec[1:]
            x = np.zeros_like(y)
            x[0] = y[0]
            for t in range(1, T):
                x[t] += -self.getFilteredStride(x, preFactVec, t) + y[t]
        return x
                
    def getFilteredStride(self, x, preFact, t):
        if len(preFact)>=t:
            return np.dot(x[:t], preFact[t-1::-1])
        else:
            return np.dot(x[t-len(preFact):t], preFact[::-1]) 